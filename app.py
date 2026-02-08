# app.py (Full Updated Version)

import os
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import librosa
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import xgboost as xgb
from rag_service import RAGService  
from fastapi import Body

class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = ""

# ============================================
# CONFIGURATION
# ============================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {DEVICE}")

# Add root to sys.path for config import
import sys
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from config import EMOTIONS, AUDIO_CONFIG, TEXT_CONFIG, MODEL_PATHS

class ResidualAttentionBlock(nn.Module):
    """
    Advanced Attention with Residual Connection to prevent signal loss
    """
    def __init__(self, input_dim=7, hidden_dim=64):
        super(ResidualAttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
        # Output projection back to input_dim for residual
        self.proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x shape: [batch, 3, 7]
        residual = x
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_weights, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = self.proj(out)
        
        # Add & Norm
        return self.norm(out + residual), attn_weights

class AttentionFusionModel(nn.Module):
    def __init__(self, num_emotions=7, hidden_dim=128):
        super(AttentionFusionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_block = ResidualAttentionBlock(input_dim=num_emotions, hidden_dim=hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_emotions * 3, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_emotions)
        )

    def forward(self, text_logits, audio_logits, video_logits):
        # Stack into [batch, 3, 7]
        x = torch.stack([text_logits, audio_logits, video_logits], dim=1)
        
        # Apply Residual Attention
        fused_feat, weights = self.attn_block(x)
        
        # Classify
        logits = self.classifier(fused_feat)
        return logits, weights

# ============================================
# MENTAL HEALTH ANALYZER (SYNTHESIS LAYER)
# ============================================

class MentalHealthAnalyzer:
    """
    Interprets multimodal patterns to provide deeper psychological insights.
    Moves from "Emotion Labels" to "Well-being Indicators".
    """
    @staticmethod
    def analyze_patterns(results):
        text = results.get('text', {})
        audio = results.get('audio', {})
        video = results.get('video', {})
        fusion = results.get('fusion', {})
        
        findings = []
        cluster = "Stable"
        
        # 1. Congruence & Masking Analysis
        text_emo = text.get('emotion')
        audio_emo = audio.get('emotion')
        video_emo = video.get('emotion')
        
        # Filter out None values to see what we actually have
        active_emotions = [e for e in [text_emo, audio_emo, video_emo] if e]
        
        if not active_emotions:
             return {
                "findings": ["No clear emotional signal detected from inputs."],
                "wellbeing_cluster": "Indeterminate",
                "description": "Waiting for user input to generate insights."
            }

        # Check if voice/expression reveal distress hidden in text
        if text_emo in ['neutral', 'happy'] and (audio_emo in ['sad', 'fear'] or video_emo in ['sad', 'fear']):
            findings.append("Emotional Suppression: User is using stable language while voice or expression indicates underlying distress.")
            cluster = "Masked Distress"
        elif text_emo and audio_emo and video_emo and (text_emo == audio_emo == video_emo):
            findings.append(f"High Congruence: Multimodal signals strongly align on {text_emo.upper()} state.")
            cluster = "Harmonized " + text_emo.capitalize()
        elif len(set(active_emotions)) == 1:
             # Case where we have 1 or 2 modalities and they agree
             findings.append(f"Consistent Signals: Detected {active_emotions[0]} state across inputs.")
             cluster = active_emotions[0].capitalize()
        else:
            findings.append("Mixed Signals: Differing states between verbal and non-verbal cues detected.")
            cluster = "Fluctuating State"

        # 2. Risk Indicators
        high_risk_emotions = ['sad', 'fear', 'disgust']
        distress_count = sum(1 for e in active_emotions if e in high_risk_emotions)
        
        if distress_count >= 2:
            cluster = "Significant Distress"
        elif 'angry' in active_emotions:
            cluster = "Agitated/Stressed"
            
        return {
            "findings": findings,
            "wellbeing_cluster": cluster,
            "description": f"Overall state shows {cluster} patterns."
        }

# ============================================
# TEXT MODEL
# ============================================

class TextModelWrapper:
    def __init__(self):
        self.model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(EMOTIONS)
        ).to(DEVICE)
        
        if os.path.exists(MODEL_PATHS['text']):
            self.model.load_state_dict(torch.load(MODEL_PATHS['text'], map_location=DEVICE))
            print("✓ Loaded Text Model")
        else:
            print("⚠️ Text Model not found, using untrained weights")
            
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()[0]

# ============================================
# AUDIO MODEL
# ============================================

class AudioModelWrapper:
    def __init__(self):
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CONFIG['model_name'])
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_CONFIG['model_name'], 
            num_labels=len(EMOTIONS),
            ignore_mismatched_sizes=True
        ).to(DEVICE)
        
        # Load the fine-tuned weights
        FINE_TUNED_AUDIO = "models/audio_model_90_acc_finetuned.pt"
        if os.path.exists(FINE_TUNED_AUDIO):
            self.model.load_state_dict(torch.load(FINE_TUNED_AUDIO, map_location=DEVICE))
            print("✓ Loaded Audio Model")
        else:
            print("⚠️ Audio Model not found")
            
        self.model.eval()

    def predict(self, y, sr):
        inputs = self.processor(y, sampling_rate=sr, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs.cpu().numpy()[0]

# ============================================
# VIDEO MODEL
# ============================================

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class VideoModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        # Match training architecture: ConvNeXt-Tiny
        convnext = models.convnext_tiny(weights=None)
        self.features = convnext.features
        self.spatial_attn = SpatialAttention()
        
        in_features = 768  # ConvNeXt-Tiny output channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Original head (matched to weights)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, len(EMOTIONS))
        )
        
        self.to(DEVICE)
        
        # Load the 70% model for feature extraction
        TRAINED_VIDEO_WEIGHTS = "models/trained_models/video_model.pth"
        if os.path.exists(TRAINED_VIDEO_WEIGHTS):
            self.load_state_dict(torch.load(TRAINED_VIDEO_WEIGHTS, map_location=DEVICE))
            print("✓ Loaded Base Video Model for Embeddings")
        
        # Load XGBoost Classifier (if exists, optional)
        XGB_VIDEO_PATH = "models/video_xgboost.model"
        if os.path.exists(XGB_VIDEO_PATH):
            self.xgb_model = joblib.load(XGB_VIDEO_PATH)
            print("✓ Loaded Video XGBoost Classifier")
        else:
            self.xgb_model = None
            # print("⚠️ Video XGBoost Model not found")
            
        self.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # 1. Extract Embeddings
            x = self.features(img_tensor)
            attn = self.spatial_attn(x)
            x = x * attn
            x = self.global_pool(x)
            pooled_output = torch.flatten(x, 1).cpu().numpy()
            
            # 2. Predict with XGBoost
            if self.xgb_model:
                probs = self.xgb_model.predict_proba(pooled_output)[0]
                return probs
            else:
                # Fallback to pure deep learning head
                x_head = self.global_pool(x * attn) # Re-apply global pool for head consistency?
                # Wait, the head expects flattened input.
                # Let's check previous implementation.
                # Previous implementation (step 3978, line 335): outputs = self.head(x) where x is global pooled?
                # Line 326: x = self.global_pool(x).
                # So x is [batch, 768, 1, 1].
                # Head starts with Flatten.
                outputs = self.head(x)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                return probs.cpu().numpy()[0]

app = FastAPI()

# Global models
text_model = None
audio_model = None
video_model = None
fusion_model = None
fusion_scaler = None
rag_service = None

@app.on_event("startup")
async def startup_event():
    global text_model, audio_model, video_model, fusion_model, fusion_scaler, rag_service
    
    print("\nINITIALIZING MODELS...")
    text_model = TextModelWrapper()
    audio_model = AudioModelWrapper()
    video_model = VideoModelWrapper()
    
    # Fusion model loading
    FUSION_PATH = "models/trained_models/fusion_model.pth"
    if os.path.exists(FUSION_PATH):
        fusion_model = AttentionFusionModel(num_emotions=len(EMOTIONS)).to(DEVICE)
        fusion_model.load_state_dict(torch.load(FUSION_PATH, map_location=DEVICE))
        fusion_model.eval()
        print("✓ Loaded Attention Fusion Model")
    else:
        print("⚠️ Fusion Model not found (Training Pending)")
        
    print("🚀 Initializing RAG Service...")
    try:
        rag_service = RAGService()
    except Exception as e:
        print(f"Error initializing RAG Service: {str(e)}")
        rag_service = None

@app.post("/chat")
async def chat(
    payload: dict = Body(...),   # accept any JSON body as dict
):
    message = payload.get("message", "")
    emotion = payload.get("emotion", "")  # defaults to empty string if missing
    
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not available")
    
    if not message.strip():
        raise HTTPException(status_code=400, detail="Message field is required and cannot be empty")
    
    try:
        response_text = rag_service.generate_chat_response(
            user_message=message,
            emotion_context=emotion
        )
        return {"response": response_text}
    except Exception as e:
        print(f"Chat generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    # 1. Text Prediction
    if text:
        text_probs = text_model.predict(text)
        results['text'] = {
            'emotion': EMOTIONS[np.argmax(text_probs)],
            'confidence': float(np.max(text_probs)),
            'probs': {e: float(p) for e, p in zip(EMOTIONS, text_probs)}
        }
        probs_list.append(text_probs)
    else:
        probs_list.append(np.zeros(len(EMOTIONS)))
        
    # 2. Audio Prediction
    if audio:
        try:
            contents = await audio.read()
            with open("temp.wav", "wb") as f:
                f.write(contents)
            
            y, sr = librosa.load("temp.wav", sr=16000)
            audio_probs = audio_model.predict(y, sr)
            
            results['audio'] = {
                'emotion': EMOTIONS[np.argmax(audio_probs)],
                'confidence': float(np.max(audio_probs)),
                'probs': {e: float(p) for e, p in zip(EMOTIONS, audio_probs)}
            }
            probs_list.append(audio_probs)
            
            os.remove("temp.wav")
        except Exception as e:
            print(f"Error processing audio: {e}")
            probs_list.append(np.zeros(len(EMOTIONS)))
    else:
        probs_list.append(np.zeros(len(EMOTIONS)))
            
    # 3. Video Prediction
    if video:
        try:
            contents = await video.read()
            with open("temp_video.mp4", "wb") as f:
                f.write(contents)
                
            cap = cv2.VideoCapture("temp_video.mp4")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            sample_indices = np.linspace(0, total_frames - 1, 5, dtype=int)
            frame_probs = []
            
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_probs.append(video_model.predict(frame))
            
            cap.release()
            os.remove("temp_video.mp4")
            
            if frame_probs:
                video_probs = np.mean(frame_probs, axis=0)
                results['video'] = {
                    'emotion': EMOTIONS[np.argmax(video_probs)],
                    'confidence': float(np.max(video_probs)),
                    'probs': {e: float(p) for e, p in zip(EMOTIONS, video_probs)}
                }
                probs_list.append(video_probs)
            else:
                probs_list.append(np.zeros(len(EMOTIONS)))
        except Exception as e:
            print(f"Error processing video: {e}")
            probs_list.append(np.zeros(len(EMOTIONS)))
    else:
        probs_list.append(np.zeros(len(EMOTIONS)))

    # 4. Fusion
    if fusion_model:
        t_feat = torch.tensor(probs_list[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        a_feat = torch.tensor(probs_list[1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        v_feat = torch.tensor(probs_list[2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            fused_logits, weights = fusion_model(t_feat, a_feat, v_feat)
            final_probs = torch.softmax(fused_logits, dim=1).cpu().numpy()[0]
            attn_weights = weights[0].cpu().numpy() 
            modality_importance = attn_weights.mean(axis=0) 
        
        final_emotion = EMOTIONS[np.argmax(final_probs)]
        results['fusion'] = {
            'emotion': final_emotion,
            'confidence': float(np.max(final_probs)),
            'probs': {e: float(p) for e, p in zip(EMOTIONS, final_probs)},
            'modality_weights': {
                'text': float(modality_importance[0]),
                'audio': float(modality_importance[1]),
                'video': float(modality_importance[2])
            }
        }
    else:
        valid_probs = [p for p in probs_list if np.sum(p) > 0]
        if valid_probs:
            avg_probs = np.mean(valid_probs, axis=0)
            results['fusion'] = {
                'emotion': EMOTIONS[np.argmax(avg_probs)],
                'confidence': float(np.max(avg_probs)),
                'probs': {e: float(p) for e, p in zip(EMOTIONS, avg_probs)},
                'note': 'Simple Average (Attention model not trained yet)'
            }

    # 5. Mental Health Synthesis
    synthesis = MentalHealthAnalyzer.analyze_patterns(results)
    results['synthesis'] = synthesis

    # 6. RAG Response
    if rag_service:
        try:
            emotion = results['fusion']['emotion']
            confidence = results['fusion']['confidence']
            insight = rag_service.generate_insight(emotion, confidence, additional_context=text or "")
            results['response'] = {
                'answer': insight,
                'resources': ["Curated Mental Health DB"]
            }
            
            # Update synthesis for UI
            results['synthesis'] = {
                'wellbeing_cluster': "Clinical Insight",
                'findings': ["Analysis based on RAG Knowledge Base"],
                'description': insight
            }
        except Exception as e:
            print(f"Error in RAG synthesis: {e}")
            results['response'] = {'answer': 'Guardian synthesis halted.', 'resources': []}
    else:
        results['response'] = {'answer': 'RAG Service unavailable.', 'resources': []}
    
    return results

@app.post("/chat")
async def chat(request: ChatRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not available")
    response = rag_service.generate_chat_response(request.message, emotion_context=request.emotion)
    return {"response": response}

@app.get("/")
def root():
    return {"message": "Multimodal Mental Health Prediction API is Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)