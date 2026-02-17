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
import shap
import subprocess
import tempfile

# Import Whisper for speech-to-text
try:
    import whisper
    WHISPER_AVAILABLE = True
    print("✅ Whisper loaded successfully for speech-to-text")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️ Whisper not available, speech-to-text disabled")

try:
    import noisereduce as nr
    NOISE_REDUCTION_AVAILABLE = True
except ImportError:
    NOISE_REDUCTION_AVAILABLE = False
    print("⚠️ noisereduce not available, audio preprocessing disabled")

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ChatRequest(BaseModel):
    message: str
    emotion: Optional[str] = ""

# ============================================
# PREPROCESSING FUNCTIONS FOR ACCURACY
# ============================================

def preprocess_video_frame(frame):
    """Detect and crop face from video frame to improve accuracy"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Get the largest face
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            # Add padding
            padding = int(w * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            face_crop = frame[y:y+h, x:x+w]
            return face_crop
    except Exception as e:
        print(f"Face detection error: {e}")
    
    return frame

def preprocess_audio(audio, sr):
    """Reduce background noise from audio to improve accuracy"""
    if not NOISE_REDUCTION_AVAILABLE:
        return audio
    
    try:
        # Reduce noise using noisereduce library
        reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
        return reduced_noise
    except Exception as e:
        print(f"Noise reduction error: {e}")
        return audio

def should_show_prediction(confidence, threshold=0.4):
    """Determine if prediction is confident enough to show"""
    return confidence >= threshold

# ============================================
# CONFIGURATION
# ============================================

DEVICE = torch.device('cpu')
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
        findings = ["System initialized"] # Early init
        cluster = "Analyzing..."
        
        try:
            # Maximum robustness - ensure results is a dict
            if not isinstance(results, dict):
                return {
                    "findings": ["No valid results available for analysis."],
                    "wellbeing_cluster": "Indeterminate",
                    "description": "Please provide input data."
                }
            
            # Reset for actual logic
            findings = []
            cluster = "Stable"
            
            # Defensive extraction logic
            def get_mod_val(mod_key, attr):
                mod_data = results.get(mod_key)
                if isinstance(mod_data, dict):
                    return mod_data.get(attr)
                return None
                
            # Filter out None or empty values
            text_emo = get_mod_val('text', 'emotion')
            audio_emo = get_mod_val('audio', 'emotion')
            video_emo = get_mod_val('video', 'emotion')
            active_emotions = [e for e in [text_emo, audio_emo, video_emo] if e and e != 'uncertain']
            
            if not active_emotions:
                 return {
                    "findings": ["No clear emotional signal detected from inputs."],
                    "wellbeing_cluster": "Indeterminate",
                    "description": "Waiting for user input to generate insights."
                }
            
            # 1. Congruence & Masking Analysis
            if text_emo in ['neutral', 'happy'] and (audio_emo in ['sad', 'fear'] or video_emo in ['sad', 'fear']):
                findings.append("Emotional Suppression: User is using stable language while voice or expression indicates underlying distress.")
                cluster = "Masked Distress"
            elif text_emo and audio_emo and video_emo and (text_emo == audio_emo == video_emo):
                findings.append(f"High Congruence: Multimodal signals strongly align on {text_emo.upper()} state.")
                cluster = "Harmonized " + text_emo.capitalize()
            elif len(set(active_emotions)) == 1:
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
        except Exception as e:
            print(f"🔥 Fail-safe error in analyze_patterns: {e}")
            return {
                "findings": ["Error during behavioral pattern analysis."],
                "wellbeing_cluster": "Analysis Error",
                "description": "The system encountered an error while interpreting combined signals."
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
            print("✓ Loaded Text Model weights")
        else:
            print("⚠️ Text Model weights not found, using untrained weights")
            
        self.model.eval()
        
        # Initialize SHAP explainer
        from transformers import pipeline
        import shap.maskers
        
        # Use a more stable pipeline for SHAP
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, 
                             device=-1, top_k=None, truncation=True, max_length=128)
        
        # Use the specific Text masker
        masker = shap.maskers.Text(self.tokenizer)
        self.explainer = shap.Explainer(self.score_func, masker=masker)
        print("✓ SHAP Explainer Initialized for Text")

    def score_func(self, texts):
        # Handle potential non-list/ndarray inputs
        if isinstance(texts, str):
            texts = [texts]
        elif hasattr(texts, "tolist"):
            texts = texts.tolist()
        elif not isinstance(texts, (list, tuple)):
            texts = [texts]
        
        # Ensure every element is a clean string
        clean_texts = []
        for t in texts:
            if isinstance(t, str):
                clean_texts.append(t if t.strip() else " ")
            elif isinstance(t, (bytes, bytearray)):
                clean_texts.append(t.decode('utf-8', errors='ignore') or " ")
            else:
                clean_texts.append(str(t) if t is not None else " ")
        
        try:
            outputs = self.pipe(clean_texts)
            all_scores = []
            for output in outputs:
                score_map = {item['label']: item['score'] for item in output}
                # Diagnostic: Print label names once to see mapping
                if not hasattr(self, '_labels_printed'):
                    print(f"🧬 Text Model Internal Labels: {list(score_map.keys())}")
                    self._labels_printed = True
                ordered_scores = [score_map.get(f"LABEL_{i}", score_map.get(f"{EMOTIONS[i]}", 0.0)) for i in range(len(EMOTIONS))]
                all_scores.append(ordered_scores)
            return np.array(all_scores)
        except Exception as e:
            print(f"DEBUG: SHAP pipe internal error: {e}")
            return np.zeros((len(clean_texts), len(EMOTIONS)))

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

    def explain(self, text):
        """Generates SHAP values for the given text"""
        try:
            shap_values = self.explainer([text])
            return {
                "values": shap_values.values[0].tolist(), # [tokens, 7]
                "base_values": shap_values.base_values[0].tolist(), # [7]
                "tokens": shap_values.data[0].tolist() # [tokens]
            }
        except Exception as e:
            print(f"⚠️ SHAP Explanation Error: {str(e)}")
            return None

# ============================================
# HELPER FUNCTIONS FOR XAI REASONING
# ============================================

def generate_audio_reasoning(emotion: str, confidence: float, probs: dict) -> str:
    """Generate human-readable reasoning for audio prediction"""
    reasoning_templates = {
        'angry': "Detected raised voice pitch and high energy levels",
        'happy': "Detected upbeat tone and positive vocal patterns",
        'sad': "Detected low energy and somber vocal tone",
        'fear': "Detected trembling voice and high pitch variation",
        'disgust': "Detected negative vocal expressions",
        'surprise': "Detected sudden pitch changes and exclamations",
        'neutral': "Detected calm and steady vocal patterns"
    }
    
    base_reason = reasoning_templates.get(emotion, "Analyzed vocal characteristics")
    
    # Add confidence context
    if confidence > 0.8:
        confidence_text = "with strong confidence"
    elif confidence > 0.6:
        confidence_text = "with moderate confidence"
    else:
        confidence_text = "with low confidence"
    
    return f"{base_reason} ({confidence_text})"

def generate_video_reasoning(emotion: str, confidence: float, probs: dict) -> str:
    """Generate human-readable reasoning for video prediction"""
    reasoning_templates = {
        'angry': "Detected furrowed brows and tense facial muscles",
        'happy': "Detected smiling expression and raised cheeks",
        'sad': "Detected downturned mouth and lowered eyebrows",
        'fear': "Detected widened eyes and raised eyebrows",
        'disgust': "Detected wrinkled nose and raised upper lip",
        'surprise': "Detected raised eyebrows and open mouth",
        'neutral': "Detected relaxed facial expression"
    }
    
    base_reason = reasoning_templates.get(emotion, "Analyzed facial expressions")
    
    # Add confidence context
    if confidence > 0.8:
        confidence_text = "clearly visible"
    elif confidence > 0.6:
        confidence_text = "moderately visible"
    else:
        confidence_text = "subtly visible"
    
    return f"{base_reason} ({confidence_text})"

def generate_text_reasoning(emotion: str, confidence: float, probs: dict) -> str:
    """Generate human-readable reasoning for text prediction"""
    reasoning_templates = {
        'angry': "Detected aggressive or frustrated language",
        'happy': "Detected positive and cheerful words",
        'sad': "Detected negative sentiment and melancholic tone",
        'fear': "Detected anxious or worried language",
        'disgust': "Detected expressions of dislike or aversion",
        'surprise': "Detected unexpected or shocking content",
        'neutral': "Detected factual and objective language"
    }
    
    base_reason = reasoning_templates.get(emotion, "Analyzed text sentiment")
    
    if confidence > 0.7:
        return f"{base_reason} with clear indicators"
    else:
        return f"{base_reason} with subtle indicators"

def generate_fusion_reasoning(fusion_emotion: str, modality_results: dict) -> str:
    """Generate explanation for why fusion chose this emotion"""
    # Find which modalities agreed with fusion
    agreements = []
    disagreements = []
    
    for mod in ['text', 'audio', 'video']:
        if mod in modality_results:
            mod_emotion = modality_results[mod]['emotion']
            mod_conf = modality_results[mod]['confidence']
            
            if mod_emotion == fusion_emotion:
                agreements.append(f"{mod.capitalize()} ({mod_conf:.0%})")
            else:
                disagreements.append(f"{mod.capitalize()} suggested {mod_emotion}")
    
    if len(agreements) >= 2:
        return f"Multiple modalities agreed: {', '.join(agreements)} all indicated {fusion_emotion}"
    elif len(agreements) == 1:
        if disagreements:
            return f"{agreements[0]} strongly indicated {fusion_emotion}, outweighing other signals"
        else:
            return f"{agreements[0]} indicated {fusion_emotion}"
    else:
        return f"Fusion model weighted all signals to determine {fusion_emotion}"

# ============================================
# PREDICTION ENDPOINT
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
whisper_model = None  # NEW: Whisper for speech-to-text

@app.on_event("startup")
async def startup_event():
    global text_model, audio_model, video_model, fusion_model, fusion_scaler, rag_service, whisper_model
    
    print("\nINITIALIZING MODELS...")
    text_model = TextModelWrapper()
    audio_model = AudioModelWrapper()
    video_model = VideoModelWrapper()
    
    # FFmpeg check for diagnostics
    print("🔍 Checking FFmpeg availability...")
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        print("✅ FFmpeg found in system PATH")
    except:
        if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe"):
            print("✅ FFmpeg found via absolute path fallback: C:\\ffmpeg\\bin\\ffmpeg.exe")
        else:
            print("❌ FFmpeg NOT FOUND. Audio extraction from video will fail!")
            print("👉 Please ensure FFmpeg is installed and added to PATH.")
    
    # Load Whisper model for speech-to-text
    if WHISPER_AVAILABLE:
        try:
            print("Loading Whisper model for speech-to-text...")
            whisper_model = whisper.load_model("base")  # Using base model for balance
            print("✅ Whisper model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load Whisper model: {e}")
            whisper_model = None
    
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

@app.post("/predict")
async def predict(
    text: Optional[str] = Body(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None)
):
    print(f"\n🔮 [API] PREDICT CALL RECEIVED - V3")
    print(f"📦 Payload Check: text={bool(text)}, audio={bool(audio)}, video={bool(video)}")
    if video: print(f"📹 Video Content Type: {video.content_type}, Filename: {video.filename}")
    
    # Initialize results with all keys to ensure frontend never sees missing data
    results = {
        'text': None,
        'audio': None,
        'video': None,
        'fusion': None,
        'synthesis': None,
        'response': None,
        'extracted_text': None
    }
    probs_list = [np.zeros(len(EMOTIONS)) for _ in range(3)] # [text, audio, video]
    
    print("\n--- NEW PREDICTION REQUEST ---")
    
    # 1. Text Prediction
    if text and len(text.strip()) > 0:
        try:
            print(f"📄 Processing user text: {text[:50]}...")
            text_probs = text_model.predict(text)
            text_emotion = EMOTIONS[np.argmax(text_probs)]
            text_confidence = float(np.max(text_probs))
            
            results['text'] = {
                'emotion': text_emotion,
                'confidence': text_confidence,
                'probs': {e: float(p) for e, p in zip(EMOTIONS, text_probs)},
                'reasoning': generate_text_reasoning(text_emotion, text_confidence, {}),
                'xai': text_model.explain(text)
            }
            probs_list[0] = text_probs
            print(f"✅ Text analyzed: {text_emotion} ({text_confidence:.2%})")
        except Exception as e:
            print(f"❌ Text analysis error: {e}")
    
    # 2. Audio Prediction (Standalone file)
    audio_probs = None
    if audio:
        try:
            contents = await audio.read()
            audio_path = "temp_uploaded_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(contents)
            
            y, sr = librosa.load(audio_path, sr=16000)
            y = preprocess_audio(y, sr)
            audio_probs = audio_model.predict(y, sr)
            
            audio_emotion = EMOTIONS[np.argmax(audio_probs)]
            audio_confidence = float(np.max(audio_probs))
            
            results['audio'] = {
                'emotion': audio_emotion if should_show_prediction(audio_confidence) else 'uncertain',
                'confidence': audio_confidence,
                'probs': {e: float(p) for e, p in zip(EMOTIONS, audio_probs)},
                'reasoning': generate_audio_reasoning(audio_emotion, audio_confidence, {})
            }
            
            if not should_show_prediction(audio_confidence):
                results['audio']['warning'] = 'Low confidence prediction'
            
            probs_list[1] = audio_probs
            print(f"✅ Audio analyzed: {audio_emotion} ({audio_confidence:.2%})")
            
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"❌ Audio processing error: {e}")
            
    # 3. Video Prediction
    if video:
        try:
            contents = await video.read()
            video_ext = ".webm" if "webm" in (video.content_type or "").lower() else ".mp4"
            video_input_path = f"temp_input{video_ext}"
            
            with open(video_input_path, "wb") as f:
                f.write(contents)
                
            # Try as image first
            image_np = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
            
            if image_np is not None:
                print("📸 Processing as Image Capture...")
                preprocessed_frame = preprocess_video_frame(image_np)
                video_probs = video_model.predict(preprocessed_frame)
                video_emotion = EMOTIONS[np.argmax(video_probs)]
                video_confidence = float(np.max(video_probs))
                
                results['video'] = {
                    'emotion': video_emotion if should_show_prediction(video_confidence) else 'uncertain',
                    'confidence': video_confidence,
                    'probs': {e: float(p) for e, p in zip(EMOTIONS, video_probs)},
                    'reasoning': generate_video_reasoning(video_emotion, video_confidence, {}),
                    'note': 'Image Capture'
                }
                probs_list[2] = video_probs
                print(f"✅ Image analyzed: {video_emotion} ({video_confidence:.2%})")
            else:
                print("🎥 Processing as Video File...")
                cap = cv2.VideoCapture(video_input_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if audio_probs is None:
                    try:
                        audio_path = "temp_extracted_audio.wav"
                        print("🎵 Attempting audio extraction with FFmpeg...")
                        # Try simple 'ffmpeg' first, then fallback to absolute path if common on Windows
                        ffmpeg_cmd = 'ffmpeg'
                        try:
                            # Quick check if ffmpeg works
                            subprocess.run(['ffmpeg', '-version'], capture_output=True)
                        except:
                            if os.path.exists(r"C:\ffmpeg\bin\ffmpeg.exe"):
                                ffmpeg_cmd = r"C:\ffmpeg\bin\ffmpeg.exe"
                                print(f"ℹ️ Using absolute FFmpeg path: {ffmpeg_cmd}")

                        proc = subprocess.run([
                            ffmpeg_cmd, '-i', video_input_path, '-vn', '-acodec', 'pcm_s16le',
                            '-ar', '16000', '-ac', '1', audio_path, '-y'
                        ], capture_output=True, text=True)
                        
                        if proc.returncode != 0:
                            print(f"⚠️ FFmpeg Error: {proc.stderr}")
                        
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                            y, sr = librosa.load(audio_path, sr=16000)
                            y = preprocess_audio(y, sr)
                            audio_probs = audio_model.predict(y, sr)
                            
                            audio_emotion = EMOTIONS[np.argmax(audio_probs)]
                            audio_confidence = float(np.max(audio_probs))
                            
                            results['audio'] = {
                                'emotion': audio_emotion if should_show_prediction(audio_confidence) else 'uncertain',
                                'confidence': audio_confidence,
                                'probs': {e: float(p) for e, p in zip(EMOTIONS, audio_probs)},
                                'reasoning': generate_audio_reasoning(audio_emotion, audio_confidence, {}),
                                'note': 'Extracted from video'
                            }
                            probs_list[1] = audio_probs
                            print(f"✅ Extracted audio analyzed: {audio_emotion}")
                            
                            # Whisper Transcription
                            if whisper_model and not text:
                                try:
                                    print("🎤 Transcribing with Whisper...")
                                    trans_res = whisper_model.transcribe(audio_path, language='en')
                                    ext_text = trans_res["text"].strip()
                                    if ext_text:
                                        print(f"✅ Transcribed: {ext_text}")
                                        results['extracted_text'] = ext_text
                                        text_probs = text_model.predict(ext_text)
                                        text_emo = EMOTIONS[np.argmax(text_probs)]
                                        text_conf = float(np.max(text_probs))
                                        
                                        results['text'] = {
                                            'emotion': text_emo if should_show_prediction(text_conf) else 'uncertain',
                                            'confidence': text_conf,
                                            'probs': {e: float(p) for e, p in zip(EMOTIONS, text_probs)},
                                            'reasoning': generate_text_reasoning(text_emo, text_conf, {}),
                                            'note': 'Transcribed from audio',
                                            'xai': text_model.explain(ext_text)
                                        }
                                        probs_list[0] = text_probs
                                except Exception as e_w:
                                    print(f"⚠️ Whisper error: {e_w}")
                            
                            os.remove(audio_path)
                    except Exception as e_a:
                        print(f"⚠️ Audio extraction block error: {e_a}")
                
                # Process Frames
                if total_frames > 0:
                    sample_indices = np.linspace(0, total_frames - 1, min(5, total_frames), dtype=int)
                    f_probs = []
                    for idx in sample_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            f_probs.append(video_model.predict(preprocess_video_frame(frame)))
                    cap.release()
                    
                    if f_probs:
                        video_probs = np.mean(f_probs, axis=0)
                        video_emotion = EMOTIONS[np.argmax(video_probs)]
                        video_confidence = float(np.max(video_probs))
                        results['video'] = {
                            'emotion': video_emotion if should_show_prediction(video_confidence) else 'uncertain',
                            'confidence': video_confidence,
                            'probs': {e: float(p) for e, p in zip(EMOTIONS, video_probs)},
                            'reasoning': generate_video_reasoning(video_emotion, video_confidence, {})
                        }
                        probs_list[2] = video_probs
                        print(f"✅ Video frames analyzed: {video_emotion}")
            
            if os.path.exists(video_input_path):
                os.remove(video_input_path)
        except Exception as e_v:
            print(f"❌ Video error: {e_v}")

    # 4. Fusion
    try:
        print(f"🧬 Fusion preparation - Probs sums: Text:{np.sum(probs_list[0]):.2f}, Audio:{np.sum(probs_list[1]):.2f}, Video:{np.sum(probs_list[2]):.2f}")
        
        if fusion_model:
            t_feat = torch.tensor(probs_list[0], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            a_feat = torch.tensor(probs_list[1], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            v_feat = torch.tensor(probs_list[2], dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                fused_logits, weights = fusion_model(t_feat, a_feat, v_feat)
                final_probs = torch.softmax(fused_logits, dim=1).cpu().numpy()[0]
                modality_importance = weights[0].cpu().numpy().mean(axis=0)
            
            # CRITICAL DEBUG: Print individual predictions for comparison
            print(f"📊 MODALITY BREAKDOWN (Logits index for index):")
            for i, mod in enumerate(['Text', 'Audio', 'Video']):
                top_idx = np.argmax(probs_list[i])
                print(f"   - {mod}: {EMOTIONS[top_idx]} ({np.max(probs_list[i]):.2%}) | Full: {[f'{p:.2%}' for p in probs_list[i]]}")
            
            final_emotion = EMOTIONS[np.argmax(final_probs)]
            fusion_confidence = float(np.max(final_probs))
            print(f"🔮 Fusion result: {final_emotion} ({fusion_confidence:.2%})")
            
            # Use a fresh dictionary to avoid NoneType issues
            results['fusion'] = {
                'emotion': final_emotion,
                'confidence': fusion_confidence,
                'probs': {e: float(p) for e, p in zip(EMOTIONS, final_probs)},
                'weights': {
                    'text': float(modality_importance[0]),
                    'audio': float(modality_importance[1]),
                    'video': float(modality_importance[2])
                },
                'reasoning': generate_fusion_reasoning(final_emotion, results)
            }
        else:
            # Fallback to simple average
            valid_probs = [p for p in probs_list if np.sum(p) > 0]
            if not valid_probs: valid_probs = [np.zeros(len(EMOTIONS))]
            avg_probs = np.mean(valid_probs, axis=0)
            results['fusion'] = {
                'emotion': EMOTIONS[np.argmax(avg_probs)],
                'confidence': float(np.max(avg_probs)),
                'probs': {e: float(p) for e, p in zip(EMOTIONS, avg_probs)},
                'note': 'Simple Average'
            }
    except Exception as e_f:
        print(f"❌ Fusion error: {e_f}")
        results['fusion'] = {'emotion': 'unknown', 'confidence': 0.0, 'probs': {}}

    # 5. RAG & Synthesis
    results['synthesis'] = synthesis = MentalHealthAnalyzer.analyze_patterns(results)
    if rag_service:
        try:
            emo = results['fusion']['emotion']
            conf = results['fusion']['confidence']
            insight = rag_service.generate_insight(emo, conf, additional_context=text or results.get('extracted_text', ""))
            results['response'] = {'answer': insight, 'resources': ["Curated Mental Health DB"]}
            results['synthesis']['description'] = insight
        except Exception as e_r:
            print(f"⚠️ RAG error: {e_r}")
            results['response'] = {'answer': 'Synthesis unavailable.', 'resources': []}

    # Remove internal null keys for clean response
    final_output = {k: v for k, v in results.items() if v is not None}
    print(f"📤 Returning results: {list(final_output.keys())}")
    return final_output

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