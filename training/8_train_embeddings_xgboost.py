import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAVDESS_PATH = "d:/moni/MHprediction/datasets/audio/RAVDESS"
CREMA_PATH = "d:/moni/MHprediction/datasets/audio/CREMAD"
TRAINED_MODEL_PATH = "d:/moni/MHprediction/models/audio_model_90_acc_finetuned.pt" # Using the latest fine-tuned model
XGB_MODEL_PATH = "d:/moni/MHprediction/models/audio_xgboost.model"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# Unified Emotion Map
EMOTION_MAP = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise',
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
label2id = {label: i for i, label in enumerate(TARGET_EMOTIONS)}

# --- DATA LOADING ---
def load_datasets():
    data = []
    print("📂 Scanning datasets...")
    # RAVDESS
    for root, _, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion = EMOTION_MAP.get(parts[2])
                if emotion == 'calm': emotion = 'neutral'
                if emotion in TARGET_EMOTIONS:
                    data.append({"path": os.path.join(root, file), "emotion": emotion})
    # CREMA-D
    if os.path.exists(CREMA_PATH):
        for file in os.listdir(CREMA_PATH):
            if file.endswith(".wav"):
                parts = file.split("_")
                emotion = EMOTION_MAP.get(parts[2])
                if emotion in TARGET_EMOTIONS:
                    data.append({"path": os.path.join(CREMA_PATH, file), "emotion": emotion})
    
    return pd.DataFrame(data)

class AudioFeatureDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['emotion']
        
        y, sr = librosa.load(path, sr=16000)
        # No aug, pure evaluation
        if len(y) > 48000: y = y[:48000]
        else: y = np.pad(y, (0, 48000 - len(y)), 'constant')
            
        inputs = self.processor(y, sampling_rate=16000, return_tensors="pt", padding="max_length", max_length=48000, truncation=True)
        return inputs.input_values.squeeze(), label2id[label]

# --- EXTRACT EMBEDDINGS ---
def extract_embeddings(model, loader):
    embeddings = []
    labels = []
    
    print("🧠 Extracting Embeddings (This takes ~5 mins)...")
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(DEVICE)
            
            # Forward pass through Wav2Vec2 ONLY (Skip classifier head)
            # We want the "hidden_states" from the transformer
            outputs = model.wav2vec2(inputs)
            
            # Global Average Pooling over time dimension
            # Shape: [batch, time, 768] -> [batch, 768]
            hidden_states = outputs.last_hidden_state
            pooled_output = torch.mean(hidden_states, dim=1)
            
            embeddings.append(pooled_output.cpu().numpy())
            labels.extend(targets.numpy())
            
    return np.vstack(embeddings), np.array(labels)

# --- MAIN ---
def main():
    # 1. Setup Feature Extractor (Wav2Vec2)
    model_name = "superb/wav2vec2-base-superb-er"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    print(f"🔄 Loading Trained Weights from: {TRAINED_MODEL_PATH}")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=7, ignore_mismatched_sizes=True).to(DEVICE)
    try:
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH))
        print("✅ Weights Loaded!")
    except:
        print("❌ Could not load weights. Using base model (Lower accuracy).")

    # 2. Prepare Data
    df = load_datasets()
    dataset = AudioFeatureDataset(df, processor)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0) # Larger batch for extraction
    
    # 3. Extract Features
    X, y = extract_embeddings(model, loader)
    print(f"📊 Extracted Features Shape: {X.shape}") 
    
    # 4. Train XGBoost
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("🔥 Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=7,
        tree_method='hist', # Fast histogram method
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    xgb_model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🏆 Final XGBoost Accuracy: {acc:.2%}")
    print(classification_report(y_test, preds, target_names=TARGET_EMOTIONS))
    
    # 6. Save
    joblib.dump(xgb_model, XGB_MODEL_PATH)
    print(f"💾 XGBoost Model Saved to {XGB_MODEL_PATH}")

if __name__ == "__main__":
    main()
