import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor
)
from torch.utils.data import WeightedRandomSampler
import warnings
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAVDESS_PATH = "d:/moni/MHprediction/datasets/audio/RAVDESS"
CREMA_PATH = "d:/moni/MHprediction/datasets/audio/CREMAD"
PRETRAINED_MODEL_PATH = "d:/moni/MHprediction/models/audio_model_high_acc.pt"
FINAL_MODEL_PATH = "d:/moni/MHprediction/models/audio_model_90_acc_finetuned.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using Device: {DEVICE}")

# Unified Emotion Map (Ensures alignment with CREMA-D & RAVDESS)
EMOTION_MAP = {
    # RAVDESS codes
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
    '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise',
    # CREMA-D codes (ANG, DIS, FEA, HAP, NEU, SAD)
    'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 
    'HAP': 'happy', 'NEU': 'neutral', 'SAD': 'sad'
}

TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
label2id = {label: i for i, label in enumerate(TARGET_EMOTIONS)}

# --- DATA PREPARATION ---
def load_datasets():
    data = []
    
    # 1. Load RAVDESS (Original)
    print("📂 Loading RAVDESS...")
    for root, _, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_code = parts[2]
                if emotion_code in EMOTION_MAP:
                    emotion = EMOTION_MAP[emotion_code]
                    if emotion == 'calm': emotion = 'neutral'
                    if emotion in TARGET_EMOTIONS:
                        data.append({"path": os.path.join(root, file), "emotion": emotion})

    # 2. Load CREMA-D (New High-Performance Data)
    print("📂 Loading CREMA-D...")
    if os.path.exists(CREMA_PATH):
        for file in os.listdir(CREMA_PATH):
            if file.endswith(".wav"):
                parts = file.split("_")
                emotion_code = parts[2]
                if emotion_code in EMOTION_MAP:
                    emotion = EMOTION_MAP[emotion_code]
                    if emotion in TARGET_EMOTIONS:
                        data.append({"path": os.path.join(CREMA_PATH, file), "emotion": emotion})

    df = pd.DataFrame(data)
    print(f"✅ Total Data Points: {len(df)}")
    return df

class AudioDataset(Dataset):
    def __init__(self, df, processor, max_length=48000, augment=False):
        self.df = df
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def augment_audio(self, y, sr):
        # MILDER Augmentation for Fine-Tuning (Don't destroy the signal)
        if np.random.rand() < 0.3:
            noise = np.random.randn(len(y))
            y = y + 0.002 * noise # very light noise
        if np.random.rand() < 0.3:
            steps = np.random.randint(-1, 1) # very slight pitch
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
        return y

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['emotion']
        
        y, sr = librosa.load(path, sr=16000)
        
        if self.augment:
            try:
                y = self.augment_audio(y, sr)
            except:
                pass

        if len(y) > self.max_length:
            y = y[:self.max_length]
        else:
            y = np.pad(y, (0, self.max_length - len(y)), 'constant')

        inputs = self.processor(
            y, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )

        return {
            "input_values": inputs.input_values.squeeze(),
            "labels": torch.tensor(label2id[label], dtype=torch.long)
        }

# --- FINE-TUNING LOOP ---
def main():
    # 1. Load Data
    df = load_datasets()
    train_df = df.sample(frac=0.85, random_state=42) # More training data for fine-tuning
    val_df = df.drop(train_df.index)
    
    model_name = "superb/wav2vec2-base-superb-er"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    train_dataset = AudioDataset(train_df, processor, augment=True)
    val_dataset = AudioDataset(val_df, processor, augment=False)
    
    class_counts = train_df['emotion'].value_counts().sort_index().values
    weights = 1. / class_counts
    samples_weights = weights[train_df['emotion'].map(lambda x: label2id[x]).values]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(train_dataset, batch_size=6, sampler=sampler, num_workers=0) # Smaller batch for stability
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False)

    # 2. Load PRE-TRAINED Model (The 72% one)
    print(f"🧠 Loading Best Checkpoint: {PRETRAINED_MODEL_PATH}...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_EMOTIONS),
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
        print("✅ Weights Loaded Successfully!")
    except Exception as e:
        print(f"⚠️ Could not load weights: {e}. Starting fresh (not recommended).")

    # 3. Fine-Tuning Strategy: UNFREEZE MORE LAYERS (Aggressive Mode)
    model.freeze_feature_encoder() # Keep CNN frozen
    
    # Unfreeze the last 4 encoder layers (More capacity)
    for param in model.wav2vec2.encoder.layers[-4:].parameters():
        param.requires_grad = True
    
    # Optimizer (Higher Learning Rate for Speed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01) 
    scheduler = OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=len(train_loader), epochs=5)
    criterion = nn.CrossEntropyLoss()

    # 4. Train
    best_acc = 0.0
    print("🔥 Starting TURBO Fine-Tuning (Target: 90% in 5 Epochs)...")
    
    for epoch in range(5):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [Tune]", leave=False)
        for batch in loop:
            inputs = batch['input_values'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/10 [Val]", leave=False)
        with torch.no_grad():
            for batch in loop_val:
                inputs = batch['input_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} | Train Acc: {correct/total:.2%} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"⭐ Saving Fine-Tuned Model ({best_acc:.2%})...")
            torch.save(model.state_dict(), FINAL_MODEL_PATH)

    print(f"✅ Fine-Tuning Complete. Final Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()
