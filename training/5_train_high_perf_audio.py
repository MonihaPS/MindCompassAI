import os
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    Trainer, 
    TrainingArguments
)
from torch.utils.data import WeightedRandomSampler
import soundfile as sf
import warnings
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAVDESS_PATH = "d:/moni/MHprediction/datasets/audio/RAVDESS"
CREMA_PATH = "d:/moni/MHprediction/datasets/audio/CREMAD"
MODEL_SAVE_PATH = "d:/moni/MHprediction/models/audio_model_high_acc.pt"

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

# Target standard emotions for our system
TARGET_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
label2id = {label: i for i, label in enumerate(TARGET_EMOTIONS)}
id2label = {i: label for i, label in enumerate(TARGET_EMOTIONS)}

# --- DATA PREPARATION ---
def load_datasets():
    data = []
    
    # 1. Load RAVDESS (Original)
    print("📂 Loading RAVDESS...")
    for root, _, files in os.walk(RAVDESS_PATH):
        for file in files:
            if file.endswith(".wav"):
                # Filename: 03-01-06-01-01-01-01.wav
                parts = file.split("-")
                emotion_code = parts[2]
                intensity = parts[3] # 01=normal, 02=strong
                
                if emotion_code in EMOTION_MAP:
                    emotion = EMOTION_MAP[emotion_code]
                    if emotion == 'calm': emotion = 'neutral' # Merge calm -> neutral
                    
                    if emotion in TARGET_EMOTIONS:
                        data.append({
                            "path": os.path.join(root, file),
                            "emotion": emotion,
                            "source": "ravdess"
                        })

    # 2. Load CREMA-D (New High-Performance Data)
    print("📂 Loading CREMA-D...")
    if os.path.exists(CREMA_PATH):
        for file in os.listdir(CREMA_PATH):
            if file.endswith(".wav"):
                # Filename: 1001_DFA_ANG_XX.wav
                parts = file.split("_")
                emotion_code = parts[2]
                
                if emotion_code in EMOTION_MAP:
                    emotion = EMOTION_MAP[emotion_code]
                    if emotion in TARGET_EMOTIONS:
                        data.append({
                            "path": os.path.join(CREMA_PATH, file),
                            "emotion": emotion,
                            "source": "crema"
                        })
    else:
        print(f"⚠️ CREMA-D path not found: {CREMA_PATH}")

    df = pd.DataFrame(data)
    print(f"✅ Total Data Points: {len(df)}")
    print(df['emotion'].value_counts())
    return df

class AugmentedAudioDataset(Dataset):
    def __init__(self, df, processor, max_length=48000, augment=False):
        self.df = df
        self.processor = processor
        self.max_length = max_length
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def augment_audio(self, y, sr):
        # 1. Add Gaussian Noise
        if np.random.rand() < 0.5:
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise
            
        # 2. Time Stretch (Faster/Slower)
        if np.random.rand() < 0.5:
            rate = np.random.uniform(0.8, 1.2)
            y = librosa.effects.time_stretch(y, rate=rate)
            
        # 3. Pitch Shift
        if np.random.rand() < 0.5:
            steps = np.random.randint(-2, 2)
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            
        return y

    def __getitem__(self, idx):
        path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['emotion']
        
        # Load audio
        y, sr = librosa.load(path, sr=16000)
        
        # Augmentation (Only training)
        if self.augment:
            try:
                y = self.augment_audio(y, sr)
            except:
                pass # Use original if augmentation fails

        # Pad/Crop
        if len(y) > self.max_length:
            y = y[:self.max_length]
        else:
            y = np.pad(y, (0, self.max_length - len(y)), 'constant')

        # Feature Extraction
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

# --- MODEL DEFINITION ---
def train_model():
    # 1. Prepare Data
    df = load_datasets()
    
    # Stratified Split (80/20)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Processor (Feature Extractor)
    model_name = "superb/wav2vec2-base-superb-er"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    
    train_dataset = AugmentedAudioDataset(train_df, processor, augment=True) # Augment ON
    val_dataset = AugmentedAudioDataset(val_df, processor, augment=False)
    
    # Class Weights for Imbalanced Data
    class_counts = train_df['emotion'].value_counts().sort_index().values
    weights = 1. / class_counts
    samples_weights = weights[train_df['emotion'].map(lambda x: label2id[x]).values]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, # Smaller batch size for stability
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 2. Load Model
    print("🧠 Initializing Wav2Vec2 Model...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_EMOTIONS),
        ignore_mismatched_sizes=True
    ).to(DEVICE)
    
    # Advanced Fine-Tuning: Freeze Feature Extractor, Unfreeze Encoder
    model.freeze_feature_encoder() # Always freeze the conv layers
    
    # Optimizer & Scheduler (The "90% Secret")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=1e-4, 
        steps_per_epoch=len(train_loader), 
        epochs=15, # Increased epochs for deep learning
        pct_start=0.1
    )
    
    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    best_acc = 0.0
    print("🔥 Starting High-Performance Training...")
    
    for epoch in range(15):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/15 [Train]", leave=False)
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
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            loop.set_postfix(loss=total_loss/(total/8), acc=correct/total)
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/15 [Val]", leave=False)
        with torch.no_grad():
            for batch in loop_val:
                inputs = batch['input_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/15 | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"⭐ New Best Model! Saving... ({best_acc:.2%})")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"✅ Training Complete. Best Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    train_model()
