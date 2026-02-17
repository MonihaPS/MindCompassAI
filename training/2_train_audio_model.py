import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import random

os.environ['HF_HOME'] = 'D:/moni/.cache/huggingface'
os.environ['TORCH_HOME'] = 'D:/moni/.cache/torch'

# Add root to sys.path for config import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from config import EMOTIONS, DATA_PATHS, AUDIO_CONFIG, DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATHS

# ============================================
# FOCAL LOSS IMPLEMENTATION
# ============================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

# ============================================
# AUDIO AUGMENTATION
# ============================================

class AudioAugmentor:
    """Apply random transformations to raw audio for better generalization"""
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        
    def add_noise(self, data):
        noise_amp = 0.005 * np.random.uniform() * np.amax(data)
        data = data + noise_amp * np.random.normal(size=data.shape[0])
        return data

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high=5) * self.sr / 10)
        return np.roll(data, shift_range)

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate=rate)

    def pitch(self, data, n_steps=0.7):
        return librosa.effects.pitch_shift(data, sr=self.sr, n_steps=n_steps)

    def augment(self, data):
        """Randomly apply one or more augmentations"""
        if random.random() > 0.5: data = self.add_noise(data)
        if random.random() > 0.5: data = self.shift(data)
        # Time stretch and pitch shift are expensive, apply sparingly
        if random.random() > 0.8: 
            rate = random.uniform(0.8, 1.2)
            data = self.stretch(data, rate)
        return data

# ============================================
# LOAD DATASET
# ============================================

def load_audio_data():
    """Load audio emotion dataset from RAVDESS with standardized mapping"""
    print("\n" + "="*70)
    print("LOADING AUDIO EMOTION DATASET (RAVDESS)")
    print("="*70)
    
    audio_paths = []
    emotion_labels = []
    
    dataset_path = DATA_PATHS['audio']
    emotion_map = AUDIO_CONFIG['emotion_map']
    
    if not os.path.exists(dataset_path):
        print(f"⚠️  {dataset_path} not found!")
        return [], []
    
    print(f"🔍 Searching in {dataset_path}...")
    actors = list(Path(dataset_path).glob('Actor_*'))
    
    for actor_dir in actors:
        for audio_file in actor_dir.glob('*.wav'):
            parts = audio_file.stem.split('-')
            if len(parts) >= 7:
                emotion_code = parts[2]
                if emotion_code in emotion_map:
                    audio_paths.append(str(audio_file))
                    emotion_labels.append(emotion_map[emotion_code])
    
    print(f"✓ Loaded {len(audio_paths)} audio paths")
    return audio_paths, emotion_labels

class AudioEmotionDataset(Dataset):
    def __init__(self, audio_paths, labels, extractor, augment=False):
        self.audio_paths = audio_paths
        self.labels = labels
        self.extractor = extractor
        self.augment = augment
        self.augmentor = AudioAugmentor(sample_rate=AUDIO_CONFIG['sample_rate'])
        self.sample_rate = AUDIO_CONFIG['sample_rate']
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(EMOTIONS)
        self.encoded_labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        try:
            audio, _ = librosa.load(self.audio_paths[idx], sr=self.sample_rate)
            
            # Apply augmentation only for training set
            if self.augment:
                audio = self.augmentor.augment(audio)
                
            inputs = self.extractor(
                audio,
                sampling_rate=self.sample_rate,
                padding='max_length',
                max_length=AUDIO_CONFIG['max_length'],
                truncation=True,
                return_tensors="pt"
            )
            return {
                'input_values': inputs.input_values.squeeze(0),
                'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
            }
        except Exception as e:
            print(f"⚠️ Error loading {self.audio_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = batch['input_values'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(dataloader), 100 * correct / total

def validate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def main():
    # Setup logging
    log_file = os.path.join(root_dir, 'logs', 'audio_train.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    audio_paths, labels = load_audio_data()
    if not audio_paths: return

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        audio_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_CONFIG['model_name'])
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        AUDIO_CONFIG['model_name'],
        num_labels=len(EMOTIONS),
        ignore_mismatched_sizes=True # Important for switching models
    ).to(DEVICE)

    # Load existing weights if they exist to refine the 71% model
    if os.path.exists(MODEL_PATHS['audio']):
        print(f"📈 Loading existing Audio Specialist weights from {MODEL_PATHS['audio']} for refinement...")
        try:
            # Use weights_only=True for safety if using torch 2.0+
            model.load_state_dict(torch.load(MODEL_PATHS['audio'], map_location=DEVICE))
            print("✅ Successfully loaded 71% checkpoint. Building upon existing progress!")
        except Exception as e:
            print(f"⚠️ Could not load exact weights (likely architecture change), starting fresh fine-tuning: {e}")

    train_ds = AudioEmotionDataset(train_paths, train_labels, extractor, augment=True)
    val_ds = AudioEmotionDataset(val_paths, val_labels, extractor, augment=False)
    train_loader = DataLoader(train_ds, batch_size=AUDIO_CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=AUDIO_CONFIG['batch_size'], num_workers=0)

    # Differential Learning Rates: Backbone vs Classifier Head
    # backbone parameters usually start with 'wav2vec2' or reside in 'base_model'
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name or "projector" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = AdamW([
        {'params': backbone_params, 'lr': 1e-5}, # Lower LR for pre-trained weights
        {'params': head_params, 'lr': 1e-4}      # Higher LR for new classification head
    ])
    
    # OneCycleLR is better for hitting peak accuracy quickly
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-5, 1e-4], # Max LRs for each group
        steps_per_epoch=len(train_loader), 
        epochs=EPOCHS
    )
    criterion = FocalLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_acc = validate(model, val_loader, DEVICE)
        msg = f"Epoch {epoch+1}: Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        log_print(msg)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATHS['audio'])
            log_print(f"⭐ Saving Improved Audio Specialist ({val_acc:.2f}%)")

    print(f"\n✅ Audio Training Complete. Best Val Acc: {best_acc:.2f}%")

if __name__ == "__main__":
    main()