import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt

# Force cache to D: drive due to C: being full
os.environ['HF_HOME'] = 'D:/moni/.cache/huggingface'
os.environ['TORCH_HOME'] = 'D:/moni/.cache/torch'

# Add root to sys.path for config import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from config import EMOTIONS, DATA_PATHS, TEXT_CONFIG, DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATHS

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
# LOAD DATASET
# ============================================

def load_text_data():
    """Load text emotion dataset with standardized mapping"""
    print("\n" + "="*70)
    print("LOADING TEXT EMOTION DATASET")
    print("="*70)
    
    texts, labels = [], []
    dataset_path = DATA_PATHS['text']
    
    # Mapping for common text emotion datasets
    text_map = {
        'sadness': 'sad',
        'joy': 'happy',
        'love': 'happy',
        'anger': 'angry',
        'fear': 'fear',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }

    train_file = os.path.join(dataset_path, "train.txt")
    if not os.path.exists(train_file):
        print(f"⚠️  {train_file} not found! Using existing demo logic...")
        demo_texts = ['i am happy', 'i am sad', 'this is scary', 'i am furious', 'what?!', 'i am okay'] * 100
        demo_labels = ['happy', 'sad', 'fear', 'angry', 'surprise', 'neutral'] * 100
        return demo_texts, demo_labels
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) == 2:
                text, emotion = parts
                if emotion.strip() in text_map:
                    texts.append(text.strip())
                    labels.append(text_map[emotion.strip()])
    
    print(f"✓ Loaded {len(texts)} text samples")
    return texts, labels

class TextEmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(EMOTIONS)
        self.encoded_labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=TEXT_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        }

# ============================================
# TRAINING LOGIC
# ============================================

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    num_labels = model.config.num_labels
    
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Safety check for label indices to prevent CUDA assert
        if torch.any(labels >= num_labels) or torch.any(labels < 0):
            print(f"❌ ERROR: Label index out of range! Labels: {labels}, Max allowed: {num_labels-1}")
            continue
            
        outputs = model(ids, attention_mask=mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return 100 * correct / total

# ============================================
# MAIN
# ============================================

def main():
    texts, labels = load_text_data()
    if not texts: return

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(TEXT_CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        TEXT_CONFIG['model_name'], 
        num_labels=len(EMOTIONS)
    ).to(DEVICE)

    train_ds = TextEmotionDataset(train_texts, train_labels, tokenizer)
    val_ds = TextEmotionDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, len(train_loader) * EPOCHS)
    criterion = FocalLoss()

    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_acc = validate(model, val_loader, DEVICE)
        
        print(f"Loss: {loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATHS['text'])
            print(f"⭐ Saving Text Specialist ({val_acc:.2f}%)")

    print("\n✅ Text Training Complete.")

if __name__ == "__main__":
    main()