"""
MULTIMODAL FUSION - TRAINING SCRIPT
====================================
Trains meta-classifier that fuses text, audio, and video predictions
Shows final fused accuracy and performance metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import os
import numpy as np

# Add root to sys.path for config import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from config import EMOTIONS, DATA_PATHS, DEVICE, MODEL_PATHS

# ============================================
# CONFIGURATION
# ============================================

# ============================================
# RESIDUAL ATTENTION FUSION MODEL
# ============================================

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
        attn_weights = F.softmax(attn_weights, dim=-1)

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
# CREATE SYNTHETIC VALIDATION DATA
# ============================================

# ============================================
# DATASET AND TRAINING LOOPS
# ============================================

class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Flattened features: [text(7), audio(7), video(7)]
        feat = self.X[idx]
        text = feat[0:7]
        audio = feat[7:14]
        video = feat[14:21]
        return text, audio, video, self.y[idx]

def train_fusion_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for t, a, v, labels in loader:
        t, a, v, labels = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(t, a, v)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), 100 * correct / total

def validate_fusion(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for t, a, v, labels in loader:
            t, a, v, labels = t.to(DEVICE), a.to(DEVICE), v.to(DEVICE), labels.to(DEVICE)
            logits, weights = model(t, a, v)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total, weights[0].cpu().numpy()

# ============================================
# CREATE SYNTHETIC FUSION DATA (FOR INITIAL SETUP)
# ============================================

def create_fusion_data(n_samples=2000):
    """
    Generate synthetic data matching current model accuracies:
    Text (~85%), Audio (~77%), Video (~70%)
    """
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(n_samples):
        # Pick true emotion
        true_emotion = np.random.randint(0, 7)
        
        # 1. Text Modality (~85% accuracy simulation)
        text_logits = np.random.normal(0, 0.5, 7)
        if np.random.random() < 0.85:
            text_logits[true_emotion] += 2.0
        else:
            wrong_emo = np.random.randint(0, 7)
            text_logits[wrong_emo] += 2.0
        
        # 2. Audio Modality (~77% accuracy simulation)
        audio_logits = np.random.normal(0, 0.7, 7)
        if np.random.random() < 0.77:
            audio_logits[true_emotion] += 1.8
        else:
            wrong_emo = np.random.randint(0, 7)
            audio_logits[wrong_emo] += 1.8
            
        # 3. Video Modality (~70% accuracy simulation)
        video_logits = np.random.normal(0, 0.8, 7)
        if np.random.random() < 0.70:
            video_logits[true_emotion] += 1.5
        else:
            wrong_emo = np.random.randint(0, 7)
            video_logits[wrong_emo] += 1.5
            
        X.append(np.hstack([
            F.softmax(torch.tensor(text_logits), dim=-1).numpy(),
            F.softmax(torch.tensor(audio_logits), dim=-1).numpy(),
            F.softmax(torch.tensor(video_logits), dim=-1).numpy()
        ]))
        y.append(true_emotion)
        
    return np.array(X), np.array(y)

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*70)
    print("ATTENTION-BASED MULTIMODAL FUSION TRAINING")
    print("="*70)
    
    # 1. Prepare Data
    X, y = create_fusion_data()
    split = int(0.8 * len(X))
    train_loader = DataLoader(FusionDataset(X[:split], y[:split]), batch_size=32, shuffle=True)
    val_loader = DataLoader(FusionDataset(X[split:], y[split:]), batch_size=32)

    # 2. Initialize Model
    model = AttentionFusionModel(num_emotions=7).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"✓ Model: {model.__class__.__name__}")
    print(f"✓ Device: {DEVICE}")

    # 3. Training Loop
    epochs = 20
    best_acc = 0
    
    print("\nTraining Phase...")
    for epoch in range(epochs):
        loss, train_acc = train_fusion_epoch(model, train_loader, optimizer, criterion)
        val_acc, sample_weights = validate_fusion(model, val_loader)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join("models", "trained_models", "fusion_model.pth"))
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Sample Modality Weights [Text, Audio, Video]: {sample_weights.flatten()}")

    print("\n" + "="*70)
    print(f"✅ FUSION TRAINING COMPLETE. Best Val Acc: {best_acc:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()