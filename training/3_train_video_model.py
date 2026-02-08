import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Force cache to D: drive due to C: being full
os.environ['HF_HOME'] = 'D:/moni/.cache/huggingface'
os.environ['TORCH_HOME'] = 'D:/moni/.cache/torch'

# Add root to sys.path for config import
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
from config import EMOTIONS, DATA_PATHS, VIDEO_CONFIG, DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS, MODEL_PATHS

# ============================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, label_smoothing=0.1, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                true_dist = torch.zeros_like(inputs)
                true_dist.fill_(self.label_smoothing / (num_classes - 1))
                true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - self.label_smoothing)
        else:
            true_dist = targets

        log_pt = F.log_softmax(inputs, dim=-1)
        if self.label_smoothing > 0:
            ce_loss = -(true_dist * log_pt).sum(dim=-1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss

# ============================================
# MODERN CONVNEXT MODEL ARCHITECTURE
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

class HighAccEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(HighAccEmotionModel, self).__init__()
        # Upgrade to ConvNeXt-Tiny for SOTA accuracy
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = convnext.features
        
        # Spatial Attention remains a powerful addition
        self.spatial_attn = SpatialAttention()
        
        # Output channels for ConvNeXt-Tiny is 768
        in_features = 768 
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.GELU(), # GELU is native to ConvNeXt
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Apply Spatial Attention to features
        attn = self.spatial_attn(x)
        x = x * attn
        x = self.global_pool(x)
        x = self.head(x)
        return x

def load_video_data():
    image_paths, emotion_labels = [], []
    dataset_path = DATA_PATHS['video']
    if not os.path.exists(dataset_path): return [], []
    for emotion_dir in Path(dataset_path).glob('*'):
        if emotion_dir.is_dir() and emotion_dir.name in EMOTIONS:
            emotion = emotion_dir.name
            for img_file in emotion_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                emotion_labels.append(emotion)
    return image_paths, emotion_labels

class EmotionImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(EMOTIONS)
        self.encoded_labels = self.label_encoder.transform(labels)
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            label = self.encoded_labels[idx]
            if self.transform: image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))

def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def main():
    # Setup logging
    log_file = os.path.join(root_dir, 'logs', 'video_train.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_print(msg):
        print(msg)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

    image_paths, labels = load_video_data()
    if not image_paths: return
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.15, random_state=42, stratify=labels
    )
    transform_train = transforms.Compose([
        transforms.Resize((VIDEO_CONFIG['image_size'], VIDEO_CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((VIDEO_CONFIG['image_size'], VIDEO_CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_ds = EmotionImageDataset(train_paths, train_labels, transform_train)
    val_ds = EmotionImageDataset(val_paths, val_labels, transform_val)
    train_loader = DataLoader(train_ds, batch_size=VIDEO_CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=VIDEO_CONFIG['batch_size'], num_workers=0)
    model = HighAccEmotionModel(num_classes=len(EMOTIONS)).to(DEVICE)
    
    # Differential Learning Rates
    param_groups = [
        {'params': model.features.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': model.spatial_attn.parameters(), 'lr': LEARNING_RATE},
        {'params': model.head.parameters(), 'lr': LEARNING_RATE}
    ]
    
    optimizer = AdamW(param_groups, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=[LEARNING_RATE*0.5, LEARNING_RATE*5, LEARNING_RATE*5], 
                          steps_per_epoch=len(train_loader), epochs=EPOCHS)
    
    # Focal Loss with Label Smoothing effect
    criterion = FocalLoss() # We keep Focal Loss as primary but can add smoothing if needed
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, DEVICE)
        val_acc = validate(model, val_loader, DEVICE)
        msg = f"Epoch {epoch+1}: Loss: {loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        log_print(msg)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATHS['video'])
            log_print(f"⭐ Saving Improved Video Specialist ({val_acc:.2f}%)")
    print("\n✅ Video Training Complete.")

if __name__ == "__main__":
    main()