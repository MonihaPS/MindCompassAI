import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
RAVDESS_PATH = "d:/moni/MHprediction/datasets/video/RAVDESS"
FER_PATH = "d:/moni/MHprediction/datasets/video/FER2013" # Or "d:/moni/MHprediction/data/fer2013"
RAF_DB_PATH = "d:/moni/MHprediction/datasets/video/RAF_DB" # The new one

MODEL_SAVE_PATH = "d:/moni/MHprediction/models/video_model_high_acc.pt"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Using Device: {DEVICE}")

EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
label2id = {label: i for i, label in enumerate(EMOTIONS)}

# --- ADVANCED MIXUP / CUTMIX ---
def mixup_data(x, y, alpha=0.4):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- DATASET ---
class HighPerfVideoDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image_path = item['path']
        label = item['label']

        try:
            # Handle Video Frames (Extract middle frame on the fly if video)
            if image_path.endswith('.mp4') or image_path.endswith('.avi'):
                cap = cv2.VideoCapture(image_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                else:
                    # Fallback blank image
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                # Standard Image
                image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.long)
            
        except Exception:
            # Blank fallback on error
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

def load_data():
    all_data = []

    # 1. Load FER2013 (Images)
    if os.path.exists(FER_PATH):
        print(f"📂 Scanning FER2013 at {FER_PATH}...")
        for emotion in os.listdir(FER_PATH):
            if emotion in EMOTIONS:
                cls_idx = label2id[emotion]
                folder = os.path.join(FER_PATH, emotion)
                for file in os.listdir(folder):
                    all_data.append({'path': os.path.join(folder, file), 'label': cls_idx})

    # 2. Load RAVDESS (Video)
    if os.path.exists(RAVDESS_PATH):
        print(f"📂 Scanning RAVDESS Video at {RAVDESS_PATH}...")
        for actor in os.listdir(RAVDESS_PATH):
            actor_path = os.path.join(RAVDESS_PATH, actor)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.mp4'):
                        # 02-01-06-01-01-01-01.mp4
                        parts = file.split('-')
                        if len(parts) > 3:
                            code = parts[2]
                            # Ravdess Code Map
                            # 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
                            raw_emotion = {
                                '01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', 
                                '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
                            }.get(code)
                            
                            if raw_emotion in EMOTIONS:
                                all_data.append({'path': os.path.join(actor_path, file), 'label': label2id[raw_emotion]})

    # 3. Load RAF-DB (Images - High Res)
    if os.path.exists(RAF_DB_PATH):
        print(f"📂 Scanning RAF-DB at {RAF_DB_PATH}...")
         # Assuming structured folders (train/happy/x.jpg)
        for split in ['train', 'test']:
            split_path = os.path.join(RAF_DB_PATH, split)
            if os.path.exists(split_path):
                for emotion in os.listdir(split_path):
                    mapped_emotion = emotion.lower()
                    if mapped_emotion in EMOTIONS:
                        folder = os.path.join(split_path, emotion)
                        for file in os.listdir(folder):
                             all_data.append({'path': os.path.join(folder, file), 'label': label2id[mapped_emotion]})

    print(f"✅ Total Samples Found: {len(all_data)}")
    return all_data

# --- MODEL ---
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

class HighPerfVideoModel(nn.Module):
    def __init__(self):
        super().__init__()
        # ConvNeXt Tiny (State of The Art Backbone)
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Custom Attention Head
        self.features = self.backbone.features
        self.spatial_attn = SpatialAttention()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.5), # Standard Dropout
            nn.Linear(512, len(EMOTIONS))
        )

    def forward(self, x):
        x = self.features(x)
        attn = self.spatial_attn(x)
        x = x * attn # Apply attention
        x = self.avgpool(x)
        x = self.head(x)
        return x

# --- TRAINING LOOP ---
def main():
    # 1. Data Prep
    data = load_data()
    if not data:
        print("❌ No data found! Check paths.")
        return

    # Transforms (Strong Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Light robustness
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2) # Cutout regularization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Split
    np.random.shuffle(data)
    split = int(0.85 * len(data)) # More training data for 90% goal
    train_data = data[:split]
    val_data = data[split:]
    
    train_ds = HighPerfVideoDataset(train_data, transform=train_transform)
    val_ds = HighPerfVideoDataset(val_data, transform=val_transform)

    # Sampler
    labels = [x['label'] for x in train_data]
    class_counts = np.bincount(labels)
    weights = 1. / class_counts
    sample_weights = weights[labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. Model Setup
    print("🧠 Initializing High-Performance Video Model...")
    model = HighPerfVideoModel().to(DEVICE)
    
    # Optimizer (AdamW + OneCycle)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = OneCycleLR(optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=10)
    
    # Label Smoothing Loss (Better generalization)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 3. Train
    best_acc = 0.0
    print("🔥 Starting MixUp Training for 10 Epochs...")
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 [Train]", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Apply MixUp (50% chance)
            use_mixup = False
            if np.random.rand() < 0.5:
                use_mixup = True
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            # Accuracy is tricky with MixUp, approximate it
            total += targets.size(0)
            if use_mixup:
                correct += (lam * predicted.eq(targets_a).sum().float() + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
            else:
                correct += predicted.eq(targets).sum().item()
                
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1} Results | Train Acc: {correct/total:.2%} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"⭐ Saving New Best Model ({best_acc:.2%})...")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"✅ Training Complete. Best Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()
