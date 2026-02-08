import os
import torch
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# --- FACE DETECTION SETUP ---
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- CONFIGURATION ---
RAVDESS_PATH = "d:/moni/MHprediction/datasets/video/RAVDESS"
FER_PATH = "d:/moni/MHprediction/datasets/video/FER2013"
# The 70% accuracy model from 3_train_video_model.py
TRAINED_VIDEO_WEIGHTS = "d:/moni/MHprediction/models/trained_models/video_model.pth"
XGB_MODEL_SAVE_PATH = "d:/moni/MHprediction/models/video_xgboost.model"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"🚀 Using Device: {DEVICE}")

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label2id = {label: i for i, label in enumerate(EMOTIONS)}

# --- MODEL ARCHITECTURE (From 3_train_video_model.py) ---
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
        convnext = models.convnext_tiny(weights=None) # We will load weights
        self.features = convnext.features
        self.spatial_attn = SpatialAttention()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        attn = self.spatial_attn(x)
        x = x * attn
        x = self.global_pool(x)
        # We want the output BEFORE the final classifier head for XGBoost
        return x

# --- DATA LOADING ---
def load_all_paths():
    all_data = []

    # 1. Load FER2013 (Images)
    if os.path.exists(FER_PATH):
        print(f"📂 Scanning FER2013 at {FER_PATH}...")
        for split in ['train', 'test']:
            split_path = os.path.join(FER_PATH, split)
            if os.path.exists(split_path):
                for emotion in os.listdir(split_path):
                    if emotion.lower() in EMOTIONS:
                        cls_idx = label2id[emotion.lower()]
                        folder = os.path.join(split_path, emotion)
                        for file in os.listdir(folder):
                            all_data.append({'path': os.path.join(folder, file), 'label': cls_idx, 'type': 'image'})

    # 2. Load RAVDESS (Video)
    if os.path.exists(RAVDESS_PATH):
        print(f"📂 Scanning RAVDESS Video at {RAVDESS_PATH}...")
        for actor in os.listdir(RAVDESS_PATH):
            actor_path = os.path.join(RAVDESS_PATH, actor)
            if os.path.isdir(actor_path):
                for file in os.listdir(actor_path):
                    if file.endswith('.mp4'):
                        parts = file.split('-')
                        if len(parts) > 3:
                            code = parts[2]
                            raw_emotion = {
                                '01': 'neutral', '02': 'neutral', '03': 'happy', '04': 'sad', 
                                '05': 'angry', '06': 'fear', '07': 'disgust', '08': 'surprise'
                            }.get(code)
                            if raw_emotion in EMOTIONS:
                                all_data.append({'path': os.path.join(actor_path, file), 'label': label2id[raw_emotion], 'type': 'video'})

    print(f"✅ Total Samples Found: {len(all_data)}")
    return all_data

class VideoFeatureDataset(Dataset):
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
            if item['type'] == 'video':
                cap = cv2.VideoCapture(image_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        frame = frame[y:y+h, x:x+w]
                    
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                else:
                    image = Image.new('RGB', (224, 224), (0, 0, 0))
            else:
                image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)
        except:
            img = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform: img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)

# --- EXTRACTION ---
def extract_embeddings(model, loader):
    embeddings = []
    labels = []
    
    model.eval()
    print("🧠 Extracting Visual Embeddings (Wav2Vec-style strategy)...")
    with torch.no_grad():
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(DEVICE)
            # Forward pass through the trained model up to global_pool
            features = model(inputs) # Returns global_pool output
            features = torch.flatten(features, 1) # [Batch, 768]
            
            embeddings.append(features.cpu().numpy())
            labels.extend(targets.numpy())
            
    return np.vstack(embeddings), np.array(labels)

# --- MAIN ---
def main():
    # 1. Prepare Data
    data_list = load_all_paths()
    if not data_list:
        print("❌ No data found! Check paths.")
        return

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = VideoFeatureDataset(data_list, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. Setup Feature Extractor (Local trained model)
    print(f"🔄 Loading Trained Video Model from {TRAINED_VIDEO_WEIGHTS}...")
    model = HighAccEmotionModel(num_classes=7).to(DEVICE)
    if os.path.exists(TRAINED_VIDEO_WEIGHTS):
        model.load_state_dict(torch.load(TRAINED_VIDEO_WEIGHTS, map_location=DEVICE))
        print("✅ Weights Loaded!")
    else:
        print("⚠️ Weights not found! This will use random init (Not recommended).")

    # 3. Extract Features
    X, y = extract_embeddings(model, loader)
    print(f"📊 Extracted Visual Features Shape: {X.shape}")

    # 4. Train XGBoost
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    print("🔥 Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=len(EMOTIONS),
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    xgb_model.fit(X_train, y_train)

    # 5. Evaluate
    preds = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🏆 Final Video Accuracy (Embeddings + XGBoost): {acc:.2%}")
    print(classification_report(y_test, preds, target_names=EMOTIONS))

    # 6. Save
    joblib.dump(xgb_model, XGB_MODEL_SAVE_PATH)
    print(f"💾 Video XGBoost Model Saved to {XGB_MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
