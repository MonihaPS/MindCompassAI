import torch
import os

# Force cache to D: drive due to C: being full
os.environ['HF_HOME'] = 'D:/moni/.cache/huggingface'
os.environ['TORCH_HOME'] = 'D:/moni/.cache/torch'

# ============================================
# UNIFIED EMOTION SCHEMA
# ============================================

# The standard 7 emotions used across FER2013 and most fusion research
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_LABELS = len(EMOTIONS)

# ============================================
# DATASET PATHS
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATHS = {
    'audio': os.path.join(BASE_DIR, 'datasets/audio/RAVDESS'),
    'video': os.path.join(BASE_DIR, 'datasets/video/train'),
    'text': os.path.join(BASE_DIR, 'datasets/text'), # Future use
}

# ============================================
# MODEL PATHS
# ============================================

MODEL_DIR = os.path.join(BASE_DIR, 'models/trained_models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATHS = {
    'text': os.path.join(MODEL_DIR, 'text_model.pth'),
    'audio': os.path.join(MODEL_DIR, 'audio_model.pth'),
    'video': os.path.join(MODEL_DIR, 'video_model.pth'),
    'fusion': os.path.join(BASE_DIR, 'models/fusion_meta_model.pkl'),
    'scaler': os.path.join(BASE_DIR, 'models/fusion_scaler.pkl'),
}

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16 # Safer default
LEARNING_RATE = 5e-5 # Boosted Fine-tuning LR
EPOCHS = 10 # Reduced as requested for faster solution

# ============================================
# MODALITY SPECIFIC CONFIG
# ============================================

AUDIO_CONFIG = {
    'model_name': 'superb/wav2vec2-base-superb-er', # Public & Specialized
    'batch_size': 4, # Increased for faster standalone training
    'sample_rate': 16000,
    'max_length': 160000,
    'emotion_map': {
        '01': 'neutral',
        '02': 'neutral', # calm -> neutral
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprise'
    }
}

VIDEO_CONFIG = {
    'image_size': 224,
    'batch_size': 4, # Reduced for parallel safety
    'dropout': 0.3
}

TEXT_CONFIG = {
    'model_name': 'roberta-base', # Public model to avoid auth issues
    'max_length': 128
}
