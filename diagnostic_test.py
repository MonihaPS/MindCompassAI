
import torch
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.getcwd())
from config import EMOTIONS
from app import TextModelWrapper, AudioModelWrapper

def test_models():
    print("🔍 DIAGNOSTIC TEST: MODEL LABEL MAPPING")
    print(f"Standard Emotions List: {EMOTIONS}")
    
    print("\n--- TEXT MODEL TEST ---")
    text_model = TextModelWrapper()
    test_text = "I am very sad and feeling terrible"
    # Call the score_func method
    probs = text_model.score_func([test_text])[0]
    top_idx = np.argmax(probs)
    print(f"Text Input: '{test_text}'")
    print(f"Predicted: {EMOTIONS[top_idx]} ({probs[top_idx]:.2%})")
    print(f"Full Probability Vector: {probs}")
    
    if EMOTIONS[top_idx] != 'sad':
        print("❌ WARNING: Text model predicted something else! Indexing might be swapped.")
    else:
        print("✅ Text model correctly identified sadness.")

    print("\n--- CONCLUSION ---")
    if EMOTIONS[3] == 'happy':
        print("Index 3 is HAPPY. If negative sentences give index 3, labels are misaligned.")

if __name__ == "__main__":
    test_models()
