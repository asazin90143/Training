import os
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path

# Disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

def load_latest_model():
    if not MODELS_DIR.exists():
        print("❌ No models/ directory found.")
        return None, None
        
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        print("❌ No .keras models found.")
        return None, None
        
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {latest_model.name}")
    
    # Load Model
    model = tf.keras.models.load_model(str(latest_model))
    
    # Load Labels
    labels_path = MODELS_DIR / f"{latest_model.stem}_labels.json"
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
    else:
        print("⚠️ Labels file not found.")
        labels = None
        
    return model, labels

def extract_features(audio_path, yamnet):
    try:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
        _, emb, _ = yamnet(wav)
        mean_emb = tf.reduce_mean(emb, axis=0)
        max_emb = tf.reduce_max(emb, axis=0)
        final = tf.concat([mean_emb, max_emb], axis=0)
        return tf.expand_dims(final, axis=0)
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict(audio_path):
    print("="*50)
    print("🔎 HIERARCHICAL FORENSIC ANALYSIS")
    print("="*50)
    
    # 1. Load Custom Model
    model, labels = load_latest_model()
    if not model: return
    
    # 2. Load YAMNet
    print("📦 Loading YAMNet base...")
    yamnet = hub.load(YAMNET_MODEL_URL)
    
    # 3. Features
    print(f"🎵 Analyzing: {audio_path}")
    feats = extract_features(audio_path, yamnet)
    if feats is None: return
    
    # 4. Predict
    # Returns [main_output, sub_output]
    preds = model.predict(feats, verbose=0)
    
    main_pred = preds[0][0] # Batch 0
    sub_pred = preds[1][0]
    
    # Threshold for detection
    THRESHOLD = 0.3
    
    # 5. Display Results
    print("\n📊 RESULTS (Multi-Label Detection)")
    print("-" * 30)
    
    # --- Main Class ---
    print("📁 MAIN CATEGORIES:")
    if labels and "main_classes" in labels:
        main_classes = labels["main_classes"]
        found_main = False
        # Sort by confidence
        idxs = np.argsort(main_pred)[::-1]
        for idx in idxs:
            conf = main_pred[idx]
            if conf >= THRESHOLD:
                print(f"   • {main_classes[idx].upper()} ({conf*100:.1f}%)")
                found_main = True
        
        if not found_main:
             top_idx = np.argmax(main_pred)
             print(f"   (Best guess: {main_classes[top_idx].upper()} {main_pred[top_idx]*100:.1f}%)")
    else:
         print(f"   Raw: {main_pred}")

    # --- Sub Class ---
    print("\n🏷️ SPECIFIC EVENTS:")
    if labels and "sub_classes" in labels:
        sub_classes = labels["sub_classes"]
        found_sub = False
        
        idxs = np.argsort(sub_pred)[::-1]
        for idx in idxs:
            conf = sub_pred[idx]
            if conf >= THRESHOLD:
                 print(f"   🎯 {sub_classes[idx]:<30} {conf*100:.1f}%")
                 found_sub = True
        
        if not found_sub:
             top_idx = np.argmax(sub_pred)
             print(f"   (Best guess: {sub_classes[top_idx]} {sub_pred[top_idx]*100:.1f}%)")
            
    else:
        print(f"   Raw: {sub_pred}")
    
    print("-" * 30)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?")
    args = parser.parse_args()
    
    if args.file:
        predict(args.file)
    else:
        print("Usage: python test_model.py <file.wav>")

if __name__ == "__main__":
    main()
