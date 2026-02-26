import os
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path
from collections import defaultdict

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

def extract_features_from_chunk(chunk, yamnet):
    try:
        _, emb, _ = yamnet(chunk)
        if emb.shape[0] == 0: return None
        mean_emb = tf.reduce_mean(emb, axis=0)
        max_emb = tf.reduce_max(emb, axis=0)
        final = tf.concat([mean_emb, max_emb], axis=0)
        return final
    except Exception as e:
        print(f"Error: {e}")
        return None

def merge_intervals(intervals):
    if not intervals: return []
    intervals.sort()
    merged = [list(intervals[0])]
    for current in intervals[1:]:
        previous = merged[-1]
        # If overlap or gap is very small (smooth out 1s gaps)
        if current[0] <= previous[1] + 1.0: 
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(list(current))
    return merged

def predict(audio_path, window_sec=5.0, hop_sec=1.0):
    print("="*50)
    print("🔎 HIERARCHICAL FORENSIC TIMELINE ANALYSIS")
    print("="*50)
    
    # 1. Load Custom Model
    model, labels = load_latest_model()
    if not model or not labels: return
    
    # 2. Load YAMNet
    print("📦 Loading YAMNet base...")
    yamnet = hub.load(YAMNET_MODEL_URL)
    
    # 3. Load Audio
    print(f"🎵 Analyzing: {audio_path}")
    try:
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return
        
    duration = len(wav) / sr
    print(f"⏱️ Audio Duration: {duration:.1f} seconds")
    
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    features_list = []
    timestamps = []
    
    # Chunk audio
    print(f"⏳ Extracting features in {window_sec}s windows (sliding every {hop_sec}s)...")
    for start in range(0, len(wav), hop_samples):
        end = start + window_samples
        chunk = wav[start:end]
        
        if len(chunk) < sr * 1.0: 
            continue
            
        feat = extract_features_from_chunk(chunk, yamnet)
        if feat is not None:
            features_list.append(feat)
            
            start_t = start / sr
            end_t = min((start + len(chunk)) / sr, duration)
            timestamps.append((start_t, end_t))
            
    if not features_list:
        print("❌ No valid audio segments found.")
        return
        
    # 4. Predict
    X = tf.stack(features_list)
    preds = model.predict(X, verbose=0)
    
    main_preds = preds[0] # (num_windows, num_main_classes)
    sub_preds = preds[1]  # (num_windows, num_sub_classes)
    
    THRESHOLD = 0.3
    
    main_classes = labels.get("main_classes", [])
    sub_classes = labels.get("sub_classes", [])
    
    # Store raw interval detections
    main_detections = defaultdict(list)
    sub_detections = defaultdict(list)
    
    for i, (start_t, end_t) in enumerate(timestamps):
        # Process Main Classes
        for cls_idx, conf in enumerate(main_preds[i]):
            if conf >= THRESHOLD:
                main_detections[main_classes[cls_idx]].append((start_t, end_t))
                
        # Process Sub Classes
        for cls_idx, conf in enumerate(sub_preds[i]):
            if conf >= THRESHOLD:
                sub_detections[sub_classes[cls_idx]].append((start_t, end_t))
                
    # 5. Display Results
    print("\n📊 TIMELINE RESULTS (>30% Confidence)")
    print("-" * 50)
    
    # --- Main Class ---
    print("📁 MAIN CATEGORIES DETECTED:")
    if not main_detections:
        print("   (None above threshold)")
    else:
        for cls_name, intervals in main_detections.items():
            merged = merge_intervals(intervals)
            timeline = ", ".join([f"{s:.1f}s - {e:.1f}s" for s, e in merged])
            print(f"   • {cls_name.upper():<15} ⏱️ {timeline}")

    # --- Sub Class ---
    print("\n🏷️ SPECIFIC EVENTS DETECTED:")
    if not sub_detections:
        print("   (None above threshold)")
    else:
        for cls_name, intervals in sub_detections.items():
            merged = merge_intervals(intervals)
            timeline = ", ".join([f"{s:.1f}s - {e:.1f}s" for s, e in merged])
            print(f"   🎯 {cls_name:<25} ⏱️ {timeline}")
            
    print("-" * 50)

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
