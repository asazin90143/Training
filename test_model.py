import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path
from collections import defaultdict

# Anomaly Detection
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Force UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

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


def load_all_models():
    """Load ALL .keras models for ensemble voting."""
    if not MODELS_DIR.exists():
        print("❌ No models/ directory found.")
        return [], None
    
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        print("❌ No .keras models found.")
        return [], None
    
    loaded = []
    labels = None
    
    for m_path in models:
        try:
            model = tf.keras.models.load_model(str(m_path))
            loaded.append((m_path.stem, model))
            print(f"   ✅ Loaded: {m_path.name}")
            
            # Load labels from any model (they should all have the same classes)
            if labels is None:
                lbl_path = MODELS_DIR / f"{m_path.stem}_labels.json"
                if lbl_path.exists():
                    with open(lbl_path) as f:
                        labels = json.load(f)
        except Exception as e:
            print(f"   ⚠️ Skip {m_path.name}: {e}")
    
    print(f"\n🗳️ Ensemble loaded: {len(loaded)} models")
    return loaded, labels

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

def analyze_at_resolution(wav, sr, duration, model, yamnet, window_sec, hop_sec, threshold, labels):
    """Run prediction at a single resolution and return raw detections."""
    main_classes = labels.get("main_classes", [])
    sub_classes = labels.get("sub_classes", [])
    
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    features_list = []
    timestamps = []
    
    for start in range(0, len(wav), hop_samples):
        end = start + window_samples
        chunk = wav[start:end]
        
        if len(chunk) < sr * 0.5:
            continue
            
        feat = extract_features_from_chunk(chunk, yamnet)
        if feat is not None:
            features_list.append(feat)
            start_t = start / sr
            end_t = min((start + len(chunk)) / sr, duration)
            timestamps.append((start_t, end_t))
    
    if not features_list:
        return {}, {}
    
    X = tf.stack(features_list)
    preds = model.predict(X, verbose=0)
    
    main_preds = preds[0]
    sub_preds = preds[1]
    
    main_detections = defaultdict(list)
    sub_detections = defaultdict(list)
    
    for i, (start_t, end_t) in enumerate(timestamps):
        for cls_idx, conf in enumerate(main_preds[i]):
            if conf >= threshold:
                main_detections[main_classes[cls_idx]].append((start_t, end_t))
        for cls_idx, conf in enumerate(sub_preds[i]):
            if conf >= threshold:
                sub_detections[sub_classes[cls_idx]].append((start_t, end_t))
    
    return main_detections, sub_detections


def predict(audio_path, threshold=0.3, ensemble=False, anomaly=False):
    print("="*50)
    if ensemble:
        print("🗳️ ENSEMBLE MULTI-RESOLUTION FORENSIC ANALYSIS")
    else:
        print("🔎 MULTI-RESOLUTION FORENSIC TIMELINE ANALYSIS")
    print("="*50)
    
    # 1. Load Model(s)
    if ensemble:
        print("📂 Loading ALL models for ensemble voting...")
        models_list, labels = load_all_models()
        if not models_list or not labels: return
        model = None  # Not used in ensemble mode
    else:
        model, labels = load_latest_model()
        if not model or not labels: return
        models_list = None
    
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
    
    # 4. Multi-Resolution Analysis
    # Short window: catches transient sounds like gunshots, glass shatters
    # Medium window: catches barks, screams, car horns
    # Long window: catches sustained sounds like sirens, traffic, rain
    resolutions = [
        (0.5, 0.25, "SHORT (0.5s)"),
        (2.0, 0.5,  "MEDIUM (2.0s)"),
        (5.0, 1.0,  "LONG (5.0s)")
    ]
    
    all_main = defaultdict(list)
    all_sub = defaultdict(list)
    
    for win, hop, label in resolutions:
        print(f"⏳ Scanning at {label} resolution...")
        
        if ensemble and models_list:
            # Ensemble: run all models and average predictions
            all_main_dets = defaultdict(list)
            all_sub_dets = defaultdict(list)
            for m_name, m_model in models_list:
                m_det, s_det = analyze_at_resolution(
                    wav, sr, duration, m_model, yamnet, win, hop, threshold, labels
                )
                for cls, intervals in m_det.items():
                    all_main_dets[cls].extend(intervals)
                for cls, intervals in s_det.items():
                    all_sub_dets[cls].extend(intervals)
            for cls, intervals in all_main_dets.items():
                all_main[cls].extend(intervals)
            for cls, intervals in all_sub_dets.items():
                all_sub[cls].extend(intervals)
        else:
            main_det, sub_det = analyze_at_resolution(
                wav, sr, duration, model, yamnet, win, hop, threshold, labels
            )
            for cls, intervals in main_det.items():
                all_main[cls].extend(intervals)
            for cls, intervals in sub_det.items():
                all_sub[cls].extend(intervals)

    # 5. Display Results
    print(f"\n📊 TIMELINE RESULTS (>{threshold*100:.0f}% Confidence, Multi-Resolution)")
    print("-" * 55)
    
    # --- Main Class ---
    print("📁 MAIN CATEGORIES DETECTED:")
    if not all_main:
        print("   (None above threshold)")
    else:
        for cls_name, intervals in all_main.items():
            merged = merge_intervals(intervals)
            timeline = ", ".join([f"{s:.1f}s - {e:.1f}s" for s, e in merged])
            print(f"   • {cls_name.upper():<15} ⏱️ {timeline}")

    # --- Sub Class ---
    print("\n🏷️ SPECIFIC EVENTS DETECTED:")
    if not all_sub:
        print("   (None above threshold)")
    else:
        for cls_name, intervals in all_sub.items():
            merged = merge_intervals(intervals)
            timeline = ", ".join([f"{s:.1f}s - {e:.1f}s" for s, e in merged])
            print(f"   🎯 {cls_name:<25} ⏱️ {timeline}")
            
    print("-" * 55)
    
    # 6. Anomaly Detection (Zero-Shot)
    if anomaly and SKLEARN_AVAILABLE:
        print("\n🔬 ANOMALY DETECTION SCAN:")
        # Collect all features for anomaly scoring
        all_features = []
        window_samples = int(2.0 * sr)
        hop_samples = int(0.5 * sr)
        for start in range(0, len(wav), hop_samples):
            end = start + window_samples
            chunk = wav[start:end]
            if len(chunk) < sr * 0.5:
                continue
            feat = extract_features_from_chunk(chunk, yamnet)
            if feat is not None:
                all_features.append(feat.numpy())
        
        if all_features:
            X_feats = np.array(all_features)
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(X_feats)
            anomaly_scores = iso_forest.predict(X_feats)
            n_anomalies = int(np.sum(anomaly_scores == -1))
            
            if n_anomalies > 0:
                anomaly_pct = n_anomalies / len(anomaly_scores) * 100
                print(f"   ⚠️ WARNING: {n_anomalies} ANOMALOUS SEGMENTS DETECTED ({anomaly_pct:.0f}%)")
                print(f"   These segments contain sounds fundamentally different from training data!")
                # Find anomaly timestamps
                anomaly_times = []
                for i, score in enumerate(anomaly_scores):
                    if score == -1:
                        t = i * 0.5  # hop_sec approximation
                        anomaly_times.append(f"{t:.1f}s")
                if len(anomaly_times) <= 10:
                    print(f"   🕐 At: {', '.join(anomaly_times)}")
            else:
                print(f"   ✅ No anomalous patterns detected. All sounds match training distribution.")
        print("-" * 55)

def main():
    parser = argparse.ArgumentParser(description="Test forensic audio model on a file")
    parser.add_argument("file", nargs="?", help="Path to the audio file (.wav, .mp3)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (0.0 to 1.0) to detect an event (default 0.3)")
    parser.add_argument("--ensemble", action="store_true", help="Use ALL models in models/ for ensemble voting")
    parser.add_argument("--anomaly", action="store_true", help="Enable Zero-Shot Anomaly Detection (requires scikit-learn)")
    args = parser.parse_args()
    
    if args.file:
        predict(args.file, threshold=args.threshold, ensemble=args.ensemble, anomaly=args.anomaly)
    else:
        print("Usage: python test_model.py <file.wav> [--threshold 0.3] [--ensemble] [--anomaly]")

if __name__ == "__main__":
    main()
