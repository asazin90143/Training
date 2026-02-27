"""
Active Learning Pipeline for Forensic Audio Model
Scans a folder of unlabeled audio files, identifies uncertain predictions,
and moves them to a 'needs_review' folder for human labeling.

Usage:
    python active_learner.py "D:\\unlabeled_audio" --low 0.35 --high 0.65
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config import get_paths
PATHS = get_paths()

MODELS_DIR = PATHS["models_root"]
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def load_latest_model():
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        print("❌ No .keras models found in models/")
        return None, None
    latest = max(models, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading model: {latest.name}")
    model = tf.keras.models.load_model(str(latest))
    labels_path = MODELS_DIR / f"{latest.stem}_labels.json"
    labels = None
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
    return model, labels


def extract_features(audio_path, yamnet):
    try:
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        _, emb, _ = yamnet(wav)
        if emb.shape[0] == 0:
            return None
        mean_emb = tf.reduce_mean(emb, axis=0)
        max_emb = tf.reduce_max(emb, axis=0)
        return tf.concat([mean_emb, max_emb], axis=0)
    except Exception:
        return None


def scan_for_uncertainty(audio_dir, output_dir, low_thresh=0.35, high_thresh=0.65):
    """Scan audio files and move uncertain ones to review folder."""
    model, labels = load_latest_model()
    if not model or not labels:
        return

    print("📦 Loading YAMNet...")
    yamnet = hub.load(YAMNET_MODEL_URL)

    audio_dir = Path(audio_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = []
    for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac"]:
        audio_files.extend(list(audio_dir.rglob(ext)))

    print(f"🔍 Scanning {len(audio_files)} audio files...")
    print(f"   Uncertainty range: {low_thresh:.0%} - {high_thresh:.0%}")

    uncertain = 0
    confident = 0
    errors = 0
    review_log = []

    for i, audio_file in enumerate(audio_files):
        if (i + 1) % 50 == 0:
            print(f"   Processed {i+1}/{len(audio_files)}...")

        feat = extract_features(audio_file, yamnet)
        if feat is None:
            errors += 1
            continue

        X = tf.expand_dims(feat, 0)
        preds = model.predict(X, verbose=0)

        main_preds = preds[0][0]
        sub_preds = preds[1][0]

        max_main_conf = float(np.max(main_preds))
        max_sub_conf = float(np.max(sub_preds))

        # Check if any prediction falls in the uncertain zone
        is_uncertain = (low_thresh < max_main_conf < high_thresh) or \
                       (low_thresh < max_sub_conf < high_thresh)

        if is_uncertain:
            # Move to review folder
            dest = output_dir / audio_file.name
            shutil.copy2(str(audio_file), str(dest))
            uncertain += 1

            main_classes = labels.get("main_classes", [])
            sub_classes = labels.get("sub_classes", [])
            best_main = main_classes[int(np.argmax(main_preds))] if main_classes else "unknown"
            best_sub = sub_classes[int(np.argmax(sub_preds))] if sub_classes else "unknown"

            review_log.append({
                "file": audio_file.name,
                "best_main_guess": best_main,
                "main_confidence": max_main_conf,
                "best_sub_guess": best_sub,
                "sub_confidence": max_sub_conf
            })
        else:
            confident += 1

    # Save review log
    log_path = output_dir / "review_log.json"
    with open(log_path, "w") as f:
        json.dump(review_log, f, indent=2)

    print(f"\n{'='*50}")
    print(f"📊 ACTIVE LEARNING SCAN COMPLETE")
    print(f"{'='*50}")
    print(f"   ✅ Confident:  {confident}")
    print(f"   ❓ Uncertain:  {uncertain} → moved to {output_dir}")
    print(f"   ❌ Errors:     {errors}")
    print(f"   📋 Review log: {log_path}")
    print(f"\n💡 Listen to the files in '{output_dir}', manually label them,")
    print(f"   move them to the correct dataset folder, and retrain!")


def main():
    parser = argparse.ArgumentParser(description="Active Learning: Find uncertain audio for human review")
    parser.add_argument("audio_dir", help="Directory containing unlabeled audio files")
    parser.add_argument("--output", default=None, help="Output directory for uncertain files (default: audio_dir/needs_review)")
    parser.add_argument("--low", type=float, default=0.35, help="Lower confidence threshold (default: 0.35)")
    parser.add_argument("--high", type=float, default=0.65, help="Upper confidence threshold (default: 0.65)")
    args = parser.parse_args()

    output = args.output or str(Path(args.audio_dir) / "needs_review")
    scan_for_uncertainty(args.audio_dir, output, args.low, args.high)


if __name__ == "__main__":
    main()
