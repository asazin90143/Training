"""
Auto-Hard Negative Mining for Forensic Audio Model
Evaluates the trained model against the training set, identifies the most
confused class pairs, and generates a confusion report.

Usage:
    python evaluate_weaknesses.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import librosa

from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def load_latest_model():
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        return None, None
    latest = max(models, key=lambda p: p.stat().st_mtime)
    model = tf.keras.models.load_model(str(latest))
    labels_path = MODELS_DIR / f"{latest.stem}_labels.json"
    labels = None
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
    return model, labels


def extract_features(audio_path, yamnet):
    try:
        wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        _, emb, _ = yamnet(wav)
        if emb.shape[0] == 0:
            return None
        mean_emb = tf.reduce_mean(emb, axis=0)
        max_emb = tf.reduce_max(emb, axis=0)
        return tf.concat([mean_emb, max_emb], axis=0)
    except Exception:
        return None


def evaluate():
    model, labels = load_latest_model()
    if not model or not labels:
        print("❌ No model found.")
        return

    manifest_path = PATHS["manifest"]
    if not manifest_path.exists():
        print("❌ No manifest found. Run preprocess_audio.py first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("📦 Loading YAMNet...")
    yamnet = hub.load(YAMNET_MODEL_URL)

    main_classes = labels["main_classes"]
    sub_classes = labels["sub_classes"]

    # Track confusion: {true_class: {predicted_class: count}}
    main_confusion = defaultdict(lambda: defaultdict(int))
    sub_confusion = defaultdict(lambda: defaultdict(int))
    main_correct = 0
    sub_correct = 0
    total = 0
    errors = 0

    samples = manifest["samples"]
    print(f"🔍 Evaluating {len(samples)} samples...")

    for i, sample in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{len(samples)}...")

        path = Path(sample["file"])
        if not path.exists():
            try:
                parts = path.parts
                if "processed" in parts:
                    idx = parts.index("processed")
                    rel_parts = parts[idx+1:]
                    new_path = PATHS["processed"].joinpath(*rel_parts)
                    if new_path.exists():
                        path = new_path
            except Exception:
                pass

        if not path.exists():
            errors += 1
            continue

        feat = extract_features(str(path), yamnet)
        if feat is None:
            errors += 1
            continue

        X = tf.expand_dims(feat, 0)
        preds = model.predict(X, verbose=0)

        true_main = sample.get("main_class", "")
        true_sub = sample.get("sub_class_full", sample.get("sub_class", ""))

        pred_main_idx = int(np.argmax(preds[0][0]))
        pred_sub_idx = int(np.argmax(preds[1][0]))

        pred_main = main_classes[pred_main_idx] if pred_main_idx < len(main_classes) else "unknown"
        pred_sub = sub_classes[pred_sub_idx] if pred_sub_idx < len(sub_classes) else "unknown"

        main_confusion[true_main][pred_main] += 1
        sub_confusion[true_sub][pred_sub] += 1

        if pred_main == true_main:
            main_correct += 1
        if pred_sub == true_sub:
            sub_correct += 1
        total += 1

    # Generate confusion report
    print(f"\n{'='*60}")
    print(f"📊 WEAKNESS EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"   Total evaluated: {total}")
    print(f"   Main accuracy:   {main_correct/max(total,1)*100:.1f}%")
    print(f"   Sub accuracy:    {sub_correct/max(total,1)*100:.1f}%")
    print(f"   Errors/Skipped:  {errors}")

    # Find top confused pairs
    confused_pairs = []
    for true_cls, preds_dict in sub_confusion.items():
        for pred_cls, count in preds_dict.items():
            if true_cls != pred_cls and count > 2:
                confused_pairs.append({
                    "true": true_cls,
                    "predicted_as": pred_cls,
                    "count": count
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    print(f"\n⚠️ TOP CONFUSED CLASS PAIRS:")
    for pair in confused_pairs[:15]:
        print(f"   '{pair['true']}' misclassified as '{pair['predicted_as']}' ({pair['count']} times)")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat() if 'datetime' in dir() else "N/A",
        "total_evaluated": total,
        "main_accuracy": main_correct / max(total, 1),
        "sub_accuracy": sub_correct / max(total, 1),
        "confused_pairs": confused_pairs[:30],
        "main_confusion_matrix": {k: dict(v) for k, v in main_confusion.items()},
        "sub_confusion_matrix": {k: dict(v) for k, v in sub_confusion.items()}
    }

    report_path = MODELS_DIR / "weakness_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n💾 Full report saved to: {report_path}")
    print(f"💡 Use this report to add more training data for confused classes!")


if __name__ == "__main__":
    evaluate()
