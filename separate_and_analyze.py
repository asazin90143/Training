"""
DEMUCS Audio Source Separation for Forensic Analysis
Uses Meta's DEMUCS model to split audio into separate stems before classification.
This isolates overlapping sounds (e.g., dog bark hidden in a siren).

Requires: pip install demucs

Usage:
    python separate_and_analyze.py "audio.mp3"
    python separate_and_analyze.py "audio.mp3" --threshold 0.2
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import tensorflow_hub as hub
import librosa

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def check_demucs():
    """Check if demucs is installed."""
    try:
        subprocess.run(["demucs", "--help"], capture_output=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def separate_audio(audio_path, output_dir):
    """Use DEMUCS to separate audio into stems."""
    print("🎛️ Running DEMUCS source separation...")
    print("   (This may take a few minutes for long audio)")
    
    cmd = [
        "demucs",
        "--two-stems", "vocals",  # Split into vocals + other
        "-o", str(output_dir),
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"   ⚠️ DEMUCS error: {result.stderr[:200]}")
            # Fallback: also try the 4-stem separation
            cmd_4stem = [
                "demucs",
                "-o", str(output_dir),
                str(audio_path)
            ]
            subprocess.run(cmd_4stem, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("   ⚠️ DEMUCS timed out (>10 minutes)")
        return []
    
    # Find separated stems
    stems = []
    audio_name = Path(audio_path).stem
    
    # Check for htdemucs output (default model)
    for model_name in ["htdemucs", "htdemucs_ft", "mdx_extra"]:
        stem_dir = output_dir / model_name / audio_name
        if stem_dir.exists():
            for stem_file in stem_dir.glob("*.wav"):
                stems.append(stem_file)
                print(f"   🎵 Found stem: {stem_file.name}")
    
    if not stems:
        print("   ⚠️ No stems found. Analyzing original file only.")
    
    return stems


def load_model_and_yamnet():
    """Load the forensic model and YAMNet."""
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        print("❌ No models found.")
        return None, None, None
    
    latest = max(models, key=lambda p: p.stat().st_mtime)
    model = tf.keras.models.load_model(str(latest))
    
    labels_path = MODELS_DIR / f"{latest.stem}_labels.json"
    labels = None
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
    
    yamnet = hub.load(YAMNET_MODEL_URL)
    return model, labels, yamnet


def analyze_stem(stem_path, model, yamnet, labels, threshold):
    """Run forensic analysis on a single separated stem."""
    wav, sr = librosa.load(str(stem_path), sr=16000, mono=True)
    
    main_classes = labels.get("main_classes", [])
    sub_classes = labels.get("sub_classes", [])
    
    window_samples = int(2.0 * sr)
    hop_samples = int(0.5 * sr)
    
    detections_main = defaultdict(list)
    detections_sub = defaultdict(list)
    duration = len(wav) / sr
    
    for start in range(0, len(wav), hop_samples):
        end = start + window_samples
        chunk = wav[start:end]
        if len(chunk) < sr * 0.5:
            continue
        
        try:
            _, emb, _ = yamnet(chunk)
            if emb.shape[0] == 0:
                continue
            mean_emb = tf.reduce_mean(emb, axis=0)
            max_emb = tf.reduce_max(emb, axis=0)
            feat = tf.concat([mean_emb, max_emb], axis=0)
            
            X = tf.expand_dims(feat, 0)
            preds = model.predict(X, verbose=0)
            
            start_t = start / sr
            end_t = min(end / sr, duration)
            
            for idx, conf in enumerate(preds[0][0]):
                if conf >= threshold and idx < len(main_classes):
                    detections_main[main_classes[idx]].append((start_t, end_t))
            for idx, conf in enumerate(preds[1][0]):
                if conf >= threshold and idx < len(sub_classes):
                    detections_sub[sub_classes[idx]].append((start_t, end_t))
        except Exception:
            continue
    
    return detections_main, detections_sub


def main():
    parser = argparse.ArgumentParser(description="DEMUCS Separate & Analyze")
    parser.add_argument("file", help="Audio file to separate and analyze")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--keep_stems", action="store_true", help="Keep separated stem files")
    args = parser.parse_args()
    
    if not check_demucs():
        print("❌ DEMUCS not installed. Run: pip install demucs")
        return
    
    print("="*50)
    print("🎛️ DEMUCS SOURCE SEPARATION + FORENSIC ANALYSIS")
    print("="*50)
    
    model, labels, yamnet = load_model_and_yamnet()
    if not model or not labels:
        return
    
    # Separate audio
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        stems = separate_audio(args.file, tmp_path)
        
        all_main = defaultdict(list)
        all_sub = defaultdict(list)
        
        # Analyze original file
        print(f"\n🔍 Analyzing original audio...")
        m, s = analyze_stem(args.file, model, yamnet, labels, args.threshold)
        for cls, intervals in m.items():
            all_main[cls].extend(intervals)
        for cls, intervals in s.items():
            all_sub[cls].extend(intervals)
        
        # Analyze each separated stem
        for stem in stems:
            print(f"\n🔍 Analyzing stem: {stem.name}")
            m, s = analyze_stem(stem, model, yamnet, labels, args.threshold)
            for cls, intervals in m.items():
                all_main[cls].extend(intervals)
            for cls, intervals in s.items():
                all_sub[cls].extend(intervals)
    
    # Display combined results
    print(f"\n{'='*55}")
    print(f"📊 DEMUCS-ENHANCED RESULTS (>{args.threshold*100:.0f}% Confidence)")
    print(f"{'='*55}")
    
    print("📁 MAIN CATEGORIES:")
    if not all_main:
        print("   (None)")
    else:
        for cls, intervals in all_main.items():
            print(f"   • {cls.upper()}")
    
    print("\n🏷️ SPECIFIC EVENTS:")
    if not all_sub:
        print("   (None)")
    else:
        for cls, intervals in all_sub.items():
            print(f"   🎯 {cls}")
    
    print(f"\n💡 DEMUCS separated the audio into {len(stems)} stems for deeper analysis!")


if __name__ == "__main__":
    main()
