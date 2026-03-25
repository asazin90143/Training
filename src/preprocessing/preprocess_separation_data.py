"""
Separation Dataset Preprocessing Script (Phase 2 — Step 1)
Standalone preprocessor for voice separation datasets ONLY.

This script is COMPLETELY ISOLATED from the classification preprocessing pipeline.
It will NEVER touch your existing D:\\dataset or D:\\processed data.

Usage:
    python src/preprocessing/preprocess_separation_data.py
    python src/preprocessing/preprocess_separation_data.py --source "E:\\separation_dataset"
    python src/preprocessing/preprocess_separation_data.py --dest "E:\\separation_processed"

Input:  D:\\separation_dataset  (downloaded by download_separation_datasets.py)
Output: D:\\separation_processed
"""

import os
import sys
import json
import random
import time
import argparse
import numpy as np
import concurrent.futures
from pathlib import Path
from datetime import datetime

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import audio libraries
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("❌ librosa/soundfile not installed. Run: pip install librosa soundfile")

# ─── Configuration ────────────────────────────────────────────────────────
DEFAULT_SOURCE = r"D:\separation_dataset"
DEFAULT_DEST = r"D:\separation_processed"

TARGET_SR = 16000       # 16kHz standard
CHUNK_DURATION = 5.0    # seconds per chunk
MIN_DURATION = 0.5      # minimum useful duration
MAX_WORKERS = 4         # parallel processing workers


# ─── Audio Processing Functions ───────────────────────────────────────────

def validate_audio(file_path):
    """Check if an audio file is valid and meets quality thresholds."""
    try:
        info = sf.info(str(file_path))
        if info.duration < MIN_DURATION:
            return False, "too_short"
        return True, "ok"
    except Exception:
        return False, "corrupt"


def load_and_normalize(file_path, target_sr=TARGET_SR):
    """Load audio, convert to mono, resample, and normalize amplitude."""
    try:
        wav, sr = librosa.load(str(file_path), sr=target_sr, mono=True)

        # Remove silence from edges
        wav, _ = librosa.effects.trim(wav, top_db=30)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val * 0.95

        return wav, target_sr
    except Exception as e:
        return None, str(e)


def chunk_audio(wav, sr, chunk_duration=CHUNK_DURATION):
    """Split audio into fixed-length chunks."""
    chunk_samples = int(chunk_duration * sr)
    chunks = []

    for start in range(0, len(wav), chunk_samples):
        chunk = wav[start:start + chunk_samples]
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
        chunks.append(chunk)

    return chunks


def apply_forensic_augmentations(wav, sr, noise_files=None):
    """
    Apply forensic-grade augmentations for separation training.
    - Pitch shifting (simulate different microphones)
    - Time stretching (simulate different playback speeds)
    - SNR noise mixing (simulate real forensic environments)
    """
    augmented = []

    # Original
    augmented.append(("original", wav.copy()))

    # Pitch shift (±2 semitones) — simulates different microphone profiles
    try:
        pitch_up = librosa.effects.pitch_shift(wav, sr=sr, n_steps=1.5)
        augmented.append(("pitch_up", pitch_up))

        pitch_down = librosa.effects.pitch_shift(wav, sr=sr, n_steps=-1.5)
        augmented.append(("pitch_down", pitch_down))
    except Exception:
        pass

    # Time stretch (0.9x and 1.1x)
    try:
        slow = librosa.effects.time_stretch(wav, rate=0.9)
        augmented.append(("slow", slow))

        fast = librosa.effects.time_stretch(wav, rate=1.1)
        augmented.append(("fast", fast))
    except Exception:
        pass

    # SNR noise mixing (if noise files available)
    if noise_files and len(noise_files) > 0:
        noise_path = random.choice(noise_files)
        try:
            noise, _ = librosa.load(str(noise_path), sr=sr, mono=True)
            if len(noise) > 0:
                # Tile noise to match signal length
                if len(noise) < len(wav):
                    noise = np.tile(noise, int(np.ceil(len(wav) / len(noise))))
                noise = noise[:len(wav)]

                for snr_db, label in [(5, "snr5"), (10, "snr10"), (20, "snr20")]:
                    signal_power = np.mean(wav ** 2)
                    noise_power = np.mean(noise ** 2)
                    if noise_power > 0:
                        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
                        mixed = wav + scale * noise
                        # Normalize
                        max_val = np.max(np.abs(mixed))
                        if max_val > 0:
                            mixed = mixed / max_val * 0.95
                        augmented.append((label, mixed))
        except Exception:
            pass

    return augmented


def process_single_file(args):
    """Process a single audio file (used by multiprocessing pool)."""
    file_path, dest_dir, noise_files, file_idx = args

    try:
        valid, reason = validate_audio(file_path)
        if not valid:
            return {"status": "skipped", "reason": reason, "path": str(file_path)}

        wav, sr = load_and_normalize(file_path)
        if wav is None:
            return {"status": "error", "reason": sr, "path": str(file_path)}

        # Determine category from directory structure
        rel = file_path.relative_to(file_path.parent.parent)
        category = file_path.parent.name

        # Apply augmentations
        augmented_pairs = apply_forensic_augmentations(wav, sr, noise_files)

        saved_count = 0
        records = []

        for aug_name, aug_wav in augmented_pairs:
            chunks = chunk_audio(aug_wav, sr)

            for chunk_idx, chunk in enumerate(chunks):
                # Build output filename
                stem = file_path.stem
                out_name = f"{stem}_{aug_name}_chunk{chunk_idx:03d}.wav"
                out_dir = dest_dir / category
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / out_name

                sf.write(str(out_path), chunk, sr)
                saved_count += 1

                records.append({
                    "path": str(out_path),
                    "category": category,
                    "source": str(file_path),
                    "augmentation": aug_name,
                    "chunk_index": chunk_idx,
                    "duration": len(chunk) / sr
                })

        return {"status": "ok", "saved": saved_count, "records": records}
    except Exception as e:
        return {"status": "error", "reason": str(e), "path": str(file_path)}


# ─── Dataset Discovery ───────────────────────────────────────────────────

def discover_audio_files(source_dir):
    """Recursively find all audio files in the source directory."""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    source = Path(source_dir)
    files = []

    for ext in extensions:
        files.extend(source.rglob(f"*{ext}"))

    return sorted(files)


def discover_noise_files(source_dir):
    """Find noise files from MUSAN or DNS datasets for augmentation."""
    noise_dirs = [
        Path(source_dir) / "musan" / "noise",
        Path(source_dir) / "musan" / "music",
        Path(source_dir) / "dns_challenge",
        Path(source_dir) / "wham_noise",
    ]

    noise_files = []
    for d in noise_dirs:
        if d.exists():
            noise_files.extend(d.rglob("*.wav"))

    return noise_files[:500]  # Cap to prevent memory issues


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess separation datasets (STANDALONE — does NOT touch classification data)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/preprocessing/preprocess_separation_data.py
  python src/preprocessing/preprocess_separation_data.py --source "E:\\separation_dataset"
  python src/preprocessing/preprocess_separation_data.py --dest "E:\\separation_processed"
  python src/preprocessing/preprocess_separation_data.py --workers 8
        """
    )
    parser.add_argument("--source", type=str, default=DEFAULT_SOURCE,
                        help=f"Source directory with raw separation datasets (default: {DEFAULT_SOURCE})")
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST,
                        help=f"Destination for processed data (default: {DEFAULT_DEST})")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help=f"Number of parallel workers (default: {MAX_WORKERS})")
    parser.add_argument("--no_augment", action="store_true",
                        help="Skip augmentation (faster, less data)")
    args = parser.parse_args()

    if not LIBROSA_AVAILABLE:
        print("❌ Cannot proceed without librosa. Run: pip install librosa soundfile")
        return

    source_dir = Path(args.source)
    dest_dir = Path(args.dest)

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        print(f"   Run download_separation_datasets.py first!")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔧 SEPARATION DATA PREPROCESSOR (STANDALONE)")
    print("=" * 60)
    print(f"  Source:       {source_dir}")
    print(f"  Destination:  {dest_dir}")
    print(f"  Workers:      {args.workers}")
    print(f"  Augmentation: {'Disabled' if args.no_augment else 'Enabled'}")
    print(f"  ⚠️  This script ONLY touches separation data!")
    print(f"  ⚠️  Your classification data at D:\\dataset is UNTOUCHED.")
    print()

    # Discover audio files
    print("📂 Scanning for audio files...")
    audio_files = discover_audio_files(source_dir)
    print(f"   Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        print("❌ No audio files found. Download datasets first!")
        return

    # Discover noise files for augmentation
    noise_files = []
    if not args.no_augment:
        print("🔊 Scanning for noise files (MUSAN, WHAM, DNS)...")
        noise_files = discover_noise_files(source_dir)
        print(f"   Found {len(noise_files)} noise files for augmentation")

    # Process files
    print(f"\n🔄 Processing {len(audio_files)} files with {args.workers} workers...")
    start_time = time.time()

    tasks = [
        (f, dest_dir, noise_files if not args.no_augment else None, i)
        for i, f in enumerate(audio_files)
    ]

    all_records = []
    ok_count = 0
    skip_count = 0
    err_count = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, result in enumerate(pool.map(process_single_file, tasks, chunksize=10)):
            if result["status"] == "ok":
                ok_count += 1
                all_records.extend(result.get("records", []))
            elif result["status"] == "skipped":
                skip_count += 1
            else:
                err_count += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"   [{i+1}/{len(tasks)}] ✅ {ok_count} | ⏭️ {skip_count} | ❌ {err_count} | {rate:.1f} files/s")

    elapsed = time.time() - start_time

    # Save manifest
    manifest = {
        "created": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "total_source_files": len(audio_files),
        "total_processed_chunks": len(all_records),
        "ok": ok_count,
        "skipped": skip_count,
        "errors": err_count,
        "categories": list(set(r["category"] for r in all_records)),
        "samples": all_records
    }

    manifest_path = dest_dir / "separation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print("📊 PREPROCESSING SUMMARY")
    print(f"{'=' * 60}")
    print(f"  ✅ Processed:  {ok_count} files")
    print(f"  ⏭️  Skipped:    {skip_count} files")
    print(f"  ❌ Errors:     {err_count} files")
    print(f"  📦 Total chunks: {len(all_records)}")
    print(f"  📁 Categories: {len(manifest['categories'])}")
    print(f"  📄 Manifest:   {manifest_path}")
    print(f"  ⏱️  Time:       {elapsed/60:.1f} minutes")
    print(f"\n  Next step: Run 'python src/training/train_separation_model.py'")


if __name__ == "__main__":
    main()
