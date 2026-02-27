"""
Audio Preprocessing Script for Hierarchical Forensic Model Training
Prepares audio files for training by standardizing format and creating manifest.

Supports a two-level hierarchy:
  E:\dataset\
    main_class/       (e.g., vehicle, effect, human)
      sub_class/      (e.g., siren, gunshot, scream)
        *.wav / *.mp3
"""

import os
import json
import wave
import struct
import random
import numpy as np
from pathlib import Path

# Try to import optional dependencies
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("⚠️ librosa not installed. Run: pip install librosa soundfile")

# Configuration
from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = PATHS["dataset"]
PROCESSED_DIR = PATHS["processed"]
TARGET_SR = 16000  # 16kHz for YAMNet compatibility
TARGET_DURATION = 5.0  # seconds

# Data Augmentation Configuration
AUGMENTATION_ENABLED = True
AUGMENTATION_MULTIPLIER = 3

# Audio Quality Thresholds
MIN_DURATION_SECONDS = 0.5
MIN_RMS_THRESHOLD = 0.01
MAX_CLIPPING_RATIO = 0.1


def discover_classes(dataset_dir: Path):
    """
    Dynamically discover main classes and subclasses from the folder structure.
    Returns:
        main_classes: sorted list of main class names
        sub_classes: sorted list of ALL subclass names ("main/sub" format)
        hierarchy: dict mapping main_class -> [sub_class, ...]
    """
    hierarchy = {}
    
    for main_dir in sorted(dataset_dir.iterdir()):
        if not main_dir.is_dir():
            continue
        
        main_name = main_dir.name
        subs = []
        
        for sub_dir in sorted(main_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            audio_count = len(list(sub_dir.glob("*.wav")) + list(sub_dir.glob("*.mp3")) + list(sub_dir.glob("*.ogg")))
            if audio_count > 0:
                subs.append(sub_dir.name)
        
        if len(subs) > 0:
            hierarchy[main_name] = subs
    
    main_classes = sorted(hierarchy.keys())
    
    sub_classes = []
    for main_cls in main_classes:
        for sub_cls in hierarchy[main_cls]:
            sub_classes.append(f"{main_cls}/{sub_cls}")
    
    return main_classes, sub_classes, hierarchy


def augment_audio(audio: np.ndarray, sr: int, ambient_noises: list = None) -> list:
    """Generate augmented versions of an audio clip."""
    augmented = []
    
    try:
        pitched_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append((pitched_up, "pitch_up"))
    except Exception:
        pass
    
    try:
        pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        augmented.append((pitched_down, "pitch_down"))
    except Exception:
        pass
    
    try:
        noise = np.random.normal(0, 0.005, len(audio))
        noisy = audio + noise
        noisy = noisy / np.max(np.abs(noisy))
        augmented.append((noisy, "noisy"))
    except Exception:
        pass
    
    try:
        stretched = librosa.effects.time_stretch(audio, rate=1.1)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmented.append((stretched, "fast"))
    except Exception:
        pass
    
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmented.append((stretched, "slow"))
    except Exception:
        pass
    
    if ambient_noises and len(ambient_noises) > 0:
        # Mix at multiple SNR levels for robustness
        snr_levels = [("mixed_faint", 0.15), ("mixed_med", 0.3), ("mixed_loud", 0.5)]
        for suffix, noise_vol in snr_levels:
            try:
                bg_noise = random.choice(ambient_noises)
                if len(bg_noise) < len(audio):
                    repeats = int(np.ceil(len(audio) / len(bg_noise)))
                    bg_noise = np.tile(bg_noise, repeats)
                bg_noise = bg_noise[:len(audio)]
                mixed = (audio * 0.8) + (bg_noise * noise_vol)
                mixed = mixed / np.max(np.abs(mixed))
                augmented.append((mixed, suffix))
            except Exception:
                pass
    
    # SpecAugment: Frequency Masking
    try:
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_masked = S.copy()
        n_mels = S_masked.shape[0]
        # Mask 1-3 random frequency bands
        for _ in range(random.randint(1, 3)):
            f_start = random.randint(0, n_mels - 20)
            f_width = random.randint(5, 20)
            S_masked[f_start:f_start + f_width, :] = 0
        # Reconstruct audio from masked spectrogram
        masked_audio = librosa.feature.inverse.mel_to_audio(S_masked, sr=sr)
        if len(masked_audio) > len(audio):
            masked_audio = masked_audio[:len(audio)]
        else:
            masked_audio = np.pad(masked_audio, (0, max(0, len(audio) - len(masked_audio))))
        masked_audio = masked_audio / (np.max(np.abs(masked_audio)) + 1e-8)
        augmented.append((masked_audio, "freq_masked"))
    except Exception:
        pass
    
    return augmented[:AUGMENTATION_MULTIPLIER]


def validate_audio_quality(file_path: Path) -> tuple:
    """Check audio file for quality issues."""
    issues = []
    try:
        audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        duration = len(audio) / sr
        
        if duration < MIN_DURATION_SECONDS:
            issues.append(f"Too short ({duration:.2f}s)")
        
        rms = np.sqrt(np.mean(audio**2))
        if rms < MIN_RMS_THRESHOLD:
            issues.append(f"Too quiet (RMS={rms:.4f})")
        
        peak = np.max(np.abs(audio))
        if peak > 0:
            clipping_samples = np.sum(np.abs(audio) > 0.99 * peak)
            clipping_ratio = clipping_samples / len(audio)
            if clipping_ratio > MAX_CLIPPING_RATIO:
                issues.append(f"Clipping ({clipping_ratio*100:.1f}%)")
        
        if np.all(audio == 0):
            issues.append("All zeros")
        
        return len(issues) == 0, issues
    except Exception as e:
        return False, [f"Cannot read: {e}"]


def load_and_preprocess_audio(file_path: Path) -> np.ndarray:
    """Load audio file and preprocess to standard format."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required")
    
    try:
        audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        target_length = int(TARGET_DURATION * TARGET_SR)
        
        if len(audio) < target_length:
            padding = target_length - len(audio)
            offset = padding // 2
            audio = np.pad(audio, (offset, padding - offset), mode='constant')
        else:
            frame_length = 1024
            hop_length = 512
            rmse = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
            frames_needed = int((target_length / float(len(audio))) * len(rmse))
            
            if frames_needed < len(rmse):
                current_sum = np.sum(rmse[:frames_needed])
                max_sum = current_sum
                max_start_frame = 0
                for i in range(1, len(rmse) - frames_needed):
                    current_sum = current_sum - rmse[i-1] + rmse[i+frames_needed-1]
                    if current_sum > max_sum:
                        max_sum = current_sum
                        max_start_frame = i
                start_sample = librosa.frames_to_samples(max_start_frame, hop_length=hop_length)
                end_sample = min(start_sample + target_length, len(audio))
                start_sample = end_sample - target_length
                audio = audio[start_sample:end_sample]
            else:
                start = (len(audio) - target_length) // 2
                audio = audio[start:start+target_length]
        
        return audio
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")
        return None


def process_dataset():
    """Process all audio files using the hierarchical folder structure."""
    if not LIBROSA_AVAILABLE:
        print("❌ Cannot process without librosa.")
        return None
    
    print(f"\n🔄 Discovering dataset structure from: {DATASET_DIR}")
    main_classes, sub_classes, hierarchy = discover_classes(DATASET_DIR)
    
    if not main_classes:
        print("❌ No valid class folders found!")
        return None
    
    print(f"\n📊 Discovered {len(main_classes)} main classes, {len(sub_classes)} subclasses:")
    for mc in main_classes:
        subs = hierarchy[mc]
        print(f"  📁 {mc}: {len(subs)} subclasses → {', '.join(subs[:5])}{'...' if len(subs) > 5 else ''}")
    
    manifest = {
        "main_classes": main_classes,
        "sub_classes": sub_classes,
        "hierarchy": hierarchy,
        "samples": [],
        "statistics": {}
    }
    
    # Pre-load ambient/environment noises for mixing
    print("\n  🎵 Pre-loading background noises for mixing...")
    ambient_noises = []
    
    # Source 1: Environment folder (existing dataset)
    env_dir = DATASET_DIR / "environment"
    if env_dir.exists():
        for sub_dir in env_dir.iterdir():
            if not sub_dir.is_dir():
                continue
            for f in list(sub_dir.glob("*.wav"))[:50] + list(sub_dir.glob("*.mp3"))[:50]:
                try:
                    audio, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
                    if len(audio) > TARGET_SR * 1.0:
                        ambient_noises.append(audio)
                except:
                    pass
    
    # Source 2: Dedicated background noise folder (optional, for realism)
    bg_noise_dir = DATASET_DIR / "_background_noise"
    if bg_noise_dir.exists():
        print("    📂 Found _background_noise folder, loading extras...")
        for f in list(bg_noise_dir.rglob("*.wav")) + list(bg_noise_dir.rglob("*.mp3")):
            try:
                audio, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
                if len(audio) > TARGET_SR * 1.0:
                    ambient_noises.append(audio)
            except:
                pass
    
    if len(ambient_noises) > 300:
        random.shuffle(ambient_noises)
        ambient_noises = ambient_noises[:300]
    print(f"    ✅ Loaded {len(ambient_noises)} background tracks")
    
    # Create processed directory
    print(f"\n  💾 Processed data will be saved to: {PROCESSED_DIR}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_skipped = 0
    
    for main_cls in main_classes:
        main_cls_id = main_classes.index(main_cls)
        
        for sub_cls in hierarchy[main_cls]:
            sub_cls_full = f"{main_cls}/{sub_cls}"
            sub_cls_id = sub_classes.index(sub_cls_full)
            
            src_dir = DATASET_DIR / main_cls / sub_cls
            proc_dir = PROCESSED_DIR / main_cls / sub_cls
            proc_dir.mkdir(parents=True, exist_ok=True)
            
            audio_files = list(src_dir.glob("*.wav")) + list(src_dir.glob("*.mp3")) + list(src_dir.glob("*.ogg"))
            
            if len(audio_files) == 0:
                continue
            
            print(f"\n  📂 [{main_cls}] {sub_cls}: {len(audio_files)} files")
            
            processed_count = 0
            skipped_count = 0
            
            for audio_file in audio_files:
                is_valid, issues = validate_audio_quality(audio_file)
                if not is_valid:
                    skipped_count += 1
                    continue
                
                audio = load_and_preprocess_audio(audio_file)
                
                if audio is not None:
                    # Save original
                    output_path = proc_dir / f"{audio_file.stem}_processed.wav"
                    sf.write(str(output_path), audio, TARGET_SR)
                    
                    # Store ABSOLUTE path so training script can find it on external drive
                    manifest["samples"].append({
                        "file": str(output_path),
                        "main_class": main_cls,
                        "main_class_id": main_cls_id,
                        "sub_class": sub_cls,
                        "sub_class_full": sub_cls_full,
                        "sub_class_id": sub_cls_id,
                        "duration": TARGET_DURATION,
                        "augmented": False
                    })
                    processed_count += 1
                    
                    # Augment
                    if AUGMENTATION_ENABLED:
                        bg = ambient_noises if main_cls != "environment" else []
                        augmented_versions = augment_audio(audio, TARGET_SR, bg)
                        for aug_audio, aug_suffix in augmented_versions:
                            aug_path = proc_dir / f"{audio_file.stem}_{aug_suffix}.wav"
                            sf.write(str(aug_path), aug_audio, TARGET_SR)
                            
                            manifest["samples"].append({
                                "file": str(aug_path),
                                "main_class": main_cls,
                                "main_class_id": main_cls_id,
                                "sub_class": sub_cls,
                                "sub_class_full": sub_cls_full,
                                "sub_class_id": sub_cls_id,
                                "duration": TARGET_DURATION,
                                "augmented": True
                            })
                            processed_count += 1
            
            manifest["statistics"][sub_cls_full] = processed_count
            total_processed += processed_count
            total_skipped += skipped_count
            
            if skipped_count > 0:
                print(f"    ✅ Processed: {processed_count} | ⚠️ Skipped: {skipped_count}")
            else:
                print(f"    ✅ Processed: {processed_count} (incl. augmented)")
    
    print(f"\n{'='*50}")
    print(f"📊 TOTAL: {total_processed} processed, {total_skipped} skipped")
    
    return manifest


def generate_manifest(manifest: dict):
    """Save the data manifest."""
    # Save manifest locally (with the code) so training script can find it
    manifest_path = PATHS["manifest"]
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n📋 Manifest saved to: {manifest_path}")
    
    # Also save a backup copy next to the processed data on the external drive
    external_manifest = PROCESSED_DIR / "data_manifest.json"
    with open(external_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Backup manifest saved to: {external_manifest}")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY (Hierarchical)")
    print("="*50)
    
    main_classes = manifest.get("main_classes", [])
    hierarchy = manifest.get("hierarchy", {})
    stats = manifest.get("statistics", {})
    
    grand_total = 0
    for mc in main_classes:
        mc_total = 0
        subs = hierarchy.get(mc, [])
        for sc in subs:
            key = f"{mc}/{sc}"
            count = stats.get(key, 0)
            mc_total += count
        
        print(f"\n  📁 {mc.upper()} ({mc_total} total)")
        for sc in subs:
            key = f"{mc}/{sc}"
            count = stats.get(key, 0)
            status = "✅" if count >= 50 else "⚠️"
            print(f"      {sc}: {count} {status}")
        
        grand_total += mc_total
    
    print(f"\n  🎯 Grand Total: {grand_total} samples")
    print(f"  📦 Main Classes: {len(main_classes)}")
    print(f"  🏷️ Subclasses: {len(manifest.get('sub_classes', []))}")


def check_dataset_status():
    """Check current dataset status."""
    print(f"\n📊 DATASET STATUS (Source: {DATASET_DIR})")
    print("="*50)
    
    if not DATASET_DIR.exists():
        print(f"  ❌ Dataset directory not found: {DATASET_DIR}")
        print(f"     Make sure your external drive is connected!")
        return
    
    main_classes, sub_classes, hierarchy = discover_classes(DATASET_DIR)
    
    for mc in main_classes:
        subs = hierarchy[mc]
        mc_total = 0
        for sc in subs:
            sc_dir = DATASET_DIR / mc / sc
            count = len(list(sc_dir.glob("*.wav")) + list(sc_dir.glob("*.mp3")) + list(sc_dir.glob("*.ogg")))
            mc_total += count
        
        print(f"  📁 {mc}: {mc_total} files across {len(subs)} subclasses")


def main():
    print("="*50)
    print("🎵 HIERARCHICAL AUDIO PREPROCESSING TOOL")
    print("="*50)
    print(f"  Dataset:   {DATASET_DIR}")
    print(f"  Processed: {PROCESSED_DIR}")
    
    check_dataset_status()
    
    if LIBROSA_AVAILABLE:
        manifest = process_dataset()
        
        if manifest and len(manifest["samples"]) > 0:
            generate_manifest(manifest)
            print("\n✅ Preprocessing complete! Ready for training.")
        else:
            print("\n📝 No samples found. Check your dataset folder.")
    else:
        print("\n📦 Install required: pip install librosa soundfile numpy")


if __name__ == "__main__":
    main()
