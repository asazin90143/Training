"""
Audio Preprocessing Script for Custom Forensic Model Training
Prepares audio files for training by standardizing format and creating manifest.
"""

import os
import json
import wave
import struct
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
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset"
PROCESSED_DIR = SCRIPT_DIR / "processed"
TARGET_SR = 16000  # 16kHz for YAMNet compatibility
TARGET_DURATION = 5.0  # seconds

# Define forensic sound classes
FORENSIC_CLASSES = [
    "gunshot",
    "glass_shatter", 
    "human_scream",
    "siren",
    "car_alarm",
    "explosion",
    "dog_bark",
    "power_tools",     # Drills, saws (forced entry)
    "aggressive_shout", # Aggressive speech/fighting
    "ambient"          # Background/silence
]

# Data Augmentation Configuration
AUGMENTATION_ENABLED = True
AUGMENTATION_MULTIPLIER = 3  # Each file generates 3 augmented versions

def augment_audio(audio: np.ndarray, sr: int, ambient_noises: list = None) -> list:
    """
    Generate augmented versions of an audio clip.
    Returns a list of (audio_array, suffix_name) tuples.
    """
    augmented = []
    
    # 1. Pitch Shift (slightly higher)
    try:
        pitched_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append((pitched_up, "pitch_up"))
    except Exception:
        pass
    
    # 2. Pitch Shift (slightly lower)
    try:
        pitched_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        augmented.append((pitched_down, "pitch_down"))
    except Exception:
        pass
    
    # 3. Add background noise
    try:
        noise = np.random.normal(0, 0.005, len(audio))
        noisy = audio + noise
        noisy = noisy / np.max(np.abs(noisy))  # Re-normalize
        augmented.append((noisy, "noisy"))
    except Exception:
        pass
    
    # 4. Time stretch (slightly faster)
    try:
        stretched = librosa.effects.time_stretch(audio, rate=1.1)
        # Ensure same length
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmented.append((stretched, "fast"))
    except Exception:
        pass
    
    # 5. Time stretch (slightly slower)
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        augmented.append((stretched, "slow"))
    except Exception:
        pass
    
    # 6. Background Mixing (New!)
    if ambient_noises and len(ambient_noises) > 0:
        try:
            # Pick a random background noise
            bg_noise = random.choice(ambient_noises)
            
            # Ensure proper length (loop if too short, crop if too long)
            if len(bg_noise) < len(audio):
                repeats = int(np.ceil(len(audio) / len(bg_noise)))
                bg_noise = np.tile(bg_noise, repeats)
            
            bg_noise = bg_noise[:len(audio)]
            
            # Mix: Audio (0.8) + Noise (0.3)
            mixed = (audio * 0.8) + (bg_noise * 0.3)
            mixed = mixed / np.max(np.abs(mixed)) # Re-normalize
            augmented.append((mixed, "mixed_bg"))
        except Exception:
            pass
            
    # Return only the requested number of augmentations
    return augmented[:AUGMENTATION_MULTIPLIER]


def create_directory_structure():
    """Create the required directory structure."""
    print("📁 Creating directory structure...")
    
    DATASET_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)
    
    for cls in FORENSIC_CLASSES:
        (DATASET_DIR / cls).mkdir(exist_ok=True)
        (PROCESSED_DIR / cls).mkdir(exist_ok=True)
    
    print(f"✅ Created folders in: {DATASET_DIR}")
    print(f"   Classes: {', '.join(FORENSIC_CLASSES)}")

# Audio Quality Thresholds
MIN_DURATION_SECONDS = 0.5
MIN_RMS_THRESHOLD = 0.01  # Silence detection
MAX_CLIPPING_RATIO = 0.1  # Max percentage of samples at peak

def validate_audio_quality(file_path: Path) -> tuple:
    """
    Check audio file for quality issues.
    Returns (is_valid, issues_list).
    """
    issues = []
    
    try:
        audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        duration = len(audio) / sr
        
        # Check 1: Duration
        if duration < MIN_DURATION_SECONDS:
            issues.append(f"Too short ({duration:.2f}s < {MIN_DURATION_SECONDS}s)")
        
        # Check 2: Silence detection
        rms = np.sqrt(np.mean(audio**2))
        if rms < MIN_RMS_THRESHOLD:
            issues.append(f"Too quiet/silent (RMS={rms:.4f})")
        
        # Check 3: Clipping detection
        peak = np.max(np.abs(audio))
        if peak > 0:
            clipping_samples = np.sum(np.abs(audio) > 0.99 * peak)
            clipping_ratio = clipping_samples / len(audio)
            if clipping_ratio > MAX_CLIPPING_RATIO:
                issues.append(f"Clipping detected ({clipping_ratio*100:.1f}% at peak)")
        
        # Check 4: All zeros
        if np.all(audio == 0):
            issues.append("File contains only silence (all zeros)")
        
        is_valid = len(issues) == 0
        return is_valid, issues
        
    except Exception as e:
        return False, [f"Cannot read file: {e}"]


def load_and_preprocess_audio(file_path: Path) -> np.ndarray:
    """Load audio file and preprocess to standard format."""
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for audio processing")
    
    try:
        # Load audio with librosa (handles many formats)
        audio, sr = librosa.load(str(file_path), sr=TARGET_SR, mono=True)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Pad or trim to target duration
        target_length = int(TARGET_DURATION * TARGET_SR)
        
        if len(audio) < target_length:
            # Pad with zeros (center the audio)
            padding = target_length - len(audio)
            offset = padding // 2
            audio = np.pad(audio, (offset, padding - offset), mode='constant')
        else:
            # Smart Crop: Find 5s window with highest energy
            frame_length = 1024
            hop_length = 512
            
            # Calculate RMS energy
            rmse = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length).flatten()
            
            # frames needed for target duration
            frames_needed = int((target_length / float(len(audio))) * len(rmse))
            
            if frames_needed < len(rmse):
                # Slidng window sum over energy
                current_sum = np.sum(rmse[:frames_needed])
                max_sum = current_sum
                max_start_frame = 0
                
                # Efficient sliding window
                for i in range(1, len(rmse) - frames_needed):
                    current_sum = current_sum - rmse[i-1] + rmse[i+frames_needed-1]
                    if current_sum > max_sum:
                        max_sum = current_sum
                        max_start_frame = i
                
                # Convert frame index back to audio samples
                start_sample = librosa.frames_to_samples(max_start_frame, hop_length=hop_length)
                
                # Ensure we don't go out of bounds
                end_sample = min(start_sample + target_length, len(audio))
                start_sample = end_sample - target_length
                
                audio = audio[start_sample:end_sample]
            else:
                # Fallback to center crop
                start = (len(audio) - target_length) // 2
                audio = audio[start:start+target_length]
        
        return audio
        
    except Exception as e:
        print(f"  ❌ Error processing {file_path.name}: {e}")
        return None

def process_dataset():
    """Process all audio files in the dataset."""
    if not LIBROSA_AVAILABLE:
        print("❌ Cannot process without librosa. Please install it first.")
        return None
    
    print("\n🔄 Processing audio files...")
    
    manifest = {
        "classes": FORENSIC_CLASSES,
        "samples": [],
        "statistics": {}
    }
    
    # Load separate Ambient noises first for mixing
    print("  🎵 Pre-loading background noises for mixing...")
    ambient_noises = []
    ambient_dir = DATASET_DIR / "ambient"
    if ambient_dir.exists():
        for f in list(ambient_dir.glob("*.wav")) + list(ambient_dir.glob("*.mp3")):
            try:
                # Basic load just for mixing pool
                audio, _ = librosa.load(str(f), sr=TARGET_SR, mono=True)
                if len(audio) > TARGET_SR * 1.0: # Ignore tiny files
                    ambient_noises.append(audio)
            except:
                pass
    print(f"    ✅ Loaded {len(ambient_noises)} background tracks")
    
    for cls in FORENSIC_CLASSES:
        cls_dir = DATASET_DIR / cls
        processed_cls_dir = PROCESSED_DIR / cls
        
        if not cls_dir.exists():
            print(f"  ⚠️ Class folder not found: {cls}")
            continue
        
        audio_files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")) + list(cls_dir.glob("*.ogg"))
        
        if len(audio_files) == 0:
            print(f"  ⚠️ No audio files in: {cls}")
            continue
        
        print(f"  📂 Processing {cls}: {len(audio_files)} files")
        
        processed_count = 0
        skipped_count = 0
        for audio_file in audio_files:
            # Validate audio quality first
            is_valid, issues = validate_audio_quality(audio_file)
            if not is_valid:
                print(f"    ⚠️ Skipping {audio_file.name}: {'; '.join(issues)}")
                skipped_count += 1
                continue
            
            audio = load_and_preprocess_audio(audio_file)
            
            if audio is not None:
                # Save processed audio (original)
                output_path = processed_cls_dir / f"{audio_file.stem}_processed.wav"
                sf.write(str(output_path), audio, TARGET_SR)
                
                manifest["samples"].append({
                    "file": str(output_path.relative_to(SCRIPT_DIR)),
                    "class": cls,
                    "class_id": FORENSIC_CLASSES.index(cls),
                    "duration": TARGET_DURATION,
                    "augmented": False
                })
                
                processed_count += 1
                
                # Generate augmented versions
                if AUGMENTATION_ENABLED:
                    # Pass ambient noises (except for ambient class itself)
                    bg_tracks = ambient_noises if cls != "ambient" else []
                    augmented_versions = augment_audio(audio, TARGET_SR, bg_tracks)
                    for aug_audio, aug_suffix in augmented_versions:
                        aug_path = processed_cls_dir / f"{audio_file.stem}_{aug_suffix}.wav"
                        sf.write(str(aug_path), aug_audio, TARGET_SR)
                        
                        manifest["samples"].append({
                            "file": str(aug_path.relative_to(SCRIPT_DIR)),
                            "class": cls,
                            "class_id": FORENSIC_CLASSES.index(cls),
                            "duration": TARGET_DURATION,
                            "augmented": True
                        })
                        processed_count += 1
        
        manifest["statistics"][cls] = processed_count
        if skipped_count > 0:
            print(f"    ✅ Processed: {processed_count} | ⚠️ Skipped: {skipped_count}")
        else:
            print(f"    ✅ Processed: {processed_count} (including augmented)")
    
    return manifest

def generate_manifest(manifest: dict):
    """Save the data manifest."""
    manifest_path = SCRIPT_DIR / "data_manifest.json"
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n📋 Manifest saved to: {manifest_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("📊 DATASET SUMMARY")
    print("="*50)
    
    total = 0
    for cls, count in manifest.get("statistics", {}).items():
        status = "✅" if count >= 50 else "⚠️ (need more)"
        print(f"  {cls}: {count} samples {status}")
        total += count
    
    print(f"\n  Total samples: {total}")
    
    if total < 400:
        print("\n⚠️ WARNING: You have less than 50 samples per class.")
        print("   For best results, collect at least 50 samples per class.")

def check_dataset_status():
    """Check current dataset status without processing."""
    print("\n📊 CURRENT DATASET STATUS")
    print("="*50)
    
    for cls in FORENSIC_CLASSES:
        cls_dir = DATASET_DIR / cls
        if cls_dir.exists():
            audio_files = list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")) + list(cls_dir.glob("*.ogg"))
            status = "✅" if len(audio_files) >= 50 else "⚠️"
            print(f"  {cls}: {len(audio_files)} files {status}")
        else:
            print(f"  {cls}: 📁 Folder not created yet")

def main():
    print("="*50)
    print("🎵 FORENSIC AUDIO PREPROCESSING TOOL")
    print("="*50)
    
    # Step 1: Create directory structure
    create_directory_structure()
    
    # Step 2: Check current status
    check_dataset_status()
    
    # Step 3: Process if librosa is available
    if LIBROSA_AVAILABLE:
        manifest = process_dataset()
        
        if manifest and len(manifest["samples"]) > 0:
            generate_manifest(manifest)
            print("\n✅ Preprocessing complete! Ready for training.")
        else:
            print("\n📝 Next steps:")
            print("   1. Add audio files to the class folders in: scripts/training/dataset/")
            print("   2. Run this script again to process them")
    else:
        print("\n📦 To continue, install required packages:")
        print("   pip install librosa soundfile numpy")

if __name__ == "__main__":
    main()
