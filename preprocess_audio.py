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
    "glass_breaking", 
    "voice",
    "vehicle",
    "explosion",
    "barking",
    "scream",
    "ambient"  # Background/silence for negative class
]

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
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Trim to target length
            audio = audio[:target_length]
        
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
        for audio_file in audio_files:
            audio = load_and_preprocess_audio(audio_file)
            
            if audio is not None:
                # Save processed audio
                output_path = processed_cls_dir / f"{audio_file.stem}_processed.wav"
                sf.write(str(output_path), audio, TARGET_SR)
                
                manifest["samples"].append({
                    "file": str(output_path.relative_to(SCRIPT_DIR)),
                    "class": cls,
                    "class_id": FORENSIC_CLASSES.index(cls),
                    "duration": TARGET_DURATION
                })
                
                processed_count += 1
        
        manifest["statistics"][cls] = processed_count
        print(f"    ✅ Processed: {processed_count}/{len(audio_files)}")
    
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
