# Custom Forensic Audio Model Training Guide

This guide will help you train a custom audio classification model for forensic sound detection.

## Prerequisites

### 1. Python Environment
```bash
pip install tensorflow tensorflow-hub librosa soundfile numpy pandas scikit-learn matplotlib
```

### 2. Dataset Preparation
Create the following folder structure inside `scripts/training/`:

```
dataset/
├── gunshot/          # 50+ audio files
├── glass_breaking/   # 50+ audio files
├── voice/            # 50+ audio files
├── vehicle/          # 50+ audio files
├── explosion/        # 50+ audio files
├── barking/          # 50+ audio files
├── scream/           # 50+ audio files
└── ambient/          # 50+ background noise files
```

### Audio File Requirements
- **Format**: WAV (preferred) or MP3
- **Sample Rate**: 16kHz or higher (will be resampled)
- **Duration**: 1-10 seconds per clip
- **Quality**: Clear examples of the sound class

## Training Steps

### Step 1: Organize Your Data
Place audio files in the appropriate class folders.

### Step 2: Run Data Preprocessing
```bash
python preprocess_audio.py
```
This will:
- Resample all audio to 16kHz mono
- Validate file integrity
- Generate a data manifest

### Step 3: Train the Model
```bash
python train_forensic_model.py --epochs 50 --batch_size 32
```

### Step 4: Export to TFLite
```bash
python export_to_tflite.py
```

### Step 5: Replace the Model
Copy the generated `forensic_model.tflite` to replace `yamnet.tflite`.

## Data Collection Tips

### Gunshots
- Use movie sound effects (royalty-free)
- Record at shooting ranges (with permission)
- Include different gun types

### Glass Breaking
- Window breaks, bottle smashes
- Different glass thicknesses

### Voices
- Conversations, shouts, whispers
- Male/female voices
- Different languages

### Vehicles
- Cars, motorcycles, trucks
- Engine sounds, horns
- Include passing vehicles

## Troubleshooting

### "Not enough training data"
Minimum 50 samples per class. Use data augmentation if needed.

### "Model not improving"
- Check audio quality
- Ensure classes are distinct
- Try more epochs
