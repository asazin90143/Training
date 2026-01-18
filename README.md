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
├── gunshot/          # gunshots, shooting range
├── glass_shatter/    # window breaking, bottle smash
├── human_scream/     # distress screams (not playing children)
├── siren/            # police, ambulance, fire truck
├── car_alarm/        # distinct repetitive alarm patterns
├── explosion/        # blasts, fireworks
├── dog_bark/         # aggressive barking
├── power_tools/      # drills, saws, angle grinders
├── aggressive_shout/ # angry shouting, fighting
└── ambient/          # background noise, wind, rain, traffic
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
- **Smart Crop**: Automatically finds the loudest 5-second window in your audio (centering on gunshots/events).
- Resample all audio to 16kHz mono.
- Generate a data manifest.

### Step 3: Train the Model
```bash
python train_forensic_model.py --epochs 50 --batch_size 32
```
This trains a custom classifier using **Mean + Max Pooling** for better short-event detection (like gunshots).
It will output a **Confusion Matrix** at the end to show exactly where errors are happening.

### Step 4: Export to TFLite
```bash
python export_to_tflite.py
```

## 🧪 Validating Your Model

You can test your trained model on any audio file (e.g., a recording from your phone) using the test script:

```bash
# Run on a specific file
python test_model.py "path/to/my_recording.wav"
```
It will analyze the audio and print the confidence percentages for every category.

## Data Collection Tips

### Gunshots
- Use movie sound effects (royalty-free)
- Record at shooting ranges (with permission)
- Include different gun types

### Glass Shatter
- Window breaks, bottle smashes
- Distinct high-frequency "crash" sound

### Human Scream
- Focus on distress/fear screams
- distinctly different from "loud playing" or cheering

### Sirens & Alarms
- **Siren**: Wail, Yelp, Hi-Lo patterns (Police/Fire/Ambulance)
- **Car Alarm**: Repetitive electronic honking or chirping

### Aggressive Shout
- Arguments, fighting words
- High energy speech
- Distinct from normal conversation

### Power Tools & Explosions
- **Power Tools**: Angle grinders (high pitch), drills, saws (forced entry sounds)
- **Explosion**: Booms, blasts, fireworks (low frequency)

## Troubleshooting

### "Not enough training data"
Minimum 50 samples per class. Use data augmentation if needed.

### "Model not improving"
- Check audio quality
- Ensure classes are distinct
- Try more epochs
