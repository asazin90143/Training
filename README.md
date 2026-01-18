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
- **Quality Validation**: Automatically skips silent, clipped, or corrupted files.
- **Smart Crop**: Finds the loudest 5-second window in your audio (centering on events).
- **Data Augmentation**: Creates 3 additional versions of each file (pitch shift, noise, time stretch).
- Resample all audio to 16kHz mono.
- Generate a data manifest.

### Step 3: Train the Model
```bash
python train_forensic_model.py --epochs 50 --batch_size 32
```
This trains a custom classifier with:
- **Mean + Max Pooling**: Better detection of short impulsive sounds (gunshots).
- **Automatic Class Balancing**: Rare classes get higher weight to prevent bias.
- **Training Graphs**: Accuracy/loss plots saved as PNG files.
- **Confusion Matrix**: Shows exactly where errors are happening.
- **Per-Class Report**: Precision, Recall, and F1-Score for each category.

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

## 🔄 Advanced: K-Fold Cross-Validation

For a more reliable accuracy estimate, use K-Fold cross-validation:

```bash
python train_kfold.py --folds 5
```
This trains the model 5 times, each time using a different 20% as the test set, and reports the **average accuracy ± standard deviation**.

## 🔐 Model Versioning

Track and compare different model versions:

```bash
# List all registered models
python model_registry.py list

# Register a new model
python model_registry.py register my_model 0.85 --notes "First attempt"

# Compare two versions
python model_registry.py compare v1 v2

# Auto-register from existing labels files
python model_registry.py auto
```

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
Minimum 50 samples per class. Data augmentation is now **enabled by default**, multiplying your dataset by 4x.

### "Model not improving"
- Check the **per-class report** for weak categories.
- Check the **training graphs** for overfitting (validation loss going up).
- Ensure classes are distinct.
- Try more epochs or adjust learning rate.

### "Files being skipped"
The quality validator will skip files that are:
- Too short (< 0.5 seconds)
- Silent (RMS < 0.01)
- Clipping (> 10% of samples at peak)

## 📁 Output Files

After training, check the `models/` folder for:

| File | Description |
|------|-------------|
| `*_classifier.keras` | Full Keras model (for further training) |
| `*_labels.json` | Class names and accuracy |
| `*_history.json` | Raw training metrics |
| `*_accuracy.png` | Accuracy plot |
| `*_loss.png` | Loss plot |
| `*_confusion_matrix.json` | Confusion matrix data |
| `*_class_report.json` | Per-class precision/recall/F1 |
