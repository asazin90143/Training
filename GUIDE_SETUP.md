# 📚 Complete Setup & Testing Guide

This guide walks you through preprocessing your external dataset, training the multi-label hierarchical model, and testing it.

---

## 🔗 The Full Workflow

Your data is configured to run from an **External Drive** to save space.

**Current Configuration (`config.py`):**
- **Raw Data:** `D:\dataset`
- **Processed Data:** `D:\processed`
- **Models:** Saved locally in `models/<backbone>/` subdirectories.
- **Manifest:** Saved locally as `data_manifest.json`.

---

### Step 1: Preprocess (Scan the Drive)
Scans `D:\dataset`, validates audio, normalizes, chunks to 5 seconds, and applies **SpecAugment frequency masking**, pitch shifting, and **Multi-SNR Background Noise Mixing** (15%, 30%, 50% volume).

*Pro Tip: Create `D:\dataset\_background_noise\` with custom noise files (wind, crowds, static).*

```bash
python src/preprocessing/preprocess_audio.py
```

### Step 2: Train the Model

```bash
# Standard training (YAMNet backbone, 100 epochs)
python src/training/train_forensic_model.py

# Fine-tune YAMNet's deep layers (more accurate)
python src/training/train_forensic_model.py --finetune

# Use VGGish backbone
python src/training/train_forensic_model.py --backbone vggish

# Train ALL backbones overnight (automated multi-backbone)
python src/training/train_forensic_model.py --all_models
```
*Early Stopping is enabled: training stops automatically when the model converges.*

### Step 3: Test New Audio

```bash
# Standard analysis
python src/testing/test_model.py "audio.mp3" --threshold 0.20

# Ensemble voting (uses ALL models across all backbone subdirectories)
python src/testing/test_model.py "audio.mp3" --ensemble

# Anomaly detection (flag unknown/unrecognized sounds)
python src/testing/test_model.py "audio.mp3" --anomaly

# Full power: ensemble + anomaly + low threshold
python src/testing/test_model.py "audio.mp3" --ensemble --anomaly --threshold 0.15
```

---

## 🧪 Advanced Training Scripts

### Vision Transformer (Spectrogram-based)
Converts audio to Mel-Spectrograms and trains ResNet50 or EfficientNet to visually classify sounds.
```bash
python src/training/train_spectrogram_model.py --architecture resnet50
python src/training/train_spectrogram_model.py --architecture efficientnet
```

### Wav2Vec 2.0 (Self-Supervised)
Uses Meta's Wav2Vec 2.0 for deeper acoustic understanding. Requires `pip install transformers torch`.
```bash
python src/training/train_wav2vec_model.py --epochs 50
```

### Automated Hyperparameter Tuning
Uses KerasTuner to automatically search for the best layer sizes, dropout, and learning rate. Requires `pip install keras-tuner`.
```bash
python src/training/tune_hyperparameters.py --max_trials 100
```

### Knowledge Distillation
Trains a tiny "student" model to mimic the combined output of all your trained "teacher" models.
```bash
python src/training/train_student_model.py --temperature 3.0
```

---

## 🔬 Analysis & Utility Tools

### Active Learning (Human-in-the-Loop)
Scans a folder of unlabeled audio, finds uncertain predictions (35-65% confidence), and moves those clips to a review folder for manual labeling.
```bash
python src/analysis/active_learner.py "D:\unlabeled_audio"
```

### Hard Negative Mining (Weakness Report)
Evaluates the model against the entire dataset, generates a confusion matrix, and identifies the most commonly confused class pairs.
```bash
python src/analysis/evaluate_weaknesses.py
```

### DEMUCS Source Separation
Uses Meta's DEMUCS to split audio into separate stems before classification. Requires `pip install demucs`.
```bash
python src/analysis/separate_and_analyze.py "audio.mp3"
```

### TFLite Export (Edge Deployment)
Converts models to TFLite format for Raspberry Pi, mobile, or bodycam deployment.
```bash
python src/utils/export_to_tflite.py --quantize       # Dynamic range quantization
python src/utils/export_to_tflite.py --int8           # Full INT8 (smallest)
python src/utils/export_to_tflite.py --all --quantize # Export ALL models at once
```

---

## 📁 Project Architecture

```text
training/
├── config.py                              # Central config (paths, backbone dirs)
├── src/
│   ├── preprocessing/
│   │   └── preprocess_audio.py            # Audio preprocessing + SpecAugment
│   ├── training/
│   │   ├── train_forensic_model.py        # Core training (--all_models)
│   │   ├── train_spectrogram_model.py     # Vision ViT
│   │   ├── train_wav2vec_model.py         # Wav2Vec 2.0
│   │   ├── train_student_model.py         # Knowledge Distillation
│   │   └── tune_hyperparameters.py        # KerasTuner
│   ├── testing/
│   │   └── test_model.py                  # Ensemble + Anomaly Detection
│   ├── analysis/
│   │   ├── active_learner.py              # Active Learning
│   │   ├── evaluate_weaknesses.py         # Hard Negative Mining
│   │   └── separate_and_analyze.py        # DEMUCS separation
│   └── utils/
│       ├── export_to_tflite.py            # TFLite + INT8
│       └── model_registry.py              # Model versioning
├── models/
│   ├── yamnet/                            # YAMNet-trained models
│   ├── vggish/                            # VGGish-trained models
│   ├── spectrogram/                       # ViT-trained models
│   ├── wav2vec/                           # Wav2Vec-trained models
│   ├── student/                           # Distilled student models
│   └── tuned/                             # KerasTuner best models
```

---

## 📊 Understanding Results

When you run `test_model.py`, you will see:

```text
🔎 MULTI-RESOLUTION FORENSIC TIMELINE ANALYSIS
==================================================
📊 TIMELINE RESULTS (>20% Confidence)
--------------------------------------------------
📁 MAIN CATEGORIES DETECTED:
   • VEHICLE         ⏱️ 0.0s - 160.0s
   • ANIMALS         ⏱️ 2.0s - 9.0s

🏷️ SPECIFIC EVENTS DETECTED:
   🎯 vehicle/siren             ⏱️ 0.0s - 159.0s
   🎯 animals/dog               ⏱️ 3.0s - 7.0s

🔬 ANOMALY DETECTION SCAN:
   ⚠️ WARNING: 3 ANOMALOUS SEGMENTS DETECTED (12%)
   🕐 At: 45.0s, 67.5s, 120.0s
--------------------------------------------------
```
