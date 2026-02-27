# Custom Forensic Audio Model Training

An advanced, hierarchical Multi-Label audio classifier powered by **YAMNet Transfer Learning**. This system is designed to detect forensic audio events (e.g., sirens, glass breaking, gunshots) with high accuracy and explicitly supports analyzing external hard drives containing large datasets.

## 🌟 Key Features

1. **Hierarchical Detection**: Detects both the general environment (e.g., `vehicle`) and the specific event (e.g., `vehicle/siren`) simultaneously.
2. **Multi-Label Ready**: Can correctly identify overlapping sounds (e.g., a siren *and* a gunshot occurring at the same time).
3. **External Drive Optimization**: Built to read raw data and write processed embeddings exclusively to an external drive (configured via `config.py`), preventing your C: drive from filling up.
4. **Transfer Learning**: Built on Google's YAMNet, meaning it requires drastically fewer epochs to achieve high accuracy compared to training from scratch.
5. **Multi-Resolution Scanning**: Simultaneously analyzes audio at 0.5s, 2.0s, and 5.0s windows to catch both short transients and sustained sounds.
6. **Ensemble Voting**: Supports loading ALL trained models and averaging their predictions for supreme accuracy.
7. **Anomaly Detection**: Zero-Shot detection of unknown sounds fundamentally different from the training set.

---

## ⚙️ Prerequisites

Install the required Python packages:
```bash
pip install tensorflow tensorflow-hub librosa soundfile numpy pandas scikit-learn matplotlib
```

**Optional (for advanced features):**
```bash
pip install keras-tuner        # Automated hyperparameter tuning
pip install transformers torch # Wav2Vec 2.0 backbone
pip install demucs             # Audio source separation
```

---

## 📁 Dataset Structure

Your data should be organized on your external drive (e.g., `D:\dataset`) in a **Two-Level Hierarchy**:

```text
D:\dataset\
├── main_category_1/       (e.g., vehicle)
│   ├── sub_category_A/    (e.g., siren)
│   │   ├── sound1.wav
│   │   └── sound2.mp3
│   └── sub_category_B/    (e.g., car_horn)
│       └── ...
├── main_category_2/       (e.g., effect)
│   ├── sub_category_C/    (e.g., glass_shatter)
│   │   └── ...
├── _background_noise/     (optional, custom noise files)
│   ├── wind.wav
│   └── rain.wav
```

---

## 🔗 Code Architecture (File Purposes)

### Core Pipeline

| File | Purpose |
|------|---------|
| `config.py` | Central configuration. Tells all other scripts where to find the external hard drive and `dataset/` folder. |
| `preprocess_audio.py` | Scans raw audio, normalizes volumes, applies **SpecAugment frequency masking** and **Multi-SNR Background Noise Mixing**. |
| `train_forensic_model.py` | Core AI engine. Extracts embeddings, applies **MixUp Augmentation**, trains the Dual-Head neural network. Supports `--backbone`, `--finetune`, `--all_models`. |
| `test_model.py` | Analyzes audio using **Multi-Resolution Scanning** with optional `--ensemble` and `--anomaly` flags. |

### Advanced Training Scripts

| File | Purpose |
|------|---------|
| `train_spectrogram_model.py` | **Vision Transformer**: Converts audio to Mel-Spectrograms and trains ResNet50/EfficientNet to "look" at sounds. |
| `train_wav2vec_model.py` | **Wav2Vec 2.0**: Uses Meta's self-supervised model for deeper acoustic understanding. |
| `train_student_model.py` | **Knowledge Distillation**: Trains a tiny model to mimic the combined predictions of a teacher ensemble. |
| `tune_hyperparameters.py` | **KerasTuner**: Automatically searches for the optimal model architecture and learning rate. |

### Analysis & Utility Scripts

| File | Purpose |
|------|---------|
| `active_learner.py` | **Active Learning**: Scans unlabeled audio, identifies uncertain predictions, moves them to review folder. |
| `evaluate_weaknesses.py` | **Hard Negative Mining**: Evaluates the model against the dataset and generates a confusion report. |
| `separate_and_analyze.py` | **DEMUCS**: Separates audio into stems (vocals, instruments, etc.) before classification. |
| `export_to_tflite.py` | **Quantization**: Converts models to TFLite format for edge deployment. Supports `--int8` and `--all`. |

---

## 🚀 The Training Pipeline

### Step 1: Preprocess Audio
```bash
python preprocess_audio.py
```
Applies: pitch shifting, time stretching, **SpecAugment frequency masking**, and **Multi-SNR Background Noise Mixing** (faint 15%, medium 30%, loud 50%).

### Step 2: Train the Model

**Standard Training (YAMNet backbone):**
```bash
python train_forensic_model.py
```

**Advanced Training Options:**
```bash
# Fine-tune YAMNet internal layers (more accurate)
python train_forensic_model.py --finetune

# Use VGGish backbone instead
python train_forensic_model.py --backbone vggish

# Train ALL backbones sequentially overnight
python train_forensic_model.py --all_models

# Train alternative architectures
python train_spectrogram_model.py --architecture resnet50
python train_wav2vec_model.py --epochs 50
```

### Step 3: Test New Audio
```bash
# Standard analysis
python test_model.py "audio.mp3" --threshold 0.20

# Ensemble voting (all models vote)
python test_model.py "audio.mp3" --ensemble

# Anomaly detection (flag unknown sounds)
python test_model.py "audio.mp3" --anomaly

# Full power: ensemble + anomaly
python test_model.py "audio.mp3" --ensemble --anomaly --threshold 0.15
```

### Step 4: Advanced Tools
```bash
# Auto-tune hyperparameters overnight
python tune_hyperparameters.py --max_trials 100

# Knowledge Distillation (compress ensemble into tiny model)
python train_student_model.py --temperature 3.0

# DEMUCS source separation + analysis
python separate_and_analyze.py "audio.mp3"

# Active Learning (find uncertain samples)
python active_learner.py "D:\unlabeled_audio"

# Evaluate model weaknesses
python evaluate_weaknesses.py

# Export to TFLite for mobile/edge deployment
python export_to_tflite.py --quantize --all
```

---

## 🛠 Model Architecture

- **Base Backbones**: YAMNet, VGGish, ResNet50, EfficientNet, Wav2Vec 2.0
- **Advanced Augmentation**: Multi-SNR Noise Mixing, MixUp, SpecAugment Frequency Masking
- **Loss Functions**: Binary Crossentropy + optional Supervised Contrastive Loss
- **LSTM Variant**: Bidirectional LSTM for temporal sequence learning
- **Feature Extraction**: Global Average + Max Pooling
- **Core Network**: Shared Dense + BatchNorm + Dropout → Dual-Head output
- **Ensemble Mode**: Multi-model voting system for near-zero false positives
- **Anomaly Detection**: IsolationForest-based Zero-Shot unknown sound detection
- **Edge Deployment**: TFLite export with INT8 quantization

All model `.keras` files, labels, and training histories are saved in the local `models/` directory.
