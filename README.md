# Custom Forensic Audio Model Training

An advanced, hierarchical Multi-Label audio classifier powered by **YAMNet Transfer Learning**. This system is designed to detect forensic audio events (e.g., sirens, glass breaking, gunshots) with high accuracy and explicitly supports analyzing external hard drives containing large datasets.

## 🌟 Key Features

1. **Hierarchical Detection**: Detects both the general environment (e.g., `vehicle`) and the specific event (e.g., `vehicle/siren`) simultaneously.
2. **Multi-Label Ready**: Can correctly identify overlapping sounds (e.g., a siren *and* a gunshot occurring at the same time).
3. **External Drive Optimization**: Built to read raw data and write processed embeddings exclusively to an external drive (configured via `config.py`), preventing your C: drive from filling up.
4. **Transfer Learning**: Built on Google's YAMNet, meaning it requires drastically fewer epochs to achieve high accuracy compared to training from scratch.

---

## ⚙️ Prerequisites

Install the required Python packages:
```bash
pip install tensorflow tensorflow-hub librosa soundfile numpy pandas scikit-learn matplotlib
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
```

*Note: Ensure your `config.py` points to this root directory via the `DATASET_ROOT` variable.*

---

## 🔗 Code Architecture (File Purposes)

Here is exactly what each script in this repository does:

| File | Purpose |
|------|---------|
| `config.py` | Central configuration. Tells all other scripts where to find the external hard drive and `dataset/` folder. |
| `download_dataset.py` | (Optional) Automatically grabs sounds from Freesound/HuggingFace if you don't have your own dataset. |
| `preprocess_audio.py` | Scans the raw audio, filters out bad files, normalizes volumes, and exports 16kHz audio to the `processed/` drive. Builds the `data_manifest.json` map. |
| `train_forensic_model.py` | The core AI engine. Reads the manifest, extracts YAMNet embeddings, and trains the Multi-Label, Dual-Head neural network. |
| `test_model.py` | Used after training. Push any raw audio file into it to see what acoustic events (Main + Sub categories) are inside it. |

---

## 🚀 The Training Pipeline

### Step 1: Preprocess Audio
The preprocessor scans your external drive, validates audio quality, normalizes, chunks to 5 seconds, and applies **Advanced Data Augmentation**. It automatically applies pitch shifting, time stretching, and **Multi-SNR Background Noise Mixing** (faint, medium, loud) to teach the model to ignore overlapping noise. It writes all output to the `processed/` folder.

```bash
python preprocess_audio.py
```
*Note: You can add an optional `D:\dataset\_background_noise\` folder with custom noise files (wind, rain, static) to be automatically mixed into your training data for extreme real-world robustness.*

### Step 2: Train the Dual-Head Model
Train the forensic model using the generated manifest. The model automatically extracts embeddings using a pre-trained backbone, applies **MixUp Augmentation** on the fly (blending sounds together to teach multi-label detection), and trains the Dual-Head classification layers.

**Standard Training (100 epochs, YAMNet backbone):**
```bash
python train_forensic_model.py
```

**Advanced Training Options:**
```bash
# Fine-tune the deep layers of YAMNet (slower but highly accurate)
python train_forensic_model.py --finetune

# Use VGGish instead of YAMNet as the audio feature extractor
python train_forensic_model.py --backbone vggish
```
*Features automatic Early Stopping (patience=15) and Learning Rate Plateau reduction for optimal convergence.*

### Step 3: Test New Audio
Test your model against any custom WAV/MP3 file. The tester uses **Multi-Resolution Scanning** (simultaneously checking 0.5s, 2.0s, and 5.0s sliding windows) to catch both short transients (gunshots) and long sustained sounds (sirens) and outputs an exact timeline.

```bash
# Analyze a file using a 20% certainty threshold
python test_model.py "audio.mp3" --threshold 0.20
```

**Example Output:**
```text
🔎 MULTI-RESOLUTION FORENSIC TIMELINE ANALYSIS
==================================================
📊 TIMELINE RESULTS (>20% Confidence)
--------------------------------------------------
📁 MAIN CATEGORIES DETECTED:
   • VEHICLE         ⏱️ 0.0s - 160.0s
   • ANIMALS         ⏱️ 2.0s - 9.0s
   • ENVIRONMENT     ⏱️ 9.0s - 17.0s, 71.0s - 95.0s

🏷️ SPECIFIC EVENTS DETECTED:
   🎯 vehicle/siren             ⏱️ 0.0s - 159.0s
   🎯 animals/dog               ⏱️ 3.0s - 7.0s
   🎯 environment/traffic       ⏱️ 9.0s - 15.0s, 72.0s - 95.0s
--------------------------------------------------
```

---

## 🛠 Model Architecture

- **Base Node**: Google YAMNet or VGGish (swappable via command line).
- **Advanced Augmentation**: Multi-SNR Environmental Mixing & On-The-Fly MixUp.
- **Feature Extraction**: Global Average Pooling + Global Max Pooling (catches both sustained noise and transient impulses like gunshots).
- **Core Network**: Shared Dense layers (1024 -> 512) with Batch Normalization and 40% Dropout.
- **Main Output Head**: Dense (Sigmoid Activation, Binary Crossentropy)
- **Sub Output Head**: Dense + Main Context merging (Sigmoid Activation, Binary Crossentropy)

All model `.keras` files, labels, and training histories are saved in the local `models/` directory for version control, while heavy audio data remains external.
