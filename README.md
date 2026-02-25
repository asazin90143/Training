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
The preprocessor scans your external drive, validates audio quality (skips corrupted/silent files), normalizes, chunks to 5 seconds, and can augment (pitch/time shift). It writes all output to the `processed/` folder on your external drive.

```bash
python preprocess_audio.py
```
*Depending on your dataset size, this can take hours. A `data_manifest.json` will be created to track everything.*

### Step 2: Train the Dual-Head Model
Train the forensic model using the generated manifest. The model automatically loads YAMNet, extracts 1024-d embeddings, and trains the Multi-Label classification heads. 

```bash
python train_forensic_model.py --epochs 40 --batch_size 32
```
*Features automatic Early Stopping (monitors validation loss with patience=8) and Learning Rate Plateau reduction for optimal convergence.*

### Step 3: Test New Audio
Test your model against any custom WAV/MP3 file. It will output all detected events above a 30% confidence threshold.

```bash
python test_model.py "D:\dataset\vehicle\siren\demo.wav"
```

**Example Output:**
```text
🔎 HIERARCHICAL FORENSIC ANALYSIS
==================================================
📊 RESULTS (Multi-Label Detection)
------------------------------
📁 MAIN CATEGORIES:
   • VEHICLE (98.5%)
   • EFFECT (35.2%)

🏷️ SPECIFIC EVENTS:
   🎯 vehicle/siren                95.1%
   🎯 effect/glass_shatter         42.8%
------------------------------
```

---

## 🛠 Model Architecture

- **Base Node**: Google YAMNet (pre-trained on AudioSet)
- **Feature Extraction**: Global Average Pooling + Global Max Pooling (catches both sustained noise and transient impulses like gunshots).
- **Core Network**: Shared Dense layers (1024 -> 512) with Batch Normalization and 40% Dropout.
- **Main Output Head**: Dense (Sigmoid Activation, Binary Crossentropy)
- **Sub Output Head**: Dense + Main Context merging (Sigmoid Activation, Binary Crossentropy)

All model `.keras` files, labels, and training histories are saved in the local `models/` directory for version control, while heavy audio data remains external.
