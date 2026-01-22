# 📚 Complete Setup & Testing Guide

This guide walks you through getting the necessary API keys, downloading data, and training your first model.

---

## 🔑 Part 1: Getting API Access

### 1. Freesound.org (for High-Quality SFX)
Freesound is the best source for specific sound effects like "glass shatter" or "gunshots".

1.  **Create an Account**: Go to [Freesound.org](https://freesound.org/home/register/) and sign up.
2.  **Request API Access**:
    *   Go to [Freesound API Credentials](https://freesound.org/apiv2/apply).
    *   **Project Name**: Any name (e.g., "ForensicAudio").
    *   **App URL**: You can just put `https://google.com` if you don't have one.
    *   **Description**: "Training an AI model for sound recognition."
3.  **Get Your Key**:
    *   Once created, look for the **"Client Secret"** or **"API Key"**. It's a long string of random characters.
    *   **Copy this key.** You will use it with the `--freesound-key` flag.

### 2. HuggingFace (for General Datasets)
We use the `ESC-50` dataset hosted on HuggingFace.

1.  **No Key Required**: For public datasets like ESC-50, you usually don't need an API key!
2.  **Installation**: Just ensure you have the library installed:
    ```bash
    pip install datasets
    ```

---

## 🚀 Part 2: The Full Workflow

Follow these exact steps to go from "Empty Folder" to "Working AI Model".

### Step 1: Download Data
Run the downloader. It will create folders in `dataset/` and fill them with audio files.

```bash
# Use your Freesound Key AND HuggingFace together
python download_dataset.py --freesound-key YOUR_COPIED_KEY_HERE --use-hf
```
*Wait for it to finish. You should see "✅ Download complete".*

### Step 2: Validate & Preprocess
Clean the data. This converts everything to the right format and removes silent/bad files.

```bash
python preprocess_audio.py
```
*Look for "✅ Processed" messages.*

### Step 3: Train the Model
Train the AI. This will take a few minutes depending on your computer.

```bash
python train_forensic_model.py --epochs 30
```
*Watch the "Accuracy" go up!*

### Step 4: Test the Model
Now test it on a real file!

**Option A: Test on a file from the dataset**
Pick a file usually found in `dataset/siren/` (or similar) to confirm it works.
```bash
python test_model.py "dataset/siren/some_siren_file.wav"
```

**Option B: Test Multi-Label Detection**
If you have a customized recording with overlapping sounds:
```bash
python test_model.py "path/to/my_complex_sound.wav"
```

---

## 📊 Understanding Results

When you run `test_model.py`, you will see output like this:

```text
🎯 DETECTED EVENTS (> 30%):
   • SIREN                98.5%
   • DOG_BARK             12.1%  <-- (Low confidence, ignored if threshold is higher)
```

- **Top Section**: Shows definitive detections.
- **Full Analysis**: Shows raw probabilities for all classes.
