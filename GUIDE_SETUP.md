# 📚 Complete Setup & Testing Guide

This guide walks you through preprocessing your external dataset, training the multi-label hierarchical model, and testing it.

---

## � The Full Workflow (External Dataset)

Your data is currently configured to run from an **External Drive** to save space. 

**Current Configuration (`config.py`):**
- **Raw Data:** `D:\dataset`
- **Processed Data:** `D:\processed`
- **Models & Manifest:** Saved locally in the `training` folder.

Follow these exact steps to go from raw folders to a working AI model.

### Step 1: Validate & Preprocess (Scan the Drive)
Clean the data. This scans `D:\dataset`, finds all the "Main Category / Subclass" folders, validates the audio, and converts everything to the right format in `D:\processed`. It also automatically applies multiple data augmentations (pitch shifting, noise mixing).

*Pro Tip: Create a folder called `D:\dataset\_background_noise\` filled with random noise files (wind, crowds, traffic). The preprocessor will automatically blend these into your target sounds at varying volumes to teach the AI to ignore them!*

```bash
python preprocess_audio.py
```
*Wait for it to finish. You should see "✅ Processed" messages and the number of discovered files.*

### Step 2: Train the Model (Multi-Label & Hierarchical)
Train the AI. Our model uses **Transfer Learning (YAMNet or VGGish)** and features a **Dual-Head** architecture. 
It detects both the **Main Category** (e.g., Vehicle) and the **Specific Event** (e.g., Siren) simultaneously. Because it's **Multi-Label**, it uses **MixUp Augmentation** internally to detect *multiple* overlapping sounds at once.

```bash
# Standard training (Fastest)
python train_forensic_model.py

# Advanced training: unfreezing YAMNet's deep layers (Highly recommended for accuracy)
python train_forensic_model.py --finetune
```
*Note: The system has **Early Stopping**, so if it finishes learning before 100 epochs, it will stop automatically and save the best version!*

### Step 3: Test the Model
Now test it on a real file! Pick any `.wav` or `.mp3` file to analyze. You can use the `--threshold` flag to control sensitivity (`0.15` catches faint sounds, `0.50` is very strict).

```bash
python test_model.py "audio.mp3" --threshold 0.20
```

---

## 📊 Understanding Results

When you run `test_model.py`, you will see output like this:

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

- **Main Categories**: Shows the broad classification of overlapping environments along a timeline.
- **Specific Events**: Shows exactly what specific triggers were found above your set confidence threshold, mapped directly to their start and end times in the audio.
