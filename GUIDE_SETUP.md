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
Clean the data. This scans `D:\dataset`, finds all the "Main Category / Subclass" folders, validates the audio, and converts everything to the right format in `D:\processed`.

```bash
python preprocess_audio.py
```
*Wait for it to finish. You should see "✅ Processed" messages and the number of discovered files.*

### Step 2: Train the Model (Multi-Label & Hierarchical)
Train the AI. Our model uses **Transfer Learning (YAMNet)** and features a **Dual-Head** architecture. 
It detects both the **Main Category** (e.g., Vehicle) and the **Specific Event** (e.g., Siren) simultaneously. Because it's **Multi-Label**, it can detect *multiple* overlapping sounds at once.

```bash
python train_forensic_model.py --epochs 40
```
*Note: The system has **Early Stopping**, so if it finishes learning early (e.g. at epoch 15), it will stop automatically and save the best version!*

### Step 3: Test the Model
Now test it on a real file! Pick any `.wav` file to analyze.

```bash
python test_model.py "D:\dataset\vehicle\siren\example.wav"
```

---

## 📊 Understanding Results

When you run `test_model.py`, you will see output like this:

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

- **Main Categories**: Shows the broad classification of overlapping environments.
- **Specific Events**: Shows exactly what specific triggers were found above the 30% confidence threshold.
