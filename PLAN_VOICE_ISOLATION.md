# Advanced Voice Diarization & Separation Architecture

This document serves as the mathematical and architectural blueprint for custom-training PyAnnote and SepFormer components to perfectly isolate up to 3+ overlapping speakers in extremely noisy, unpredictable forensic environments.

## 🧭 The Core Problem
Pre-trained off-the-shelf isolation models (like `speechbrain/sepformer-wsj03mix`) strictly assume a relatively predictable "clean" room acoustic. When introduced to explosive volume spikes, microphone wind, or radio static, their internal embedding clusters shatter, causing multiple speakers to bleed onto a single output track. 

To achieve flawless extraction, we must force the neural network to memorize the exact profile of our forensic microphones.

---

## 🏛️ Pillar 1: PyAnnote Segmentation & Clustering

Generating 100% accurate timestamp boundaries is the prerequisite to separation. We will heavily fine-tune the PyAnnote `segmentation-3.0` backbone.

### 1-A. Active Learning (Ground Truth Prep)
Labeling human speech frame-by-frame across hundreds of hours of forensic audio is impossible without an army of annotators. 
1. Run your *current*, slightly flawed PyAnnote inference script over your massive unlabeled audio dataset.
2. Output the predictions as `.rttm` (Rich Transcription Time Marked) files.
3. Import the `.wav` files and their accompanying `.rttm` files into an open-source visual interface (e.g., *LabelStudio*).
4. **Action**: Visually slide the boundary boxes left or right to correct the AI's mistakes. This guarantees a hyper-accurate ground truth in a fraction of the time.

### 1-B. Forensic Acoustic Augmentations
PyAnnote often mistakenly flags a siren or a dog bark as a "human voice." To stop this hallucination:
1. Hook into the `preprocess_audio.py` pipeline. 
2. During the PyTorch training loop, randomly inject extreme background noise arrays (sirens, radio static, engine rumbles) at varying SNRs (Signal-to-Noise Ratios) directly into the clean vocal tracks.
3. The loss function will mathematically penalize PyAnnote if it tries to segment the noise, strictly forcing it to only output timestamps when biological human vocal cords are vibrating.

### 1-C. Mixup Augmentation (Data Balancing)
To ensure the model treats all classes equally and generalizes to rare forensic sounds:
1. **Linear Interpolation**: During training, the pipeline will randomly select two audio samples and their labels, blending them mathematically (e.g., 70% of a siren mixed with 30% of a rare glass-break).
2. **Boundary Smoothing**: This forces the neural network to learn "soft" decison boundaries, preventing it from over-focusing on the most common sounds in the dataset.
3. **Class Equality**: By over-sampling rare forensic events and mixing them with common background noise, we ensure the model sees every category with equal frequency.

---

## 🏛️ Pillar 2: The "Targeted" SepFormer Architecture

Once PyAnnote provides the exact milliseconds someone is speaking, SepFormer mathematically rips them out of the background track.

### 2-A. Transfer Learning (Weight Initialization)
Training a massive SepFormer from scratch takes hundreds of GPU days. Instead:
1. Initialize the neural network using the pre-trained `speechbrain/sepformer-wsj03mix` tensors.
2. The model instantly boots up possessing the fundamental physics of "human speech vs silence." It only requires ~20 sparse epochs of your custom forensic data to fine-tune its clustering logic to your unpredictable room echo.

### 2-B. Targeted Speaker Extraction (TSE)
Rather than blindly "separating all distinct sound channels," we modify the inputs.
1. The AI ingests the messy `mix.wav` **AND** a clean 3-second reference vector (`s1.wav`) of the specific target suspect.
2. The AI uses the embedding of the suspect's voice as a key to "unlock" only those specific frequencies in the main track. The model acts like a missile, destroying any sound wave that doesn't mathematically align with the reference vector.

### 2-C. Adversarial Noise Immunity (DANN / Gradient Reversal)
To permanently eliminate background static from the isolated vocal stems, we execute a dual-network conflict:
1. We attach a secondary **Domain Classifier AI** to the extraction layer of the SepFormer.
2. The Domain Classifier spends all its computing power attempting to guess what the background noise of the audio is (e.g., "Wind" or "Siren").
3. We connect the two networks via a **Gradient Reversal Layer (GRL)**. During backpropagation, whatever the Domain Classifier learns is multiplied by `-1` and sent into the SepFormer.
4. This mathematically tortures the SepFormer, forcing it to violently erase any acoustic signature of wind or sirens from its final output just to stop the Domain Classifier from guessing correctly.

### 2-D. Self-Supervised Pre-Training (Microsoft BEATs Integration)
Instead of arbitrary linear layers at the front of the SepFormer, we inject the massive **Microsoft BEATs** (`BEATs_iter3_plus_AS2M.pt`) transformer you already downloaded. 
Because BEATs is pre-trained using Contrastive Learning on millions of hours of raw audio, the SepFormer inherits extreme deep-feature geometry instantly, completely bypassing the need for millions of hours of labeled data.

### 2-E. Knowledge Distillation (The Student Separator)
Once the massive 200M+ parameter SepFormer is fully trained and extracting flawless audio:
1. Freeze the giant neural network.
2. Initialize a tiny, 10-layer "Student SepFormer."
3. Force the tiny Student network to blindly copy the exact matrix outputs of the massive Teacher network.
4. The result is a Student model that processes audio exactly identical to the Teacher, but requires 80% less VRAM and runs 5x faster in the live Next.js API.

---

## 🚀 Execution Phases

### Phase 0 — Dataset Download Script
Create `download_separation_datasets.py` at the project root.

**Fully Automated (7 datasets):**
| Dataset | Size | Source |
|---------|------|--------|
| LibriMix | ~430GB (2-spk) + ~332GB (3-spk) | [GitHub: JorisCos/LibriMix](https://github.com/JorisCos/LibriMix) |
| LibriSpeech | ~60GB | [OpenSLR](https://www.openslr.org/12) |
| WHAM! | 17GB compressed | [wham.whisper.ai](https://wham.whisper.ai) |
| WHAMR! | ~35GB | [wham.whisper.ai](https://wham.whisper.ai) |
| MUSAN | ~11GB | [OpenSLR](https://www.openslr.org/17/) |
| DNS Challenge | ~500GB+ | [GitHub: microsoft/DNS-Challenge](https://github.com/microsoft/DNS-Challenge) |
| AMI Meeting Corpus | ~100GB | [AMI Corpus](https://groups.inf.ed.ac.uk/ami/corpus/) |

**Manual Agreement Required (3 datasets):**
| Dataset | Reason | Source |
|---------|--------|--------|
| VoxCeleb 1 | Must agree to terms on website, then paste URL | [VGG VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) |
| VoxCeleb 2 | Must agree to terms on website, then paste URL | [VGG VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) |
| CHiME-5/6 | CHiME-5 is direct, CHiME-6 needs registration | [chimechallenge.org](https://chimechallenge.github.io/chime6/) |

All datasets saved to a **completely separate directory** (e.g., `D:\separation_dataset\`), fully isolated from the existing classification dataset on `D:\dataset\`.

---

### Phase 1 — Inference Pipeline (Immediate)
1. Create `src/analysis/speaker_diarization.py` using pre-trained Pyannote + SepFormer weights.
2. Integrate into `run_advanced_tools.py` with `--diarize` flag.
3. Authenticate via HuggingFace Token: Set `HF_TOKEN` environment variable (e.g., `set HF_TOKEN=hf_your_token_here`)
4. This pipeline must be working first — it is required by Phase 2's Active Learning step (Pillar 1-A).

---

### Phase 2 — Separate Preprocessing & Custom DANN Training
All preprocessing and training for voice separation is **completely isolated** from your existing classification pipeline. Your current YAMNet/VGGish/BEATs/Student models will remain **untouched**.

**Step 1: Preprocess**
1. **Create `src/preprocessing/preprocess_separation_data.py`**
   - Standalone script that **only** touches the new separation datasets downloaded in Phase 0.
   - Saves processed data to a separate folder (e.g., `D:\separation_processed\`).
   - Will **never** interfere with or re-run preprocessing on your existing classification data in `D:\dataset\`.

**Step 2: Active Learning — Generate Ground Truth (Pillar 1-A)**
1. Run the Phase 1 inference pipeline over your unlabeled forensic audio.
2. Export `.rttm` prediction files.
3. Import into LabelStudio and manually correct the AI's boundary mistakes.
4. Output: Hyper-accurate ground truth annotations for fine-tuning.

**Step 3: Fine-Tune & Train (Pillar 1-B, 1-C, 2-A, 2-B, 2-C)**
1. **Create `src/training/train_separation_model.py`**
   - Standalone training script that **only** reads from `D:\separation_processed\`.
   - Fine-tunes PyAnnote with Forensic Acoustic Augmentations (Pillar 1-B).
   - Implements **Mixup Augmentation** for class balancing (Pillar 1-C).
   - Fine-tunes SepFormer with Transfer Learning (Pillar 2-A).
   - Adds Targeted Speaker Extraction / TSE (Pillar 2-B).
   - Attaches DANN + GRL to the Separator (Pillar 2-C).

---

### Phase 3 — BEATs Integration & Knowledge Distillation
1. Inject BEATs transformer as the SepFormer's front-end feature extractor (Pillar 2-D).
2. Train the Student Separator via Knowledge Distillation (Pillar 2-E).

---

### Phase 4 — Documentation & Final Integration
1. **Update `README.md`** with instructions on how to use the new Speaker Diarization and Separation pipeline.
2. **Update `GUIDE_SETUP.md`** with instructions for downloading the required separation datasets and setting up the standalone preprocessing/training environment.
3. Final verification of unified runner scripts (`run_advanced_tools.py` / `run_testing_suite.py`) with the new diarization flags.

