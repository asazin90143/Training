"""
BEATs Feature Extractor for Forensic Audio
Uses Microsoft's official BEATs (Audio Pre-Training with Acoustic Tokenizers) model.
Paper: https://arxiv.org/abs/2212.09058
Source: https://github.com/microsoft/unilm/tree/master/beats

Requires: pip install torch torchaudio librosa

The checkpoint file (BEATs_iter3_plus_AS2M.pt) must be placed in:
    models/beats/BEATs_iter3_plus_AS2M.pt

Download from Microsoft OneDrive:
    https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf

Usage:
    python train_beats_model.py --epochs 50
    python train_beats_model.py --epochs 100 --checkpoint path/to/BEATs.pt
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# --- PyTorch + torchaudio (required by BEATs) ---
BEATS_AVAILABLE = False
try:
    import torch
    import torchaudio
    BEATS_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch or torchaudio not installed.")
    print("   Run: pip install torch torchaudio")

# --- Project paths ---
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import get_paths
PATHS = get_paths("beats")

# --- Add the cloned BEATs source to Python path ---
# Microsoft's BEATs code uses relative imports (from backbone import ..., from modules import ...)
# so we need to add the beats directory itself to sys.path
BEATS_SOURCE_DIR = Path(__file__).parent / "unilm" / "beats"
if not BEATS_SOURCE_DIR.exists():
    # Also check in utils (user may have cloned there)
    BEATS_SOURCE_DIR_ALT = Path(__file__).parent.parent / "utils" / "unilm" / "beats"
    if BEATS_SOURCE_DIR_ALT.exists():
        BEATS_SOURCE_DIR = BEATS_SOURCE_DIR_ALT

if str(BEATS_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(BEATS_SOURCE_DIR))

MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]

# Default checkpoint name
DEFAULT_CHECKPOINT_NAME = "BEATs_iter3_plus_AS2M.pt"
BEATS_EMBEDDING_SIZE = 768  # BEATs encoder_embed_dim


def load_beats_model(checkpoint_path):
    """Load the official BEATs model from a .pt checkpoint file."""
    from BEATs import BEATs, BEATsConfig

    print(f"📦 Loading BEATs checkpoint: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')

    cfg = BEATsConfig(checkpoint['cfg'])
    beats_model = BEATs(cfg)
    beats_model.load_state_dict(checkpoint['model'])
    beats_model.eval()

    print(f"✅ BEATs loaded ({cfg.encoder_layers} layers, embed_dim={cfg.encoder_embed_dim})")
    return beats_model, cfg


def extract_beats_embeddings(audio_path, beats_model):
    """Extract BEATs embeddings from an audio file."""
    try:
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)

        # BEATs expects (batch, samples) tensor
        audio_tensor = torch.tensor(wav).unsqueeze(0).float()
        padding_mask = torch.zeros(1, audio_tensor.shape[1]).bool()

        with torch.no_grad():
            representation, _ = beats_model.extract_features(audio_tensor, padding_mask=padding_mask)

        # representation shape: (1, seq_len, 768)
        hidden = representation.squeeze(0).numpy()

        # Pool: mean + max
        mean_emb = np.mean(hidden, axis=0)
        max_emb = np.max(hidden, axis=0)
        final_emb = np.concatenate([mean_emb, max_emb])

        return final_emb
    except Exception as e:
        return None


def prepare_data(manifest_path, beats_model, test_split=0.2):
    """Extract BEATs embeddings from all audio samples."""
    with open(manifest_path) as f:
        data = json.load(f)

    main_classes = data.get("main_classes", [])
    sub_classes = data.get("sub_classes", [])
    samples = data["samples"]

    print(f"📊 {len(samples)} samples, {len(main_classes)} main, {len(sub_classes)} sub")
    print("🔄 Extracting BEATs embeddings (this may take a while)...")

    X, y_main, y_sub = [], [], []
    skipped = 0

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(samples)}...")

        path = Path(sample["file"])
        if not path.exists():
            try:
                parts = path.parts
                if "processed" in parts:
                    idx = parts.index("processed")
                    new_path = PATHS["processed"].joinpath(*parts[idx+1:])
                    if new_path.exists():
                        path = new_path
            except Exception:
                pass

        if not path.exists():
            skipped += 1
            continue

        emb = extract_beats_embeddings(str(path), beats_model)
        if emb is None:
            skipped += 1
            continue

        X.append(emb)
        y_main.append(sample.get("main_class_id", main_classes.index(sample["main_class"])))
        y_sub.append(sample.get("sub_class_id", 0))

    if skipped > 0:
        print(f"   ⚠️ Skipped {skipped}")

    if len(X) == 0:
        return (np.array([]), {}), (np.array([]), {}), main_classes, sub_classes

    X = np.array(X)
    y_main = tf.keras.utils.to_categorical(y_main, len(main_classes))
    y_sub = tf.keras.utils.to_categorical(y_sub, len(sub_classes))

    idx = np.random.permutation(len(X))
    X, y_main, y_sub = X[idx], y_main[idx], y_sub[idx]

    split = int(len(X) * (1 - test_split))
    return (X[:split], {"main_output": y_main[:split], "sub_output": y_sub[:split]}), \
           (X[split:], {"main_output": y_main[split:], "sub_output": y_sub[split:]}), \
           main_classes, sub_classes


def create_beats_head(num_main, num_sub):
    """Build classification head for BEATs embeddings."""
    emb_input = BEATS_EMBEDDING_SIZE * 2  # mean + max pooling = 1536
    input_layer = tf.keras.layers.Input(shape=(emb_input,))

    x = tf.keras.layers.Dense(512, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    main_out = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(x)

    sub_branch = tf.keras.layers.concatenate([x, main_out])
    sub_branch = tf.keras.layers.Dense(256, activation='relu')(sub_branch)
    sub_out = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)

    return tf.keras.Model(inputs=input_layer, outputs=[main_out, sub_out])


def find_checkpoint(user_path=None):
    """Find the BEATs checkpoint file. Search in multiple locations."""
    candidates = []

    if user_path:
        candidates.append(Path(user_path))

    # Check models/beats/ directory
    candidates.append(MODELS_DIR / DEFAULT_CHECKPOINT_NAME)
    # Check project root
    candidates.append(Path(PROJECT_ROOT) / DEFAULT_CHECKPOINT_NAME)
    # Check current directory
    candidates.append(Path.cwd() / DEFAULT_CHECKPOINT_NAME)

    for p in candidates:
        if p.exists():
            return p

    return None


def main():
    parser = argparse.ArgumentParser(description="Train with Microsoft BEATs Backbone")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to BEATs .pt checkpoint file")
    args = parser.parse_args()

    if not BEATS_AVAILABLE:
        print("❌ Install: pip install torch torchaudio")
        return

    if not TF_AVAILABLE:
        print("❌ Install: pip install tensorflow")
        return

    print("=" * 60)
    print("🧠 MICROSOFT BEATs FORENSIC MODEL TRAINING")
    print("   Audio Pre-Training with Acoustic Tokenizers")
    print("=" * 60)

    # --- Verify BEATs source code is available ---
    if not BEATS_SOURCE_DIR.exists():
        print(f"\n❌ BEATs source code not found at: {BEATS_SOURCE_DIR}")
        print("   Clone Microsoft's unilm repository:")
        print("   git clone https://github.com/microsoft/unilm.git src/training/unilm")
        return

    # --- Find checkpoint ---
    checkpoint_path = find_checkpoint(args.checkpoint)
    if checkpoint_path is None:
        print(f"\n❌ BEATs checkpoint not found!")
        print(f"   Expected: {MODELS_DIR / DEFAULT_CHECKPOINT_NAME}")
        print(f"\n   Download the checkpoint from Microsoft OneDrive:")
        print(f"   https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf")
        print(f"\n   Then place it in: {MODELS_DIR}/")
        print(f"   Or specify the path: --checkpoint /path/to/{DEFAULT_CHECKPOINT_NAME}")
        return

    # --- Load BEATs ---
    beats_model, beats_cfg = load_beats_model(checkpoint_path)

    # --- Extract embeddings and prepare data ---
    (X_train, y_train), (X_test, y_test), main_cls, sub_cls = \
        prepare_data(MANIFEST_PATH, beats_model)

    if len(X_train) == 0:
        print("❌ No data processed successfully.")
        return

    print(f"\n📐 Training data shape: {X_train.shape}")
    print(f"📐 Test data shape: {X_test.shape}")

    # --- Build and train classification head ---
    model = create_beats_head(len(main_cls), len(sub_cls))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"main_output": "binary_crossentropy", "sub_output": "binary_crossentropy"},
        loss_weights={"main_output": 0.5, "sub_output": 1.0},
        metrics={"main_output": "binary_accuracy", "sub_output": "binary_accuracy"}
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-7)
    ]

    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)

    # --- Save model ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"beats_model_{timestamp}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODELS_DIR / f"{name}.keras"))
    with open(MODELS_DIR / f"{name}_labels.json", "w") as f:
        json.dump({"main_classes": main_cls, "sub_classes": sub_cls}, f, indent=2)
    print(f"\n✅ BEATs model saved: {name}.keras")


if __name__ == "__main__":
    main()
