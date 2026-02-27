"""
Vision Transformer (ViT) Spectrogram-Based Training Script
Converts audio to Mel-Spectrograms and trains a Vision model (ResNet50/EfficientNet)
to classify sounds by looking at their visual frequency patterns.

Usage:
    python train_spectrogram_model.py --epochs 50
    python train_spectrogram_model.py --epochs 50 --architecture efficientnet
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

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not installed.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]

# Spectrogram Config
N_MELS = 128
HOP_LENGTH = 512
N_FFT = 2048
IMG_HEIGHT = 128
IMG_WIDTH = 128


def audio_to_spectrogram(audio_path, sr=16000):
    """Convert audio file to a Mel-Spectrogram image array."""
    try:
        wav, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        # Generate Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=N_MELS,
                                           hop_length=HOP_LENGTH, n_fft=N_FFT)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize to 0-1 range
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-8)
        
        # Resize to fixed dimensions
        S_resized = tf.image.resize(
            tf.expand_dims(S_norm, axis=-1),  # Add channel dim
            [IMG_HEIGHT, IMG_WIDTH]
        ).numpy()
        
        # Convert to 3-channel (RGB) for pretrained vision models
        S_rgb = np.repeat(S_resized, 3, axis=-1)
        return S_rgb
    except Exception as e:
        return None


def prepare_spectrogram_data(manifest_path, test_split=0.2):
    """Load manifest and convert all audio to spectrograms."""
    with open(manifest_path) as f:
        data = json.load(f)
    
    main_classes = data.get("main_classes", [])
    sub_classes = data.get("sub_classes", [])
    samples = data["samples"]
    
    print(f"📊 Dataset: {len(samples)} samples, {len(main_classes)} main, {len(sub_classes)} sub classes")
    print("🖼️ Converting audio to spectrograms...")
    
    X = []
    y_main = []
    y_sub = []
    skipped = 0
    
    for i, sample in enumerate(samples):
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(samples)}...")
        
        path = Path(sample["file"])
        if not path.exists():
            try:
                parts = path.parts
                if "processed" in parts:
                    idx = parts.index("processed")
                    rel_parts = parts[idx+1:]
                    new_path = PATHS["processed"].joinpath(*rel_parts)
                    if new_path.exists():
                        path = new_path
            except Exception:
                pass
        
        if not path.exists():
            skipped += 1
            continue
        
        spec = audio_to_spectrogram(str(path))
        if spec is None:
            skipped += 1
            continue
        
        X.append(spec)
        m_id = sample.get("main_class_id", main_classes.index(sample["main_class"]))
        s_id = sample.get("sub_class_id", 0)
        y_main.append(m_id)
        y_sub.append(s_id)
    
    if skipped > 0:
        print(f"   ⚠️ Skipped {skipped} files")
    
    X = np.array(X)
    y_main = tf.keras.utils.to_categorical(y_main, num_classes=len(main_classes))
    y_sub = tf.keras.utils.to_categorical(y_sub, num_classes=len(sub_classes))
    
    # Shuffle and split
    idx = np.random.permutation(len(X))
    X = X[idx]
    y_main = y_main[idx]
    y_sub = y_sub[idx]
    
    split = int(len(X) * (1 - test_split))
    return (X[:split], {"main_output": y_main[:split], "sub_output": y_sub[:split]}), \
           (X[split:], {"main_output": y_main[split:], "sub_output": y_sub[split:]}), \
           main_classes, sub_classes


def create_vit_model(num_main, num_sub, architecture="resnet50"):
    """Build a Vision-based dual-head model using pretrained image backbones."""
    input_layer = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    
    if architecture == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_tensor=input_layer
        )
    else:  # resnet50
        base = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_tensor=input_layer
        )
    
    # Freeze early layers, fine-tune later layers
    for layer in base.layers[:-20]:
        layer.trainable = False
    
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(x)
    
    sub_branch = tf.keras.layers.concatenate([x, main_output])
    sub_branch = tf.keras.layers.Dense(256, activation='relu')(sub_branch)
    sub_output = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[main_output, sub_output])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Vision Transformer Spectrogram Model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--architecture", type=str, default="resnet50",
                        choices=["resnet50", "efficientnet"])
    args = parser.parse_args()
    
    if not TF_AVAILABLE or not LIBROSA_AVAILABLE:
        print("❌ Missing dependencies.")
        return
    
    print("="*50)
    print("🖼️ VISION TRANSFORMER SPECTROGRAM TRAINING")
    print("="*50)
    
    (X_train, y_train), (X_test, y_test), main_cls, sub_cls = \
        prepare_spectrogram_data(MANIFEST_PATH)
    
    print(f"\n📊 Train: {len(X_train)}, Test: {len(X_test)}")
    
    model = create_vit_model(len(main_cls), len(sub_cls), args.architecture)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss={"main_output": "binary_crossentropy", "sub_output": "binary_crossentropy"},
        loss_weights={"main_output": 0.5, "sub_output": 1.0},
        metrics={"main_output": "binary_accuracy", "sub_output": "binary_accuracy"}
    )
    
    model.summary()
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-7)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"vit_{args.architecture}_{timestamp}"
    MODELS_DIR.mkdir(exist_ok=True)
    
    model.save(str(MODELS_DIR / f"{name}.keras"))
    with open(MODELS_DIR / f"{name}_labels.json", "w") as f:
        json.dump({"main_classes": main_cls, "sub_classes": sub_cls}, f, indent=2)
    
    print(f"\n✅ ViT Model saved: {name}.keras")


if __name__ == "__main__":
    main()
