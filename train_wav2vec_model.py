"""
Wav2Vec 2.0 Feature Extractor for Forensic Audio
Uses Meta's self-supervised Wav2Vec 2.0 model as an alternative backbone.
Requires: pip install transformers torch

Usage:
    python train_wav2vec_model.py --epochs 50
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

# Wav2Vec 2.0 (HuggingFace)
WAV2VEC_AVAILABLE = False
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import torch
    WAV2VEC_AVAILABLE = True
except ImportError:
    print("⚠️ HuggingFace transformers or PyTorch not installed.")
    print("   Run: pip install transformers torch")

from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]

WAV2VEC_MODEL = "facebook/wav2vec2-base"
WAV2VEC_EMBEDDING_SIZE = 768  # Wav2Vec 2.0 base output


def extract_wav2vec_embeddings(audio_path, processor, model):
    """Extract Wav2Vec 2.0 embeddings from audio file."""
    try:
        wav, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        
        # Process through Wav2Vec 2.0
        inputs = processor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get hidden states (batch, seq_len, 768)
        hidden = outputs.last_hidden_state.squeeze(0).numpy()
        
        # Pool: mean + max
        mean_emb = np.mean(hidden, axis=0)
        max_emb = np.max(hidden, axis=0)
        final_emb = np.concatenate([mean_emb, max_emb])
        
        return final_emb
    except Exception as e:
        return None


def prepare_data(manifest_path, processor, wav2vec_model, test_split=0.2):
    """Extract Wav2Vec 2.0 embeddings from all audio samples."""
    with open(manifest_path) as f:
        data = json.load(f)
    
    main_classes = data.get("main_classes", [])
    sub_classes = data.get("sub_classes", [])
    samples = data["samples"]
    
    print(f"📊 {len(samples)} samples, {len(main_classes)} main, {len(sub_classes)} sub")
    print("🔄 Extracting Wav2Vec 2.0 embeddings...")
    
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
        
        emb = extract_wav2vec_embeddings(str(path), processor, wav2vec_model)
        if emb is None:
            skipped += 1
            continue
        
        X.append(emb)
        y_main.append(sample.get("main_class_id", main_classes.index(sample["main_class"])))
        y_sub.append(sample.get("sub_class_id", 0))
    
    if skipped > 0:
        print(f"   ⚠️ Skipped {skipped}")
    
    X = np.array(X)
    y_main = tf.keras.utils.to_categorical(y_main, len(main_classes))
    y_sub = tf.keras.utils.to_categorical(y_sub, len(sub_classes))
    
    idx = np.random.permutation(len(X))
    X, y_main, y_sub = X[idx], y_main[idx], y_sub[idx]
    
    split = int(len(X) * (1 - test_split))
    return (X[:split], {"main_output": y_main[:split], "sub_output": y_sub[:split]}), \
           (X[split:], {"main_output": y_main[split:], "sub_output": y_sub[split:]}), \
           main_classes, sub_classes


def create_wav2vec_head(num_main, num_sub):
    """Build classification head for Wav2Vec 2.0 embeddings."""
    emb_input = WAV2VEC_EMBEDDING_SIZE * 2  # mean + max pooling
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


def main():
    parser = argparse.ArgumentParser(description="Train with Wav2Vec 2.0 Backbone")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    if not WAV2VEC_AVAILABLE:
        print("❌ Install: pip install transformers torch")
        return
    
    print("="*50)
    print("🧠 WAV2VEC 2.0 FORENSIC MODEL TRAINING")
    print("="*50)
    
    print("\n📦 Loading Wav2Vec 2.0...")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL)
    wav2vec = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL)
    wav2vec.eval()
    print("✅ Wav2Vec 2.0 loaded")
    
    (X_train, y_train), (X_test, y_test), main_cls, sub_cls = \
        prepare_data(MANIFEST_PATH, processor, wav2vec)
    
    model = create_wav2vec_head(len(main_cls), len(sub_cls))
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"wav2vec_model_{timestamp}"
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(str(MODELS_DIR / f"{name}.keras"))
    with open(MODELS_DIR / f"{name}_labels.json", "w") as f:
        json.dump({"main_classes": main_cls, "sub_classes": sub_cls}, f, indent=2)
    print(f"\n✅ Wav2Vec model saved: {name}.keras")


if __name__ == "__main__":
    main()
