"""
Custom Forensic Audio Model Training Script (Hierarchical)
Trains a model with two classification heads:
1. Main Class (e.g., Vehicle)
2. Sub Class (e.g., Siren)
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# TensorFlow
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not installed.")

# Librosa
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Configuration
from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]

# YAMNet
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Training Config
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
EMBEDDING_SIZE = 1024


class HierarchicalDataset:
    def __init__(self, manifest_path: Path, yamnet_model):
        self.manifest_path = manifest_path
        self.yamnet_model = yamnet_model
        
        self.main_classes = []
        self.sub_classes = []
        self.samples = []
        
        self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            raise FileNotFoundError("Manifest not found. Run preprocess_audio.py first.")
        
        with open(self.manifest_path) as f:
            data = json.load(f)
        
        self.main_classes = data.get("main_classes", [])
        self.sub_classes = data.get("sub_classes", [])
        self.samples = data["samples"]
        
        print(f"📊 Dataset Loaded:")
        print(f"   - {len(self.samples)} samples")
        print(f"   - {len(self.main_classes)} main classes")
        print(f"   - {len(self.sub_classes)} sub classes")

    def extract_embeddings(self, audio_path):
        """Extract YAMNet embeddings from a processed audio file."""
        try:
            wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            _, embeddings, _ = self.yamnet_model(wav_data)
            
            mean_emb = tf.reduce_mean(embeddings, axis=0)
            max_emb = tf.reduce_max(embeddings, axis=0)
            final_emb = tf.concat([mean_emb, max_emb], axis=0)
            return final_emb.numpy()
        except Exception as e:
            print(f"  ⚠️ Error reading {audio_path}: {e}")
            return None

    def prepare_data(self, test_split=0.2):
        print("\n🔄 Extracting embeddings from processed audio...")
        X = []
        y_main = []
        y_sub = []
        skipped = 0
        
        for i, sample in enumerate(self.samples):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i+1}/{len(self.samples)}")
            # Manifest stores absolute paths
            path = Path(sample["file"])
            
            if not path.exists():
                # The external drive letter might have changed (e.g., E:\ to F:\)
                # Let's try to find it dynamically using config PATHS["processed"]
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
            
            emb = self.extract_embeddings(str(path))
            if emb is None or np.sum(np.abs(emb)) == 0:
                skipped += 1
                continue
            
            X.append(emb)
            
            # Get class IDs
            m_id = sample.get("main_class_id", self.main_classes.index(sample["main_class"]))
            s_id = sample.get("sub_class_id")
            if s_id is None:
                try:
                    s_id = self.sub_classes.index(sample.get("sub_class_full", sample["sub_class"]))
                except ValueError:
                    s_id = 0
            
            y_main.append(m_id)
            y_sub.append(s_id)
        
        if skipped > 0:
            print(f"   ⚠️ Skipped {skipped} files (missing or unreadable)")
        
        X = np.array(X)
        y_main = np.array(y_main)
        y_sub = np.array(y_sub)
        
        print(f"   ✅ Total valid samples: {len(X)}")
        
        # Shuffle
        idx = np.random.permutation(len(X))
        X = X[idx]
        y_main = y_main[idx]
        y_sub = y_sub[idx]
        
        # One-Hot
        y_main_oh = tf.keras.utils.to_categorical(y_main, num_classes=len(self.main_classes))
        y_sub_oh = tf.keras.utils.to_categorical(y_sub, num_classes=len(self.sub_classes))
        
        # Split
        split_point = int(len(X) * (1 - test_split))
        
        X_train, X_test = X[:split_point], X[split_point:]
        
        y_train = {
            "main_output": y_main_oh[:split_point],
            "sub_output": y_sub_oh[:split_point]
        }
        y_test = {
            "main_output": y_main_oh[split_point:],
            "sub_output": y_sub_oh[split_point:]
        }
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        return (X_train, y_train), (X_test, y_test)


def create_hierarchical_model(num_main, num_sub):
    """Build dual-head classification model."""
    input_layer = tf.keras.layers.Input(shape=(EMBEDDING_SIZE * 2,))
    
    # Shared Layers
    x = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Head 1: Main Class
    main_branch = tf.keras.layers.Dense(256, activation='relu')(x)
    # Sigmoid for multi-label (independent probabilities)
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(main_branch)
    
    # Head 2: Sub Class (receives shared + main context)
    sub_branch = tf.keras.layers.concatenate([x, main_branch])
    sub_branch = tf.keras.layers.Dense(512, activation='relu')(sub_branch)
    sub_branch = tf.keras.layers.Dropout(0.2)(sub_branch)
    # Sigmoid for multi-label
    sub_output = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[main_output, sub_output])
    return model


def train_model(args):
    if not TF_AVAILABLE or not LIBROSA_AVAILABLE:
        print("❌ Missing dependencies: pip install tensorflow tensorflow-hub librosa")
        return
    
    print("="*50)
    print("🧠 HIERARCHICAL MODEL TRAINING")
    print("="*50)
    
    # 1. Load YAMNet
    print("\n📦 Loading YAMNet...")
    yamnet = hub.load(YAMNET_MODEL_URL)
    print("✅ YAMNet loaded")
    
    # 2. Load Data
    dataset = HierarchicalDataset(MANIFEST_PATH, yamnet)
    (X_train, y_train), (X_test, y_test) = dataset.prepare_data()
    
    # 3. Build Model
    print("\n🔨 Building dual-head model...")
    model = create_hierarchical_model(
        len(dataset.main_classes),
        len(dataset.sub_classes)
    )
    
    # 4. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss={
            "main_output": "binary_crossentropy", 
            "sub_output": "binary_crossentropy"
        },
        loss_weights={
            "main_output": 0.5,
            "sub_output": 1.0
        },
        metrics={"main_output": "binary_accuracy", "sub_output": "binary_accuracy"}
    )
    
    model.summary()
    
    # 5. Train
    print(f"\n🚀 Training for {args.epochs} epochs...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased from 8 to give it more time to overcome plateaus
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=5,   # Increased from 4
            factor=0.3,   # Harder drop when stuck (was 0.5)
            min_lr=1e-7   # Lower minimum (was 1e-6)
        )
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks
    )
    
    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"hierarchical_model_{timestamp}"
    MODELS_DIR.mkdir(exist_ok=True)
    
    save_path = MODELS_DIR / f"{name}.keras"
    model.save(str(save_path))
    print(f"\n💾 Model saved to: {save_path}")
    
    # Save labels
    labels = {
        "main_classes": dataset.main_classes,
        "sub_classes": dataset.sub_classes
    }
    labels_path = MODELS_DIR / f"{name}_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"📋 Labels saved to: {labels_path}")
    
    # Save training history
    history_path = MODELS_DIR / f"{name}_history.json"
    with open(history_path, "w") as f:
        hist_dict = {}
        for k, v in history.history.items():
            hist_dict[k] = [float(x) for x in v]
        json.dump(hist_dict, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Train Hierarchical Forensic Audio Classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
