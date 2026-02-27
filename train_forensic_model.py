"""
Custom Forensic Audio Model Training Script (Hierarchical)
Trains a model with two classification heads:
1. Main Class (e.g., Vehicle)
2. Sub Class (e.g., Siren)

Supports optional YAMNet fine-tuning with --finetune flag.
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

# Feature Extractor Backbones
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"

BACKBONE_CONFIG = {
    "yamnet": {"url": YAMNET_MODEL_URL, "embedding_size": 1024, "returns_tuple": True},
    "vggish": {"url": VGGISH_MODEL_URL, "embedding_size": 128, "returns_tuple": False},
}

# Training Config
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
FINETUNE_LEARNING_RATE = 1e-5
EMBEDDING_SIZE = 1024  # Default (YAMNet)


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    """Supervised Contrastive Loss for tighter class clustering in embedding space.
    Forces embeddings of the same class to cluster together while pushing
    different classes apart. Use with --contrastive flag."""
    def __init__(self, temperature=0.07, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature
    
    def call(self, labels, embeddings):
        # L2 normalize embeddings
        embeddings = tf.math.l2_normalize(embeddings, axis=1)
        # Compute similarity matrix
        similarity = tf.matmul(embeddings, embeddings, transpose_b=True)
        similarity = similarity / self.temperature
        
        # Create mask: 1 where labels match, 0 otherwise
        labels = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        labels_eq = tf.cast(tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1)), tf.float32)
        
        # Remove diagonal (self-similarity)
        batch_size = tf.shape(embeddings)[0]
        mask = tf.ones_like(labels_eq) - tf.eye(batch_size)
        labels_eq = labels_eq * mask
        
        # Log-sum-exp trick for numerical stability
        logits_max = tf.reduce_max(similarity * mask, axis=1, keepdims=True)
        logits = (similarity - logits_max) * mask
        
        exp_logits = tf.exp(logits) * mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)
        
        # Mean of log-likelihood over positive pairs
        pos_count = tf.reduce_sum(labels_eq, axis=1)
        mean_log_prob = tf.reduce_sum(labels_eq * log_prob, axis=1) / (pos_count + 1e-8)
        
        loss = -mean_log_prob
        # Only use samples that have at least one positive pair
        valid = tf.cast(pos_count > 0, tf.float32)
        loss = tf.reduce_sum(loss * valid) / (tf.reduce_sum(valid) + 1e-8)
        return loss


class HierarchicalDataset:
    def __init__(self, manifest_path: Path, yamnet_model, backbone="yamnet"):
        self.manifest_path = manifest_path
        self.yamnet_model = yamnet_model
        self.backbone = backbone
        self.backbone_cfg = BACKBONE_CONFIG[backbone]
        
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
        """Extract embeddings from a processed audio file using selected backbone."""
        try:
            wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            if self.backbone_cfg["returns_tuple"]:
                # YAMNet returns (scores, embeddings, spectrogram)
                _, embeddings, _ = self.yamnet_model(wav_data)
            else:
                # VGGish returns embeddings directly
                embeddings = self.yamnet_model(wav_data)
                if len(embeddings.shape) == 1:
                    embeddings = tf.expand_dims(embeddings, axis=0)
            
            mean_emb = tf.reduce_mean(embeddings, axis=0)
            max_emb = tf.reduce_max(embeddings, axis=0)
            final_emb = tf.concat([mean_emb, max_emb], axis=0)
            return final_emb.numpy()
        except Exception as e:
            print(f"  ⚠️ Error reading {audio_path}: {e}")
            return None

    def prepare_data(self, test_split=0.2, mixup=True):
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
        y_main_train, y_main_test = y_main_oh[:split_point], y_main_oh[split_point:]
        y_sub_train, y_sub_test = y_sub_oh[:split_point], y_sub_oh[split_point:]
        
        # MixUp Augmentation (training set only)
        if mixup and len(X_train) > 10:
            X_train, y_main_train, y_sub_train = self._apply_mixup(
                X_train, y_main_train, y_sub_train
            )
        
        y_train = {
            "main_output": y_main_train,
            "sub_output": y_sub_train
        }
        y_test = {
            "main_output": y_main_test,
            "sub_output": y_sub_test
        }
        
        print(f"\n📊 Data split:")
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing:  {len(X_test)} samples")
        
        return (X_train, y_train), (X_test, y_test)
    
    def _apply_mixup(self, X, y_main, y_sub, alpha=0.4, ratio=0.3):
        """
        MixUp augmentation: blend random pairs of samples together.
        This teaches the model that multiple sounds can co-exist.
        alpha: Beta distribution parameter (lower = closer to original)
        ratio: proportion of extra mixed samples to add (0.3 = 30%)
        """
        n_samples = len(X)
        n_mix = int(n_samples * ratio)
        
        print(f"   🔀 Applying MixUp augmentation ({n_mix} blended samples)...")
        
        X_mix = []
        y_main_mix = []
        y_sub_mix = []
        
        for _ in range(n_mix):
            # Pick two random samples
            i, j = np.random.choice(n_samples, 2, replace=False)
            
            # Random blend ratio from Beta distribution
            lam = np.random.beta(alpha, alpha)
            
            # Blend embeddings
            x_new = lam * X[i] + (1 - lam) * X[j]
            
            # Blend labels (soft labels enable multi-label learning)
            y_m_new = lam * y_main[i] + (1 - lam) * y_main[j]
            y_s_new = lam * y_sub[i] + (1 - lam) * y_sub[j]
            
            X_mix.append(x_new)
            y_main_mix.append(y_m_new)
            y_sub_mix.append(y_s_new)
        
        # Concatenate original + mixed
        X_out = np.concatenate([X, np.array(X_mix)], axis=0)
        y_main_out = np.concatenate([y_main, np.array(y_main_mix)], axis=0)
        y_sub_out = np.concatenate([y_sub, np.array(y_sub_mix)], axis=0)
        
        # Shuffle everything
        idx = np.random.permutation(len(X_out))
        return X_out[idx], y_main_out[idx], y_sub_out[idx]


def create_hierarchical_model(num_main, num_sub, embedding_size=1024):
    """Build dual-head classification model (embedding-based)."""
    input_layer = tf.keras.layers.Input(shape=(embedding_size * 2,))
    
    # Shared Layers
    x = tf.keras.layers.Dense(1024, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Head 1: Main Class
    main_branch = tf.keras.layers.Dense(256, activation='relu')(x)
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(main_branch)
    
    # Head 2: Sub Class (receives shared + main context)
    sub_branch = tf.keras.layers.concatenate([x, main_branch])
    sub_branch = tf.keras.layers.Dense(512, activation='relu')(sub_branch)
    sub_branch = tf.keras.layers.Dropout(0.2)(sub_branch)
    sub_output = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[main_output, sub_output])
    return model


class YAMNetLayer(tf.keras.layers.Layer):
    """Wraps YAMNet as a trainable Keras layer for fine-tuning."""
    def __init__(self, yamnet_url, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.yamnet_url = yamnet_url
        self.hub_layer = hub.KerasLayer(yamnet_url, trainable=trainable)
    
    def call(self, inputs):
        # hub.KerasLayer for YAMNet returns (scores, embeddings, spectrogram)
        scores, embeddings, spectrogram = self.hub_layer(inputs)
        return embeddings


def create_finetune_model(num_main, num_sub):
    """Build end-to-end model with YAMNet as a trainable feature extractor."""
    # Input: raw 16kHz waveform (variable length)
    input_layer = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='waveform')
    
    # YAMNet feature extraction (trainable)
    yamnet_layer = YAMNetLayer(YAMNET_MODEL_URL, trainable=True, name='yamnet')
    embeddings = yamnet_layer(input_layer)  # (num_frames, 1024)
    
    # Pooling: Mean + Max
    mean_emb = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    max_emb = tf.keras.layers.GlobalMaxPooling1D()(embeddings)
    x = tf.keras.layers.Concatenate()([mean_emb, max_emb])  # (2048,)
    
    # Shared Layers
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Head 1: Main Class
    main_branch = tf.keras.layers.Dense(256, activation='relu')(x)
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(main_branch)
    
    # Head 2: Sub Class
    sub_branch = tf.keras.layers.concatenate([x, main_branch])
    sub_branch = tf.keras.layers.Dense(512, activation='relu')(sub_branch)
    sub_branch = tf.keras.layers.Dropout(0.2)(sub_branch)
    sub_output = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[main_output, sub_output])
    return model


def create_lstm_model(num_main, num_sub, embedding_size=1024, max_frames=10):
    """Build LSTM-based model that processes embedding sequences for temporal learning."""
    input_layer = tf.keras.layers.Input(shape=(max_frames, embedding_size), name='embedding_sequence')
    
    # Bidirectional LSTM captures patterns in both directions
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=True)
    )(input_layer)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=False)
    )(x)
    
    # Shared Dense Layers
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Head 1: Main Class
    main_branch = tf.keras.layers.Dense(128, activation='relu')(x)
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(main_branch)
    
    # Head 2: Sub Class
    sub_branch = tf.keras.layers.concatenate([x, main_branch])
    sub_branch = tf.keras.layers.Dense(256, activation='relu')(sub_branch)
    sub_branch = tf.keras.layers.Dropout(0.2)(sub_branch)
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
    
    # 1. Load Feature Extractor
    backbone = args.backbone
    bb_cfg = BACKBONE_CONFIG[backbone]
    print(f"\n📦 Loading {backbone.upper()} backbone...")
    feature_model = hub.load(bb_cfg["url"])
    print(f"✅ {backbone.upper()} loaded (embedding size: {bb_cfg['embedding_size']})")
    
    # 2. Load Data
    dataset = HierarchicalDataset(MANIFEST_PATH, feature_model, backbone=backbone)
    (X_train, y_train), (X_test, y_test) = dataset.prepare_data()
    
    # 3. Build Model
    emb_size = bb_cfg["embedding_size"]
    if args.finetune:
        print("\n🔨 Building FINE-TUNE model (YAMNet trainable)...")
        model = create_finetune_model(
            len(dataset.main_classes),
            len(dataset.sub_classes)
        )
    else:
        print("\n🔨 Building dual-head model...")
        model = create_hierarchical_model(
            len(dataset.main_classes),
            len(dataset.sub_classes),
            embedding_size=emb_size
        )
    
    # 4. Compile
    lr = FINETUNE_LEARNING_RATE if args.finetune else args.learning_rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
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
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune YAMNet base model (slower but more accurate)")
    parser.add_argument("--backbone", type=str, default="yamnet", choices=["yamnet", "vggish"],
                        help="Feature extractor backbone (default: yamnet)")
    args = parser.parse_args()
    
    train_model(args)


if __name__ == "__main__":
    main()
