"""
Custom Forensic Audio Model Training Script
Fine-tunes YAMNet embeddings for forensic sound classification.
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# TensorFlow imports
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not installed. Run: pip install tensorflow tensorflow-hub")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "processed"
MODELS_DIR = SCRIPT_DIR / "models"
MANIFEST_PATH = SCRIPT_DIR / "data_manifest.json"

# YAMNet model URL
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Training configuration
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
EMBEDDING_SIZE = 1024  # YAMNet embedding dimension

class ForensicAudioDataset:
    """Dataset handler for forensic audio training."""
    
    def __init__(self, manifest_path: Path, yamnet_model):
        self.manifest_path = manifest_path
        self.yamnet_model = yamnet_model
        self.samples = []
        self.classes = []
        
        self._load_manifest()
    
    def _load_manifest(self):
        """Load the data manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}\n"
                "Run preprocess_audio.py first!"
            )
        
        with open(self.manifest_path) as f:
            data = json.load(f)
        
        self.classes = data["classes"]
        self.samples = data["samples"]
        
        print(f"📊 Loaded {len(self.samples)} samples across {len(self.classes)} classes")
    
    def extract_embeddings(self, audio_path: str) -> np.ndarray:
        """Extract YAMNet embeddings from audio file."""
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Run through YAMNet to get embeddings
        # YAMNet returns: scores, embeddings, spectrogram
        _, embeddings, _ = self.yamnet_model(audio)
        
        # Average embeddings across time
        mean_embedding = tf.reduce_mean(embeddings, axis=0)
        
        return mean_embedding.numpy()
    
    def prepare_data(self, test_split: float = 0.2):
        """Prepare training and test datasets."""
        print("\n🔄 Extracting embeddings from audio files...")
        
        X = []
        y = []
        
        for i, sample in enumerate(self.samples):
            if (i + 1) % 10 == 0:
                print(f"  Processing: {i + 1}/{len(self.samples)}")
            
            try:
                audio_path = SCRIPT_DIR / sample["file"]
                embedding = self.extract_embeddings(str(audio_path))
                X.append(embedding)
                y.append(sample["class_id"])
            except Exception as e:
                print(f"  ⚠️ Error processing {sample['file']}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle and split
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        split_idx = int(len(X) * (1 - test_split))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\n📊 Data split:")
        print(f"  Training: {len(X_train)} samples")
        print(f"  Testing: {len(X_test)} samples")
        
        return (X_train, y_train), (X_test, y_test)

def create_classifier_model(num_classes: int):
    """Create the classification head model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(EMBEDDING_SIZE,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_model(args):
    """Main training function."""
    if not TF_AVAILABLE or not LIBROSA_AVAILABLE:
        print("❌ Missing dependencies. Install them first:")
        print("   pip install tensorflow tensorflow-hub librosa")
        return
    
    print("="*50)
    print("🎓 FORENSIC AUDIO MODEL TRAINING")
    print("="*50)
    
    # Create models directory
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load YAMNet for embedding extraction
    print("\n📦 Loading YAMNet model...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("✅ YAMNet loaded successfully")
    
    # Initialize dataset
    dataset = ForensicAudioDataset(MANIFEST_PATH, yamnet_model)
    
    # Prepare data
    (X_train, y_train), (X_test, y_test) = dataset.prepare_data()
    
    num_classes = len(dataset.classes)
    print(f"\n🏷️ Classes: {dataset.classes}")
    
    # Create model
    print("\n🔨 Building classifier model...")
    model = create_classifier_model(num_classes)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📊 Final Evaluation:")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"  Test Loss: {loss:.4f}")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"forensic_classifier_{timestamp}"
    
    # Save Keras model
    keras_path = MODELS_DIR / f"{model_name}.keras"
    model.save(str(keras_path))
    print(f"\n💾 Model saved to: {keras_path}")
    
    # Save class labels
    labels_path = MODELS_DIR / f"{model_name}_labels.json"
    with open(labels_path, "w") as f:
        json.dump({
            "classes": dataset.classes,
            "accuracy": float(accuracy),
            "trained_on": timestamp
        }, f, indent=2)
    print(f"📋 Labels saved to: {labels_path}")
    
    # Save training history
    history_path = MODELS_DIR / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "accuracy": [float(x) for x in history.history['accuracy']],
            "val_accuracy": [float(x) for x in history.history['val_accuracy']],
            "loss": [float(x) for x in history.history['loss']],
            "val_loss": [float(x) for x in history.history['val_loss']]
        }, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ TRAINING COMPLETE!")
    print("="*50)
    print(f"\nNext steps:")
    print(f"  1. Run: python export_to_tflite.py --model {keras_path}")
    print(f"  2. Replace yamnet.tflite with the exported model")

def main():
    parser = argparse.ArgumentParser(description="Train Forensic Audio Classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    
    args = parser.parse_args()
    train_model(args)

if __name__ == "__main__":
    main()
