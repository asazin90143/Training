"""
K-Fold Cross-Validation Training Script
Trains multiple model folds and reports robust average accuracy.
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
    print("❌ TensorFlow not installed.")

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ sklearn not installed. Run: pip install scikit-learn")

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROCESSED_DIR = SCRIPT_DIR / "processed"
MODELS_DIR = SCRIPT_DIR / "models"
MANIFEST_PATH = SCRIPT_DIR / "data_manifest.json"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
EMBEDDING_SIZE = 2048

def extract_all_embeddings(manifest_path, yamnet_model):
    """Extract embeddings from all audio files."""
    with open(manifest_path) as f:
        data = json.load(f)
    
    classes = data["classes"]
    samples = data["samples"]
    
    print(f"📊 Processing {len(samples)} samples...")
    
    X = []
    y = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  Extracting: {i + 1}/{len(samples)}")
        
        try:
            audio_path = SCRIPT_DIR / sample["file"]
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            _, embeddings, _ = yamnet_model(audio)
            
            mean_emb = tf.reduce_mean(embeddings, axis=0)
            max_emb = tf.reduce_max(embeddings, axis=0)
            final_emb = tf.concat([mean_emb, max_emb], axis=0)
            
            X.append(final_emb.numpy())
            y.append(sample["class_id"])
        except Exception as e:
            print(f"  ⚠️ Error: {sample['file']}: {e}")
    
    return np.array(X), np.array(y), classes

def create_model(num_classes):
    """Create fresh model for each fold."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(EMBEDDING_SIZE,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def run_kfold_validation(args):
    """Run K-Fold cross-validation."""
    if not all([TF_AVAILABLE, LIBROSA_AVAILABLE, SKLEARN_AVAILABLE]):
        print("❌ Missing dependencies.")
        return
    
    print("="*50)
    print(f"🔄 K-FOLD CROSS VALIDATION (K={args.folds})")
    print("="*50)
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    # Load YAMNet
    print("\n📦 Loading YAMNet...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    
    # Extract all embeddings once
    X, y, classes = extract_all_embeddings(MANIFEST_PATH, yamnet_model)
    num_classes = len(classes)
    
    print(f"\n🏷️ Classes: {classes}")
    print(f"📊 Total samples: {len(X)}")
    
    # K-Fold splitter
    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"📂 FOLD {fold_idx + 1}/{args.folds}")
        print(f"{'='*50}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"  Training: {len(X_train)} | Validation: {len(X_val)}")
        
        # Compute class weights
        class_counts = np.bincount(y_train, minlength=num_classes)
        total = len(y_train)
        class_weights = {i: total / (num_classes * max(1, class_counts[i])) for i in range(num_classes)}
        
        # Create fresh model
        model = create_model(num_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluate
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        fold_results.append({
            "fold": fold_idx + 1,
            "accuracy": float(accuracy),
            "loss": float(loss),
            "train_size": len(X_train),
            "val_size": len(X_val)
        })
        
        print(f"  ✅ Fold {fold_idx + 1} Accuracy: {accuracy*100:.2f}%")
    
    # Summary
    accuracies = [r["accuracy"] for r in fold_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    print("\n" + "="*50)
    print("📊 K-FOLD CROSS VALIDATION RESULTS")
    print("="*50)
    print(f"\n  Folds: {args.folds}")
    print(f"  Individual Accuracies: {[f'{a*100:.1f}%' for a in accuracies]}")
    print(f"\n  📈 Mean Accuracy: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print("="*50)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = MODELS_DIR / f"kfold_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump({
            "folds": args.folds,
            "mean_accuracy": float(mean_acc),
            "std_accuracy": float(std_acc),
            "fold_details": fold_results,
            "classes": classes
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")

def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross Validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per fold")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    run_kfold_validation(args)

if __name__ == "__main__":
    main()
