"""
Automated Hyperparameter Tuning using KerasTuner
Trains hundreds of model variations automatically to find the optimal
architecture configuration for your specific dataset.

Requires: pip install keras-tuner

Usage:
    python tune_hyperparameters.py --max_trials 50
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

import tensorflow as tf
import tensorflow_hub as hub
import librosa

try:
    import keras_tuner as kt
    KT_AVAILABLE = True
except ImportError:
    KT_AVAILABLE = False
    print("⚠️ keras-tuner not installed. Run: pip install keras-tuner")

from config import get_paths
PATHS = get_paths()

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Import dataset class from main training script
from train_forensic_model import HierarchicalDataset, BACKBONE_CONFIG


def build_tunable_model(hp, num_main, num_sub, embedding_size=1024):
    """Build a model with tunable hyperparameters."""
    input_layer = tf.keras.layers.Input(shape=(embedding_size * 2,))
    
    # Tunable shared layers
    units_1 = hp.Int('dense_1_units', min_value=256, max_value=1536, step=256)
    dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.6, step=0.1)
    
    x = tf.keras.layers.Dense(units_1, activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_1)(x)
    
    units_2 = hp.Int('dense_2_units', min_value=128, max_value=768, step=128)
    dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    
    x = tf.keras.layers.Dense(units_2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout_2)(x)
    
    # Optional 3rd layer
    use_third = hp.Boolean('use_third_layer')
    if use_third:
        units_3 = hp.Int('dense_3_units', min_value=64, max_value=512, step=64)
        x = tf.keras.layers.Dense(units_3, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
    
    # Main head
    main_units = hp.Int('main_head_units', min_value=64, max_value=512, step=64)
    main_branch = tf.keras.layers.Dense(main_units, activation='relu')(x)
    main_output = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(main_branch)
    
    # Sub head
    sub_branch = tf.keras.layers.concatenate([x, main_branch])
    sub_units = hp.Int('sub_head_units', min_value=128, max_value=768, step=128)
    sub_branch = tf.keras.layers.Dense(sub_units, activation='relu')(sub_branch)
    sub_branch = tf.keras.layers.Dropout(0.2)(sub_branch)
    sub_output = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(sub_branch)
    
    model = tf.keras.Model(inputs=input_layer, outputs=[main_output, sub_output])
    
    # Tunable learning rate
    lr = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={"main_output": "binary_crossentropy", "sub_output": "binary_crossentropy"},
        loss_weights={
            "main_output": hp.Float('main_loss_weight', 0.3, 0.7, step=0.1),
            "sub_output": 1.0
        },
        metrics={"main_output": "binary_accuracy", "sub_output": "binary_accuracy"}
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Auto-Tune Hyperparameters")
    parser.add_argument("--max_trials", type=int, default=50, help="Number of model variations to try")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial")
    parser.add_argument("--backbone", type=str, default="yamnet", choices=["yamnet", "vggish"])
    args = parser.parse_args()
    
    if not KT_AVAILABLE:
        print("❌ Install: pip install keras-tuner")
        return
    
    print("="*50)
    print("🔧 AUTOMATED HYPERPARAMETER TUNING")
    print("="*50)
    
    # Load data
    bb_cfg = BACKBONE_CONFIG[args.backbone]
    print(f"\n📦 Loading {args.backbone.upper()} backbone...")
    feature_model = hub.load(bb_cfg["url"])
    
    dataset = HierarchicalDataset(MANIFEST_PATH, feature_model, backbone=args.backbone)
    (X_train, y_train), (X_test, y_test) = dataset.prepare_data()
    
    emb_size = bb_cfg["embedding_size"]
    num_main = len(dataset.main_classes)
    num_sub = len(dataset.sub_classes)
    
    # Setup tuner
    tuner = kt.RandomSearch(
        lambda hp: build_tunable_model(hp, num_main, num_sub, emb_size),
        objective='val_loss',
        max_trials=args.max_trials,
        executions_per_trial=1,
        directory=str(MODELS_DIR / "tuner_results"),
        project_name="forensic_tuning"
    )
    
    print(f"\n🚀 Starting {args.max_trials} trials (this may take hours)...\n")
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
        ]
    )
    
    # Get best model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print(f"\n{'='*50}")
    print(f"🏆 BEST HYPERPARAMETERS FOUND:")
    print(f"{'='*50}")
    for key, val in best_hp.values.items():
        print(f"   {key}: {val}")
    
    # Save best model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"tuned_model_{timestamp}"
    best_model.save(str(MODELS_DIR / f"{name}.keras"))
    with open(MODELS_DIR / f"{name}_labels.json", "w") as f:
        json.dump({"main_classes": dataset.main_classes, "sub_classes": dataset.sub_classes}, f, indent=2)
    
    # Save best hyperparameters
    with open(MODELS_DIR / f"{name}_best_hp.json", "w") as f:
        json.dump(best_hp.values, f, indent=2)
    
    print(f"\n✅ Best model saved: {name}.keras")
    print(f"💡 Use these hyperparameters in train_forensic_model.py for production!")


if __name__ == "__main__":
    main()
