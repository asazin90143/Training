"""
Knowledge Distillation: Train a tiny Student model from a Teacher Ensemble.
The Student learns to mimic the combined predictions of multiple large models,
giving ensemble-level accuracy in a single tiny model.

Usage:
    python train_student_model.py --epochs 50
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

import sys
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Also add own directory for sibling imports
OWN_DIR = str(Path(__file__).parent)
if OWN_DIR not in sys.path:
    sys.path.insert(0, OWN_DIR)
from config import get_paths
PATHS = get_paths("student")

MODELS_DIR = PATHS["models"]
MANIFEST_PATH = PATHS["manifest"]
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

from train_forensic_model import HierarchicalDataset, BACKBONE_CONFIG


def load_teacher_models():
    """Load all trained models as the Teacher ensemble."""
    # Search all subdirectories for models
    models = list(PATHS["models_root"].rglob("*.keras"))
    teachers = []
    for m_path in models:
        try:
            model = tf.keras.models.load_model(str(m_path))
            teachers.append(model)
            print(f"   ✅ Teacher: {m_path.name}")
        except Exception:
            pass
    return teachers


def generate_soft_targets(teachers, X_data, temperature=3.0):
    """Generate soft probability targets by averaging teacher predictions."""
    all_main_preds = []
    all_sub_preds = []
    
    for teacher in teachers:
        preds = teacher.predict(X_data, verbose=0)
        # Apply temperature scaling for softer probabilities
        main_soft = tf.nn.sigmoid(tf.math.log(preds[0] / (1 - preds[0] + 1e-8)) / temperature).numpy()
        sub_soft = tf.nn.sigmoid(tf.math.log(preds[1] / (1 - preds[1] + 1e-8)) / temperature).numpy()
        all_main_preds.append(main_soft)
        all_sub_preds.append(sub_soft)
    
    # Average across all teachers
    avg_main = np.mean(all_main_preds, axis=0)
    avg_sub = np.mean(all_sub_preds, axis=0)
    
    return avg_main, avg_sub


def create_student_model(num_main, num_sub, embedding_size=1024):
    """Build a tiny, fast student model."""
    input_layer = tf.keras.layers.Input(shape=(embedding_size * 2,))
    
    # Much smaller than the teacher
    x = tf.keras.layers.Dense(256, activation='relu')(input_layer)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    main_out = tf.keras.layers.Dense(num_main, activation='sigmoid', name='main_output')(x)
    sub_out = tf.keras.layers.Dense(num_sub, activation='sigmoid', name='sub_output')(x)
    
    return tf.keras.Model(inputs=input_layer, outputs=[main_out, sub_out])


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation: Student/Teacher Training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Temperature for soft targets (higher = softer)")
    args = parser.parse_args()
    
    print("="*50)
    print("🎓 KNOWLEDGE DISTILLATION TRAINING")
    print("="*50)
    
    # 1. Load Teachers
    print("\n📂 Loading Teacher ensemble...")
    teachers = load_teacher_models()
    if len(teachers) < 1:
        print("❌ No teacher models found. Train at least one model first.")
        return
    print(f"   Total teachers: {len(teachers)}")
    
    # 2. Load Data
    bb_cfg = BACKBONE_CONFIG["yamnet"]
    print("\n📦 Loading YAMNet backbone...")
    feature_model = hub.load(bb_cfg["url"])
    
    dataset = HierarchicalDataset(MANIFEST_PATH, feature_model, backbone="yamnet")
    (X_train, y_train_hard), (X_test, y_test_hard) = dataset.prepare_data(mixup=False)
    
    # 3. Generate soft targets from teachers
    print(f"\n🌡️ Generating soft targets (temperature={args.temperature})...")
    soft_main_train, soft_sub_train = generate_soft_targets(teachers, X_train, args.temperature)
    
    y_train_soft = {"main_output": soft_main_train, "sub_output": soft_sub_train}
    
    # 4. Create tiny student
    emb_size = bb_cfg["embedding_size"]
    student = create_student_model(len(dataset.main_classes), len(dataset.sub_classes), emb_size)
    
    student.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={"main_output": "binary_crossentropy", "sub_output": "binary_crossentropy"},
        metrics={"main_output": "binary_accuracy", "sub_output": "binary_accuracy"}
    )
    
    student.summary()
    
    # 5. Train student on soft targets
    print(f"\n🚀 Training Student on Teacher's soft targets...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-7)
    ]
    
    student.fit(X_train, y_train_soft,
                validation_data=(X_test, y_test_hard),
                epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks)
    
    # 6. Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"student_model_{timestamp}"
    student.save(str(MODELS_DIR / f"{name}.keras"))
    with open(MODELS_DIR / f"{name}_labels.json", "w") as f:
        json.dump({"main_classes": dataset.main_classes, "sub_classes": dataset.sub_classes}, f, indent=2)
    
    print(f"\n✅ Student model saved: {name}.keras")
    print(f"💡 This tiny model has absorbed the knowledge of {len(teachers)} teachers!")


if __name__ == "__main__":
    main()
