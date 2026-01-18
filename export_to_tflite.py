"""
Export trained Keras model to TFLite format for deployment.
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("❌ TensorFlow not installed")

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
PARENT_SCRIPTS_DIR = SCRIPT_DIR.parent  # The main scripts folder

def export_to_tflite(model_path: str, quantize: bool = False):
    """Convert Keras model to TFLite format."""
    if not TF_AVAILABLE:
        print("❌ TensorFlow is required. Run: pip install tensorflow")
        return
    
    print("="*50)
    print("📦 TFLITE EXPORT TOOL")
    print("="*50)
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    # Load the Keras model
    print(f"\n📂 Loading model: {model_path}")
    model = tf.keras.models.load_model(str(model_path))
    
    # Create TFLite converter
    print("\n🔄 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        print("  Applying quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_name = model_path.stem.replace("_classifier", "") + ".tflite"
    tflite_path = MODELS_DIR / tflite_name
    
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"\n✅ TFLite model saved: {tflite_path}")
    print(f"   Size: {len(tflite_model) / 1024 / 1024:.2f} MB")
    
    # Copy labels file
    labels_src = Path(str(model_path).replace(".keras", "_labels.json"))
    if labels_src.exists():
        labels_dst = MODELS_DIR / (model_path.stem.replace("_classifier", "") + "_labels.json")
        shutil.copy(labels_src, labels_dst)
        print(f"   Labels: {labels_dst}")
    
    print("\n" + "="*50)
    print("📋 DEPLOYMENT INSTRUCTIONS")
    print("="*50)
    print(f"""
To use the new model in your app:

1. Backup the original model:
   copy scripts\\yamnet.tflite scripts\\yamnet_backup.tflite

2. Replace with your custom model:
   copy {tflite_path} scripts\\yamnet.tflite

3. Update the label mapping in mediapipe_audio_classifier.py:
   - Open scripts/mediapipe_audio_classifier.py
   - Update the map_to_forensic_category() function with your classes

4. Restart the application:
   npm run dev
""")
    
    return tflite_path

def find_latest_model():
    """Find the most recently trained model."""
    if not MODELS_DIR.exists():
        return None
    
    keras_models = list(MODELS_DIR.glob("*.keras"))
    if not keras_models:
        return None
    
    # Sort by modification time
    keras_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return keras_models[0]

def main():
    parser = argparse.ArgumentParser(description="Export model to TFLite")
    parser.add_argument("--model", type=str, help="Path to Keras model file")
    parser.add_argument("--quantize", action="store_true", help="Apply quantization for smaller size")
    
    args = parser.parse_args()
    
    if args.model:
        model_path = args.model
    else:
        # Try to find latest model
        model_path = find_latest_model()
        if model_path:
            print(f"📂 Found latest model: {model_path}")
        else:
            print("❌ No trained models found. Run train_forensic_model.py first!")
            return
    
    export_to_tflite(model_path, args.quantize)

if __name__ == "__main__":
    main()
