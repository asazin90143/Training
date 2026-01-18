
import os
import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path

# Disable GPU to avoid memory issues for single prediction
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

def load_latest_model():
    """Find and load the most recently trained model and its labels."""
    if not MODELS_DIR.exists():
        print("❌ No models folder found. Train the model first!")
        return None, None

    # Find latest .keras file
    models = list(MODELS_DIR.glob("*.keras"))
    if not models:
        print("❌ No trained models found.")
        return None, None
    
    latest_model_path = max(models, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading model: {latest_model_path.name}")
    
    # Load Keras model
    model = tf.keras.models.load_model(str(latest_model_path))
    
    # Load labels
    labels_path = latest_model_path.parent / f"{latest_model_path.stem}_labels.json"
    if not labels_path.exists():
        print("⚠️ Labels file not found, predicting raw IDs.")
        labels = None
    else:
        with open(labels_path, "r") as f:
            labels = json.load(f)["classes"]
            
    return model, labels

def extract_features(audio_path, yamnet_model):
    """Process audio file into embeddings (Mean + Max)."""
    try:
        # 1. Load Audio (resample to 16kHz mono)
        wav_data, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # 2. Run YAMNet
        # scores, embeddings, spectrogram
        _, embeddings, _ = yamnet_model(wav_data)
        
        # 3. Apply Mean + Max Pooling (Matches training logic)
        mean_emb = tf.reduce_mean(embeddings, axis=0)
        max_emb = tf.reduce_max(embeddings, axis=0)
        
        # Concatenate
        final_emb = tf.concat([mean_emb, max_emb], axis=0)
        
        # Add batch dimension (1, 2048)
        return tf.expand_dims(final_emb, axis=0)
        
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return None

def predict_audio(file_path):
    print("="*50)
    print("🔎 FORENSIC AUDIO ANALYZER")
    print("="*50)
    
    # 1. Load Custom Model
    model, classes = load_latest_model()
    if model is None:
        return

    # 2. Load YAMNet
    print("📦 Loading YAMNet base model...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    
    # 3. Process Audio
    print(f"🎵 Analyzing file: {file_path}")
    features = extract_features(file_path, yamnet_model)
    
    if features is not None:
        # 4. Predict
        predictions = model.predict(features, verbose=0)
        
        # Get top result
        top_idx = np.argmax(predictions[0])
        confidence = predictions[0][top_idx]
        
        print("\n" + "-"*30)
        if classes:
            result_name = classes[top_idx]
            print(f"🎯 RESULT: {result_name.upper()}")
        else:
            print(f"🎯 RESULT: Class ID {top_idx}")
        print(f"📊 Confidence: {confidence*100:.1f}%")
        print("-"*-30)
        
        # Show all probabilities
        if classes:
            print("\nFull Analysis:")
            # Sort by confidence
            sorted_indices = np.argsort(predictions[0])[::-1]
            for idx in sorted_indices:
                print(f"  {classes[idx]:<20}: {predictions[0][idx]*100:5.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Test your forensic model on an audio file")
    parser.add_argument("file", nargs='?', help="Path to audio file (wav/mp3)")
    
    args = parser.parse_args()
    
    if args.file:
        predict_audio(args.file)
    else:
        print("\n⚠️ Please provide an audio file path.")
        print("Usage: python test_model.py my_recording.wav")

if __name__ == "__main__":
    main()
