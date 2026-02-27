"""
Global Configuration for Forensic Audio Training
Edit DATASET_ROOT to point to your external drive.
"""

from pathlib import Path

# Project root (the 'training' folder)
PROJECT_ROOT = Path(__file__).parent
SCRIPT_DIR = PROJECT_ROOT  # Backward compatibility

# --- EXTERNAL DRIVE PATH ---
# Set this to the root folder on your external drive that contains the dataset.
DATASET_ROOT = r"D:\dataset"

# Explicit processed path
PROCESSED_ROOT = r"D:\processed"

# -------------------------------------------------

# Per-backbone model directories
BACKBONE_MODEL_DIRS = {
    "yamnet": "yamnet",
    "vggish": "vggish",
    "spectrogram": "spectrogram",
    "wav2vec": "wav2vec",
    "student": "student",
    "tuned": "tuned",
}

def get_paths(backbone=None):
    """Returns all configured paths. Optionally returns backbone-specific model dir."""
    
    base_models = PROJECT_ROOT / "models"
    
    if backbone and backbone in BACKBONE_MODEL_DIRS:
        models_dir = base_models / BACKBONE_MODEL_DIRS[backbone]
        models_dir.mkdir(parents=True, exist_ok=True)
    else:
        models_dir = base_models
    
    return {
        "project_root": PROJECT_ROOT,
        "dataset": Path(DATASET_ROOT),
        "processed": Path(PROCESSED_ROOT),
        "models": models_dir,
        "models_root": base_models,
        "manifest": PROJECT_ROOT / "data_manifest.json"
    }

