"""
Global Configuration for Forensic Audio Training
Edit DATASET_ROOT to point to your external drive.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# --- EXTERNAL DRIVE PATH ---
# Set this to the root folder on your external drive that contains the dataset.
DATASET_ROOT = r"D:\dataset"

# Explicit processed path
PROCESSED_ROOT = r"D:\processed"

# -------------------------------------------------

def get_paths():
    """Returns all configured paths."""
    
    return {
        "dataset": Path(DATASET_ROOT),
        "processed": Path(PROCESSED_ROOT),
        "models": SCRIPT_DIR / "models",
        "manifest": SCRIPT_DIR / "data_manifest.json"
    }
