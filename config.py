"""
Global Configuration for Forensic Audio Training
Edit DATASET_ROOT to point to your external drive.
"""

from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# --- EXTERNAL DRIVE PATH ---
# Set this to the root folder on your external drive that contains the dataset.
# The 'processed' folder will also be created here.
DATASET_ROOT = r"E:\dataset"

# -------------------------------------------------

def get_paths():
    """Returns all configured paths."""
    root = Path(DATASET_ROOT)
    
    return {
        "dataset": root,                          # E:\dataset  (where your class folders are)
        "processed": root.parent / "processed",    # E:\processed (processed output next to dataset)
        "models": SCRIPT_DIR / "models",           # Local models/ folder (stays with code)
        "manifest": SCRIPT_DIR / "data_manifest.json"  # Manifest stays with code
    }
