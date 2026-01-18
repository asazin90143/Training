"""
Model Registry and Comparison Tool
Track and compare different model versions.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
MODELS_DIR = SCRIPT_DIR / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

def load_registry():
    """Load existing registry or create new one."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {"models": {}, "created": datetime.now().isoformat()}

def save_registry(registry):
    """Save registry to file."""
    MODELS_DIR.mkdir(exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

def register_model(model_name: str, accuracy: float, notes: str = ""):
    """Register a new model version."""
    registry = load_registry()
    
    # Find model file
    model_file = MODELS_DIR / f"{model_name}.keras"
    if not model_file.exists():
        # Try to find by pattern
        matches = list(MODELS_DIR.glob(f"*{model_name}*.keras"))
        if matches:
            model_file = matches[0]
            model_name = model_file.stem
    
    # Get version number
    existing_versions = [v for v in registry["models"].keys()]
    version = f"v{len(existing_versions) + 1}"
    
    registry["models"][version] = {
        "name": model_name,
        "accuracy": accuracy,
        "registered_at": datetime.now().isoformat(),
        "notes": notes,
        "file": str(model_file.name) if model_file.exists() else None
    }
    
    save_registry(registry)
    print(f"✅ Registered model as {version}")
    print(f"   Name: {model_name}")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    return version

def list_models():
    """List all registered models."""
    registry = load_registry()
    
    if not registry["models"]:
        print("📭 No models registered yet.")
        print("   Use: python model_registry.py register <model_name> <accuracy>")
        return
    
    print("="*60)
    print("📋 MODEL REGISTRY")
    print("="*60)
    print(f"{'Version':<10} {'Accuracy':<12} {'Date':<12} {'Notes'}")
    print("-"*60)
    
    for version, info in sorted(registry["models"].items()):
        acc = f"{info['accuracy']*100:.1f}%"
        date = info['registered_at'][:10]
        notes = info.get('notes', '')[:30]
        print(f"{version:<10} {acc:<12} {date:<12} {notes}")

def compare_models(version1: str, version2: str):
    """Compare two model versions."""
    registry = load_registry()
    
    if version1 not in registry["models"] or version2 not in registry["models"]:
        print("❌ One or both versions not found.")
        list_models()
        return
    
    m1 = registry["models"][version1]
    m2 = registry["models"][version2]
    
    print("="*50)
    print(f"📊 COMPARING {version1} vs {version2}")
    print("="*50)
    
    print(f"\n{'Metric':<20} {version1:<15} {version2:<15}")
    print("-"*50)
    
    acc1, acc2 = m1['accuracy'], m2['accuracy']
    diff = acc2 - acc1
    diff_str = f"({'+' if diff > 0 else ''}{diff*100:.1f}%)"
    
    print(f"{'Accuracy':<20} {acc1*100:.1f}%{'':<10} {acc2*100:.1f}% {diff_str}")
    print(f"{'Registered':<20} {m1['registered_at'][:10]:<15} {m2['registered_at'][:10]}")
    
    if diff > 0:
        print(f"\n✅ {version2} is better by {diff*100:.1f}%")
    elif diff < 0:
        print(f"\n✅ {version1} is better by {-diff*100:.1f}%")
    else:
        print(f"\n🟰 Both models have the same accuracy")

def auto_register_from_labels():
    """Auto-register models from their labels files."""
    registry = load_registry()
    
    for labels_file in MODELS_DIR.glob("*_labels.json"):
        with open(labels_file) as f:
            data = json.load(f)
        
        model_name = labels_file.stem.replace("_labels", "")
        accuracy = data.get("accuracy", 0)
        
        # Check if already registered
        already_registered = any(
            m["name"] == model_name for m in registry["models"].values()
        )
        
        if not already_registered and accuracy > 0:
            version = register_model(model_name, accuracy)
            print(f"  Auto-registered: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Model Registry Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    subparsers.add_parser("list", help="List all registered models")
    
    # Register command
    reg_parser = subparsers.add_parser("register", help="Register a model")
    reg_parser.add_argument("name", help="Model name or file stem")
    reg_parser.add_argument("accuracy", type=float, help="Model accuracy (0-1)")
    reg_parser.add_argument("--notes", default="", help="Optional notes")
    
    # Compare command
    cmp_parser = subparsers.add_parser("compare", help="Compare two models")
    cmp_parser.add_argument("v1", help="First version (e.g. v1)")
    cmp_parser.add_argument("v2", help="Second version (e.g. v2)")
    
    # Auto-register
    subparsers.add_parser("auto", help="Auto-register from labels files")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_models()
    elif args.command == "register":
        register_model(args.name, args.accuracy, args.notes)
    elif args.command == "compare":
        compare_models(args.v1, args.v2)
    elif args.command == "auto":
        auto_register_from_labels()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
