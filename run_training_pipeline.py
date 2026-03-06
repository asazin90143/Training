"""
Unified Training Pipeline Runner
Automates the full training workflow: preprocessing → training (all backbones).

Usage:
    # Quick: Preprocess + train YAMNet only
    python run_training_pipeline.py

    # Full overnight: Preprocess → YAMNet → VGGish → Spectrogram → BEATs → Student
    python run_training_pipeline.py --full

    # Skip preprocessing (already done)
    python run_training_pipeline.py --skip_preprocess --full

    # Custom epochs
    python run_training_pipeline.py --full --epochs 80
"""

import sys
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent

def banner(text):
    width = 60
    print(f"\n{'═'*width}")
    print(f"  {text}")
    print(f"{'═'*width}")

def run_step(step_name, cmd, cwd=None):
    """Run a subprocess, stream output in real-time, and return success status."""
    banner(f"▶ {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}\n")

    start = time.time()
    try:
        result = subprocess.run(
            cmd, cwd=cwd or str(PROJECT_ROOT),
            # Stream output in real-time by inheriting stdout/stderr
            stdout=sys.stdout, stderr=sys.stderr
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"\n  ✅ {step_name} completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"\n  ⚠️ {step_name} exited with code {result.returncode} ({elapsed/60:.1f} min)")
            return False
    except FileNotFoundError:
        print(f"  ❌ Could not find: {cmd[0]}")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⛔ {step_name} interrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Pipeline — Preprocess + Train in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_pipeline.py                    # Preprocess + YAMNet training
  python run_training_pipeline.py --full             # Preprocess + ALL models
  python run_training_pipeline.py --skip_preprocess  # Skip preprocess, train YAMNet
  python run_training_pipeline.py --full --epochs 80 # All models, 80 epochs each
        """
    )
    parser.add_argument("--full", action="store_true",
                        help="Train ALL backbones (YAMNet, VGGish, Spectrogram, BEATs, Student)")
    parser.add_argument("--skip_preprocess", action="store_true",
                        help="Skip Step 1 (preprocessing). Use if you already preprocessed.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs per training run (default: 100)")
    parser.add_argument("--finetune", action="store_true",
                        help="Fine-tune YAMNet backbone layers")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training (default: 32)")
    args = parser.parse_args()

    banner("🚀 UNIFIED TRAINING PIPELINE")
    print(f"  Mode:       {'FULL (all backbones)' if args.full else 'Standard (YAMNet only)'}")
    print(f"  Preprocess: {'SKIP' if args.skip_preprocess else 'YES'}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Fine-tune:  {'YES' if args.finetune else 'NO'}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    py = sys.executable  # Use the same Python interpreter
    results = {}
    overall_start = time.time()

    # ── Step 1: Preprocessing ──────────────────────────────────────────
    if not args.skip_preprocess:
        ok = run_step("Step 1: Preprocess Audio", [
            py, str(PROJECT_ROOT / "src" / "preprocessing" / "preprocess_audio.py")
        ])
        results["Preprocess"] = ok
        if not ok:
            print("\n⛔ Preprocessing failed. Fix errors above before training.")
            return
    else:
        print("\n  ⏭️ Skipping preprocessing (--skip_preprocess)")
        results["Preprocess"] = "SKIPPED"

    # ── Step 2: Core Training (YAMNet / VGGish) ──────────────────────
    if args.full:
        # 2a: YAMNet + VGGish via --all_models flag
        cmd = [py, str(PROJECT_ROOT / "src" / "training" / "train_forensic_model.py"),
               "--all_models", "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]
        if args.finetune:
            cmd.append("--finetune")
        ok = run_step("Step 2a: Train Core Models (YAMNet + VGGish)", cmd)
        results["Core Training"] = ok

        # 2b: Spectrogram (ResNet50)
        ok = run_step("Step 2b: Train Spectrogram Model (ResNet50)", [
            py, str(PROJECT_ROOT / "src" / "training" / "train_spectrogram_model.py"),
            "--epochs", str(args.epochs), "--architecture", "resnet50"
        ])
        results["Spectrogram (ResNet50)"] = ok

        # 2c: BEATs
        ok = run_step("Step 2c: Train BEATs Model", [
            py, str(PROJECT_ROOT / "src" / "training" / "train_beats_model.py"),
            "--epochs", str(min(args.epochs, 50))
        ])
        results["BEATs"] = ok

        # 2d: Knowledge Distillation (Student)
        ok = run_step("Step 2d: Train Student Model (Knowledge Distillation)", [
            py, str(PROJECT_ROOT / "src" / "training" / "train_student_model.py"),
            "--epochs", str(args.epochs), "--temperature", "3.0"
        ])
        results["Student Model"] = ok
    else:
        # Standard: YAMNet only
        cmd = [py, str(PROJECT_ROOT / "src" / "training" / "train_forensic_model.py"),
               "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]
        if args.finetune:
            cmd.append("--finetune")
        ok = run_step("Step 2: Train YAMNet Model", cmd)
        results["YAMNet Training"] = ok

    # ── Summary ────────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    banner("📊 TRAINING PIPELINE SUMMARY")
    for step, status in results.items():
        icon = "✅" if status is True else ("⏭️" if status == "SKIPPED" else "❌")
        print(f"  {icon} {step}")

    print(f"\n  ⏱️ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
