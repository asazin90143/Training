"""
Unified Advanced Tools Runner
Automates post-training analysis, optimization, and deployment in one command.

Usage:
    # Run all advanced tools sequentially (overnight optimization)
    python run_advanced_tools.py --nightly

    # Individual tools
    python run_advanced_tools.py --tune                   # Hyperparameter tuning
    python run_advanced_tools.py --evaluate               # Weakness evaluation
    python run_advanced_tools.py --export                 # TFLite export
    python run_advanced_tools.py --active "D:\\unlabeled"  # Active learning

    # Combine tools
    python run_advanced_tools.py --tune --evaluate --export
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
            stdout=sys.stdout, stderr=sys.stderr
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"\n  ✅ {step_name} completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"\n  ⚠️ {step_name} exited with code {result.returncode}")
            return False
    except FileNotFoundError:
        print(f"  ❌ Could not find: {cmd[0]}")
        return False
    except KeyboardInterrupt:
        print(f"\n  ⛔ {step_name} interrupted by user.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Unified Advanced Tools — Post-training optimization in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_advanced_tools.py --nightly                  # Full overnight optimization
  python run_advanced_tools.py --tune --max_trials 100    # Just hyperparameter tuning
  python run_advanced_tools.py --evaluate --export        # Evaluate + export
  python run_advanced_tools.py --active "D:\\unlabeled"    # Active learning scan
        """
    )
    parser.add_argument("--nightly", action="store_true",
                        help="Run ALL tools sequentially (tune → student → evaluate → export)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning (KerasTuner)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run weakness/confusion evaluation")
    parser.add_argument("--export", action="store_true",
                        help="Export all models to TFLite")
    parser.add_argument("--active", type=str, default=None, metavar="DIR",
                        help="Run active learning on the given unlabeled audio directory")
    parser.add_argument("--student", action="store_true",
                        help="Train a student model via knowledge distillation")
    parser.add_argument("--max_trials", type=int, default=50,
                        help="Max trials for hyperparameter tuning (default: 50)")
    parser.add_argument("--temperature", type=float, default=3.0,
                        help="Temperature for knowledge distillation (default: 3.0)")
    parser.add_argument("--quantize", action="store_true",
                        help="Apply quantization when exporting to TFLite")
    args = parser.parse_args()

    # --nightly enables all major tools
    if args.nightly:
        args.tune = True
        args.student = True
        args.evaluate = True
        args.export = True

    # Validate at least one tool is selected
    any_selected = args.tune or args.evaluate or args.export or args.active or args.student
    if not any_selected:
        parser.print_help()
        print("\n❌ Please select at least one tool (--tune, --evaluate, --export, --active, --student, or --nightly)")
        return

    banner("🛠️ UNIFIED ADVANCED TOOLS RUNNER")
    tools = []
    if args.tune:
        tools.append("Hyperparameter Tuning")
    if args.student:
        tools.append("Knowledge Distillation")
    if args.evaluate:
        tools.append("Weakness Evaluation")
    if args.export:
        tools.append("TFLite Export")
    if args.active:
        tools.append(f"Active Learning ({args.active})")
    print(f"  Tools:    {', '.join(tools)}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    py = sys.executable
    results = {}
    overall_start = time.time()

    # ── 1. Hyperparameter Tuning ───────────────────────────────────
    if args.tune:
        ok = run_step("Hyperparameter Tuning (KerasTuner)", [
            py, str(PROJECT_ROOT / "src" / "training" / "tune_hyperparameters.py"),
            "--max_trials", str(args.max_trials)
        ])
        results["Hyperparameter Tuning"] = ok

    # ── 2. Knowledge Distillation ──────────────────────────────────
    if args.student:
        ok = run_step("Knowledge Distillation (Student Model)", [
            py, str(PROJECT_ROOT / "src" / "training" / "train_student_model.py"),
            "--temperature", str(args.temperature)
        ])
        results["Student Model"] = ok

    # ── 3. Weakness Evaluation ─────────────────────────────────────
    if args.evaluate:
        ok = run_step("Hard Negative Mining (Weakness Report)", [
            py, str(PROJECT_ROOT / "src" / "analysis" / "evaluate_weaknesses.py")
        ])
        results["Weakness Evaluation"] = ok

    # ── 4. Active Learning ─────────────────────────────────────────
    if args.active:
        ok = run_step("Active Learning Scan", [
            py, str(PROJECT_ROOT / "src" / "analysis" / "active_learner.py"),
            args.active
        ])
        results["Active Learning"] = ok

    # ── 5. TFLite Export ───────────────────────────────────────────
    if args.export:
        cmd = [py, str(PROJECT_ROOT / "src" / "utils" / "export_to_tflite.py"), "--all"]
        if args.quantize:
            cmd.append("--quantize")
        ok = run_step("TFLite Export (All Models)", cmd)
        results["TFLite Export"] = ok

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    banner("📊 ADVANCED TOOLS SUMMARY")
    for step, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {step}")

    print(f"\n  ⏱️ Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"  🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
