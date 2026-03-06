"""
Unified Testing Suite Runner
Automates testing with the most powerful configuration in a single command.

Usage:
    # Standard test (single audio file)
    python run_testing_suite.py "audio.mp3"

    # Full power: ensemble + anomaly + low threshold
    python run_testing_suite.py "audio.mp3" --full

    # Test all audio files in a directory
    python run_testing_suite.py "D:\\evidence_audio" --full

    # Custom threshold
    python run_testing_suite.py "audio.mp3" --threshold 0.10
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
            print(f"\n  ✅ {step_name} completed in {elapsed:.1f}s")
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


def gather_audio_files(target_path):
    """If target is a directory, collect all audio files from it. Otherwise return single file."""
    target = Path(target_path)
    if target.is_dir():
        audio_files = []
        for ext in ["*.wav", "*.mp3", "*.ogg", "*.flac", "*.m4a"]:
            audio_files.extend(target.rglob(ext))
        return sorted(audio_files)
    elif target.is_file():
        return [target]
    else:
        print(f"❌ Path not found: {target_path}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Unified Testing Suite — Test audio with maximum power in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_testing_suite.py "audio.mp3"                 # Standard test
  python run_testing_suite.py "audio.mp3" --full          # Ensemble + anomaly
  python run_testing_suite.py "D:\\evidence" --full        # Test entire folder
  python run_testing_suite.py "audio.mp3" --threshold 0.1 # Custom threshold
        """
    )
    parser.add_argument("input", help="Audio file or directory to test")
    parser.add_argument("--full", action="store_true",
                        help="Enable ensemble voting + anomaly detection (maximum accuracy)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Enable ensemble mode (multi-model voting)")
    parser.add_argument("--anomaly", action="store_true",
                        help="Enable anomaly detection (flag unknown sounds)")
    parser.add_argument("--threshold", type=float, default=0.20,
                        help="Confidence threshold (default: 0.20)")
    parser.add_argument("--demucs", action="store_true",
                        help="Also run DEMUCS source separation analysis")
    args = parser.parse_args()

    # --full enables both ensemble and anomaly
    if args.full:
        args.ensemble = True
        args.anomaly = True
        if args.threshold == 0.20:
            args.threshold = 0.15  # Lower default for full mode

    audio_files = gather_audio_files(args.input)
    if not audio_files:
        print("❌ No audio files found.")
        return

    banner("🔍 UNIFIED TESTING SUITE")
    mode_parts = []
    if args.ensemble:
        mode_parts.append("Ensemble")
    if args.anomaly:
        mode_parts.append("Anomaly")
    if args.demucs:
        mode_parts.append("DEMUCS")
    mode_str = " + ".join(mode_parts) if mode_parts else "Standard"
    print(f"  Mode:      {mode_str}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Files:     {len(audio_files)}")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    py = sys.executable
    test_script = str(PROJECT_ROOT / "src" / "testing" / "test_model.py")
    demucs_script = str(PROJECT_ROOT / "src" / "analysis" / "separate_and_analyze.py")
    results = {}
    overall_start = time.time()

    for i, audio_file in enumerate(audio_files):
        file_label = f"[{i+1}/{len(audio_files)}] {audio_file.name}"

        # Build test command
        cmd = [py, test_script, str(audio_file), "--threshold", str(args.threshold)]
        if args.ensemble:
            cmd.append("--ensemble")
        if args.anomaly:
            cmd.append("--anomaly")

        ok = run_step(f"Testing {file_label}", cmd)
        results[audio_file.name] = "✅" if ok else "❌"

        # Optional DEMUCS analysis
        if args.demucs:
            demucs_cmd = [py, demucs_script, str(audio_file), "--threshold", str(args.threshold)]
            ok_d = run_step(f"DEMUCS {file_label}", demucs_cmd)
            results[f"{audio_file.name} (DEMUCS)"] = "✅" if ok_d else "❌"

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    banner("📊 TESTING SUITE SUMMARY")
    for name, status in results.items():
        print(f"  {status} {name}")
    print(f"\n  ⏱️ Total time: {total_time:.1f}s")
    print(f"  🏁 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
