"""
Speaker Diarization & Voice Separation Pipeline (Phase 1)
Uses Pyannote.audio for speaker diarization and SpeechBrain SepFormer for voice separation.

Detects how many speakers are present, creates a timeline of who spoke when,
and isolates each speaker's voice into separate .wav files.

Usage:
    python src/analysis/speaker_diarization.py "audio.wav"
    python src/analysis/speaker_diarization.py "audio.wav" --output_dir "./separated"
    python src/analysis/speaker_diarization.py "audio.wav" --max_speakers 4
    python src/analysis/speaker_diarization.py "audio.wav" --no_separate   # Diarization only

Requirements:
    pip install pyannote.audio speechbrain torchaudio
"""

import os
import sys
import json
import argparse
import warnings
import time
from pathlib import Path
from datetime import datetime

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_paths
PATHS = get_paths()


# ─── HuggingFace Token ───────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ─── Lazy Imports (heavy libraries) ──────────────────────────────────────

def _import_torch():
    """Lazy import for torch."""
    import torch
    return torch


def _import_torchaudio():
    """Lazy import for torchaudio."""
    import torchaudio
    return torchaudio


def _load_diarization_pipeline(device="cpu"):
    """Load Pyannote diarization pipeline."""
    from pyannote.audio import Pipeline

    print("   📡 Loading Pyannote diarization pipeline...")
    print("   ℹ️  First run will download model weights (~300 MB)")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )

    torch = _import_torch()
    pipeline.to(torch.device(device))
    print(f"   ✅ Pyannote loaded on {device}")
    return pipeline


def _load_separation_model(device="cpu"):
    """Load SpeechBrain SepFormer separation model."""
    from speechbrain.inference.separation import SepformerSeparation

    print("   📡 Loading SpeechBrain SepFormer...")
    print("   ℹ️  First run will download model weights (~100 MB)")

    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=str(PATHS["models_root"] / "separation" / "sepformer_cache"),
        run_opts={"device": device}
    )
    print(f"   ✅ SepFormer loaded on {device}")
    return model


# ─── Core Functions ──────────────────────────────────────────────────────

def run_diarization(audio_path, pipeline, min_speakers=None, max_speakers=None):
    """
    Run speaker diarization on an audio file.

    Returns:
        diarization: pyannote Annotation object
        speaker_timeline: list of dicts with speaker, start, end
    """
    print(f"\n🎙️ Running diarization on: {Path(audio_path).name}")

    # Build kwargs for pipeline
    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    start = time.time()
    diarization = pipeline(str(audio_path), **kwargs)
    elapsed = time.time() - start

    # Extract timeline
    speaker_timeline = []
    speakers = set()

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
        speaker_timeline.append({
            "speaker": speaker,
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "duration": round(turn.end - turn.start, 3)
        })

    num_speakers = len(speakers)
    total_speech = sum(s["duration"] for s in speaker_timeline)

    print(f"   🎯 Detected {num_speakers} speaker(s)")
    print(f"   ⏱️  Total speech: {total_speech:.1f}s")
    print(f"   ⚡ Diarization took: {elapsed:.1f}s")

    # Print timeline summary
    print(f"\n   📋 Speaker Timeline:")
    for speaker in sorted(speakers):
        segments = [s for s in speaker_timeline if s["speaker"] == speaker]
        total = sum(s["duration"] for s in segments)
        print(f"      {speaker}: {len(segments)} segments, {total:.1f}s total")

    return diarization, speaker_timeline


def run_separation(audio_path, sep_model, output_dir, timeline=None):
    """
    Run voice separation using SepFormer.

    Args:
        audio_path: Path to the audio file
        sep_model: Loaded SepFormer model
        output_dir: Directory to save separated stems
        timeline: Optional diarization timeline for labeling

    Returns:
        list of output file paths
    """
    torchaudio = _import_torchaudio()
    torch = _import_torch()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_name = Path(audio_path).stem

    print(f"\n🔀 Running voice separation on: {Path(audio_path).name}")
    start = time.time()

    # SepFormer separation
    est_sources = sep_model.separate_file(path=str(audio_path))
    elapsed = time.time() - start

    num_sources = est_sources.shape[-1]
    print(f"   🎯 Separated into {num_sources} source(s)")
    print(f"   ⚡ Separation took: {elapsed:.1f}s")

    # Save each separated source
    output_files = []
    for i in range(num_sources):
        source = est_sources[:, :, i]

        # Normalize to prevent clipping
        max_val = source.abs().max()
        if max_val > 0:
            source = source / max_val * 0.95

        # Determine speaker label
        if timeline:
            # Find the speaker with the most speech in this source
            speaker_label = f"speaker_{i + 1}"
        else:
            speaker_label = f"speaker_{i + 1}"

        output_path = output_dir / f"{audio_name}_{speaker_label}.wav"
        torchaudio.save(str(output_path), source.cpu(), 8000)
        output_files.append(output_path)
        print(f"   💾 Saved: {output_path.name}")

    return output_files


def save_diarization_report(timeline, output_dir, audio_name):
    """Save diarization results as JSON and RTTM."""
    output_dir = Path(output_dir)

    # Save JSON report
    json_path = output_dir / f"{audio_name}_diarization.json"
    report = {
        "audio_file": audio_name,
        "timestamp": datetime.now().isoformat(),
        "num_speakers": len(set(s["speaker"] for s in timeline)),
        "total_speech_duration": sum(s["duration"] for s in timeline),
        "timeline": timeline
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"   📄 JSON report: {json_path.name}")

    # Save RTTM file (standard diarization format)
    rttm_path = output_dir / f"{audio_name}_diarization.rttm"
    with open(rttm_path, "w") as f:
        for segment in timeline:
            # RTTM format: SPEAKER file 1 start duration <NA> <NA> speaker <NA> <NA>
            f.write(
                f"SPEAKER {audio_name} 1 {segment['start']:.3f} "
                f"{segment['duration']:.3f} <NA> <NA> "
                f"{segment['speaker']} <NA> <NA>\n"
            )
    print(f"   📄 RTTM file: {rttm_path.name}")

    return json_path, rttm_path


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Speaker Diarization & Voice Separation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/analysis/speaker_diarization.py "audio.wav"
  python src/analysis/speaker_diarization.py "audio.wav" --output_dir "./separated"
  python src/analysis/speaker_diarization.py "audio.wav" --max_speakers 3
  python src/analysis/speaker_diarization.py "audio.wav" --no_separate
  python src/analysis/speaker_diarization.py "audio.wav" --device cuda
        """
    )
    parser.add_argument("file", help="Audio file to process")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save separated audio (default: ./diarization_output)")
    parser.add_argument("--min_speakers", type=int, default=None,
                        help="Minimum expected number of speakers")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Maximum expected number of speakers")
    parser.add_argument("--no_separate", action="store_true",
                        help="Only run diarization (skip voice separation)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to run on (default: cpu)")
    args = parser.parse_args()

    # Validate input file
    audio_path = Path(args.file)
    if not audio_path.exists():
        print(f"❌ File not found: {audio_path}")
        return

    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path("diarization_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🎙️ SPEAKER DIARIZATION & VOICE SEPARATION")
    print("=" * 60)
    print(f"  Input:       {audio_path}")
    print(f"  Output:      {output_dir}")
    print(f"  Device:      {args.device}")
    print(f"  Separation:  {'Disabled' if args.no_separate else 'Enabled'}")
    if args.min_speakers:
        print(f"  Min speakers: {args.min_speakers}")
    if args.max_speakers:
        print(f"  Max speakers: {args.max_speakers}")

    overall_start = time.time()

    # ── Step 1: Speaker Diarization ──────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  📡 STEP 1: Speaker Diarization (Pyannote)")
    print(f"{'─' * 60}")

    try:
        pipeline = _load_diarization_pipeline(device=args.device)
    except Exception as e:
        print(f"\n❌ Failed to load Pyannote pipeline: {e}")
        print("\n💡 To fix this, run:")
        print("   pip install pyannote.audio")
        print("   Then make sure you have accepted the model terms at:")
        print("   https://huggingface.co/pyannote/speaker-diarization-3.1")
        return

    diarization, timeline = run_diarization(
        audio_path, pipeline,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers
    )

    # Save diarization report
    print(f"\n   💾 Saving diarization report...")
    save_diarization_report(timeline, output_dir, audio_path.stem)

    if len(timeline) == 0:
        print("\n⚠️  No speech detected in this audio file.")
        return

    # ── Step 2: Voice Separation ────────────────────────────────────
    if not args.no_separate:
        print(f"\n{'─' * 60}")
        print("  🔀 STEP 2: Voice Separation (SpeechBrain SepFormer)")
        print(f"{'─' * 60}")

        try:
            sep_model = _load_separation_model(device=args.device)
        except Exception as e:
            print(f"\n❌ Failed to load SepFormer: {e}")
            print("\n💡 To fix this, run:")
            print("   pip install speechbrain torchaudio")
            return

        output_files = run_separation(
            audio_path, sep_model, output_dir, timeline
        )
    else:
        output_files = []

    # ── Summary ─────────────────────────────────────────────────────
    total_time = time.time() - overall_start
    num_speakers = len(set(s["speaker"] for s in timeline))

    print(f"\n{'=' * 60}")
    print("📊 RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  🎯 Speakers detected:  {num_speakers}")
    print(f"  📋 Speech segments:    {len(timeline)}")

    for speaker in sorted(set(s["speaker"] for s in timeline)):
        segments = [s for s in timeline if s["speaker"] == speaker]
        total = sum(s["duration"] for s in segments)
        first = segments[0]["start"]
        last = segments[-1]["end"]
        print(f"     {speaker}: {total:.1f}s speech ({first:.1f}s - {last:.1f}s)")

    if output_files:
        print(f"\n  🎵 Separated audio files:")
        for f in output_files:
            print(f"     💾 {f.name}")

    print(f"\n  📁 All outputs saved to: {output_dir}")
    print(f"  ⏱️  Total time: {total_time:.1f}s")
    print(f"  🏁 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
