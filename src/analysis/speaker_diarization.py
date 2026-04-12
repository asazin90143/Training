"""
Speaker Diarization & Voice Separation Pipeline (Phase 1)

Uses custom forensic-trained models when available:
  - Fine-tuned PyAnnote segmentation-3.0 for speaker diarization
  - Student Separator (distilled from SepFormer + DANN) for fast voice isolation

Falls back to pre-trained HuggingFace/SpeechBrain models if custom checkpoints
are not found.

Detects how many speakers are present, creates a timeline of who spoke when,
and isolates each speaker's voice into separate .wav files.

Usage:
    python src/analysis/speaker_diarization.py "audio.wav"
    python src/analysis/speaker_diarization.py "audio.wav" --output_dir "./separated"
    python src/analysis/speaker_diarization.py "audio.wav" --max_speakers 4
    python src/analysis/speaker_diarization.py "audio.wav" --no_separate   # Diarization only
    python src/analysis/speaker_diarization.py "audio.wav" --use_pretrained  # Force generic models

Requirements:
    pip install pyannote.audio speechbrain python-dotenv
"""

import os
import sys
import json
import wave
import argparse
import warnings
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load .env for HF_TOKEN
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Resolve project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_paths
PATHS = get_paths()

# Paths to custom-trained models
CUSTOM_PYANNOTE_DIR = PATHS["models_root"] / "separation" / "pyannote_finetune"
CUSTOM_STUDENT_PATH = PATHS["models_root"] / "separation" / "student_separator" / "student_separator.pt"
CUSTOM_SEPFORMER_DIR = PATHS["models_root"] / "separation" / "sepformer_dann"


# ─── HuggingFace Token ───────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ─── Lazy Imports (heavy libraries) ──────────────────────────────────────

def _import_torch():
    """Lazy import for torch."""
    import torch
    return torch


def _load_wav_native(filepath, target_sr=16000):
    """Load WAV using Python's built-in wave module (no torchaudio dependency)."""
    torch = _import_torch()
    with wave.open(str(filepath), 'rb') as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)
    if sampwidth == 2:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # Convert stereo to mono
    if n_channels > 1:
        arr = arr.reshape(-1, n_channels).mean(axis=1)
    return torch.from_numpy(arr).unsqueeze(0), sr  # [1, samples]


def _load_diarization_pipeline(device="cpu", use_pretrained=False):
    """Load Pyannote diarization pipeline with custom fine-tuned model if available."""
    from pyannote.audio import Pipeline
    torch = _import_torch()

    # Check for custom fine-tuned PyAnnote checkpoint
    custom_ckpt = None
    if not use_pretrained:
        ckpts = sorted(CUSTOM_PYANNOTE_DIR.glob("checkpoint_epoch_*.pt")) if CUSTOM_PYANNOTE_DIR.exists() else []
        if ckpts:
            custom_ckpt = ckpts[-1]  # Use the latest checkpoint

    if custom_ckpt:
        print(f"   📡 Loading custom fine-tuned PyAnnote (forensic)...")
        print(f"   📂 Checkpoint: {custom_ckpt.name}")
        # Load the base pipeline, then override the segmentation model weights
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        # Override segmentation model with our fine-tuned weights
        try:
            from pyannote.audio import Model
            seg_model = Model.from_pretrained("pyannote/segmentation-3.0", token=HF_TOKEN)
            state = torch.load(str(custom_ckpt), map_location="cpu")
            seg_model.load_state_dict(state["model_state"])
            pipeline._segmentation.model = seg_model
            print(f"   ✅ Custom forensic PyAnnote loaded (fine-tuned segmentation)")
        except Exception as e:
            print(f"   ⚠️  Could not inject custom weights: {e}")
            print(f"   ℹ️  Falling back to pre-trained segmentation")
    else:
        print("   📡 Loading pre-trained Pyannote diarization pipeline...")
        if not use_pretrained and CUSTOM_PYANNOTE_DIR.exists():
            print("   ℹ️  No custom PyAnnote checkpoint found, using pre-trained")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        print(f"   ✅ Pre-trained Pyannote loaded")

    pipeline.to(torch.device(device))
    return pipeline


def _load_separation_model(device="cpu", use_pretrained=False):
    """
    Load voice separation model.
    Priority: Custom Student Separator > Pre-trained SepFormer
    """
    torch = _import_torch()
    nn = _import_torch_nn()

    # Try loading custom Student Separator (fast, distilled model)
    if not use_pretrained and CUSTOM_STUDENT_PATH.exists():
        print(f"   📡 Loading custom Student Separator (forensic-distilled)...")
        print(f"   📂 Model: {CUSTOM_STUDENT_PATH.name}")

        # Reconstruct the StudentSeparator architecture
        class StudentSeparator(nn.Module):
            """Tiny 10-layer separator distilled from Teacher SepFormer."""
            def __init__(self, input_dim=256, hidden_dim=128, num_sources=3):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                )
                self.separator = nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                )
                self.decoder = nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, 1),
                    nn.ReLU(),
                    nn.Conv1d(hidden_dim, input_dim * num_sources, 1),
                )
                self.num_sources = num_sources
                self.input_dim = input_dim

            def forward(self, x):
                encoded = self.encoder(x)
                separated = self.separator(encoded)
                decoded = self.decoder(separated)
                b, _, t = decoded.size()
                return decoded.view(b, self.num_sources, self.input_dim, t)

        student = StudentSeparator(input_dim=256, hidden_dim=128, num_sources=3)
        state = torch.load(str(CUSTOM_STUDENT_PATH), map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            student.load_state_dict(state["model_state"])
        else:
            student.load_state_dict(state)
        student.eval()
        student.to(device)

        total_params = sum(p.numel() for p in student.parameters())
        print(f"   ✅ Student Separator loaded: {total_params / 1e6:.2f}M params (5x faster than Teacher)")
        return student, "student"

    # Fallback: Load pre-trained SpeechBrain SepFormer
    from speechbrain.inference.separation import SepformerSeparation
    print("   📡 Loading pre-trained SpeechBrain SepFormer...")
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=str(PATHS["models_root"] / "separation" / "sepformer_cache"),
        run_opts={"device": device}
    )
    print(f"   ✅ Pre-trained SepFormer loaded on {device}")
    return model, "sepformer"


def _import_torch_nn():
    """Lazy import for torch.nn."""
    import torch.nn as nn
    return nn


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


def run_separation(audio_path, sep_model, model_type, output_dir, timeline=None):
    """
    Run voice separation using either the Student Separator or pre-trained SepFormer.

    Args:
        audio_path: Path to the audio file
        sep_model: Loaded model (Student or SepFormer)
        model_type: "student" or "sepformer"
        output_dir: Directory to save separated stems
        timeline: Optional diarization timeline for labeling

    Returns:
        list of output file paths
    """
    torch = _import_torch()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_name = Path(audio_path).stem

    print(f"\n🔀 Running voice separation on: {Path(audio_path).name}")
    print(f"   🧠 Model: {'Custom Student Separator (forensic)' if model_type == 'student' else 'Pre-trained SepFormer'}")
    start = time.time()

    if model_type == "student":
        # ── Student Separator path ──
        # Load audio via native wav loader
        waveform, sr = _load_wav_native(audio_path)

        # The Student Separator works on SepFormer encoder features [B, 256, T]
        # We need to load the SepFormer encoder to extract features first
        from speechbrain.inference.separation import SepformerSeparation
        teacher = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wsj03mix",
            savedir=str(PATHS["models_root"] / "separation" / "sepformer_cache"),
            run_opts={"device": "cpu"}
        )

        with torch.no_grad():
            # Encode through teacher's encoder
            encoded = teacher.mods.encoder(waveform)  # [B, C, T]
            # Run through student separator
            student_out = sep_model(encoded)  # [B, 3, 256, T]
            # Decode each source back to waveform through teacher's decoder
            num_sources = student_out.shape[1]
            est_sources_list = []
            for src_idx in range(num_sources):
                source_features = student_out[:, src_idx, :, :]  # [B, 256, T]
                decoded = teacher.mods.decoder(source_features)  # [B, T']
                est_sources_list.append(decoded)
            est_sources = torch.stack(est_sources_list, dim=-1)  # [B, T', num_sources]

    else:
        # ── Pre-trained SepFormer path ──
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

        speaker_label = f"speaker_{i + 1}"
        output_path = output_dir / f"{audio_name}_{speaker_label}.wav"

        # Save WAV using native wave module (no torchaudio dependency)
        audio_np = (source.squeeze().cpu().numpy() * 32767).astype(np.int16)
        with wave.open(str(output_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(audio_np.tobytes())

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
    parser.add_argument("--use_pretrained", action="store_true",
                        help="Force use of pre-trained models instead of custom forensic models")
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
    print(f"  Model mode:  {'Pre-trained (generic)' if args.use_pretrained else 'Custom forensic (if available)'}")
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
        pipeline = _load_diarization_pipeline(device=args.device, use_pretrained=args.use_pretrained)
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
            sep_model, model_type = _load_separation_model(device=args.device, use_pretrained=args.use_pretrained)
        except Exception as e:
            print(f"\n❌ Failed to load separation model: {e}")
            print("\n💡 To fix this, run:")
            print("   pip install speechbrain python-dotenv")
            return

        output_files = run_separation(
            audio_path, sep_model, model_type, output_dir, timeline
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
