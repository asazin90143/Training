"""
Speaker Diarization & Separation — Dataset Download Script (Phase 0)
Downloads all required datasets for the voice separation pipeline.

Usage:
    python download_separation_datasets.py                     # Download all datasets
    python download_separation_datasets.py --only librimix     # Download a specific dataset
    python download_separation_datasets.py --list              # List available datasets
    python download_separation_datasets.py --dest "E:\\data"    # Custom destination

Default destination: D:\\separation_dataset
"""

import os
import sys
import argparse
import subprocess
import tarfile
import zipfile
import shutil
import hashlib
import time
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


# ─── Default destination (completely separate from classification data) ───
DEFAULT_DEST = r"D:\separation_dataset"


# ─── Dataset Registry ────────────────────────────────────────────────────
# Each entry: { name, description, size_hint, auto, download_fn }
DATASET_REGISTRY = {}


def register_dataset(name, description, size_hint, auto=True):
    """Decorator to register a dataset download function."""
    def decorator(fn):
        DATASET_REGISTRY[name] = {
            "name": name,
            "description": description,
            "size_hint": size_hint,
            "auto": auto,
            "download_fn": fn,
        }
        return fn
    return decorator


# ─── Utility Functions ────────────────────────────────────────────────────

def progress_hook(block_num, block_size, total_size):
    """Display download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"\r   [{bar}] {pct:5.1f}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)
    else:
        mb_done = downloaded / (1024 * 1024)
        print(f"\r   Downloaded: {mb_done:.0f} MB...", end="", flush=True)


def safe_download(url, dest_path, description="file"):
    """Download a file with progress bar and retry logic."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"   ⏭️  Already exists: {dest_path.name}")
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"   ⬇️  Downloading {description}...")
    print(f"   URL: {url}")

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            urlretrieve(url, str(dest_path), reporthook=progress_hook)
            print()  # newline after progress bar
            return True
        except (URLError, ConnectionError, OSError) as e:
            print(f"\n   ⚠️  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 5 * attempt
                print(f"   Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"   ❌ Failed to download: {description}")
                return False


def extract_archive(archive_path, dest_dir, remove_after=False):
    """Extract a tar.gz or zip archive."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"   📦 Extracting {archive_path.name}...")
    try:
        if archive_path.suffix == '.zip' or str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(str(archive_path), 'r') as zf:
                zf.extractall(str(dest_dir))
        elif '.tar' in archive_path.name:
            with tarfile.open(str(archive_path), 'r:*') as tf:
                tf.extractall(str(dest_dir))
        else:
            print(f"   ⚠️  Unknown archive format: {archive_path.name}")
            return False

        print(f"   ✅ Extracted to: {dest_dir}")
        if remove_after:
            archive_path.unlink()
            print(f"   🗑️  Removed archive: {archive_path.name}")
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def run_shell(cmd, cwd=None, description="command"):
    """Run a shell command with real-time output."""
    print(f"   🔧 Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, cwd=cwd, stdout=sys.stdout, stderr=sys.stderr)
        return result.returncode == 0
    except FileNotFoundError:
        print(f"   ❌ Command not found: {cmd[0]}")
        return False
    except Exception as e:
        print(f"   ❌ Error running {description}: {e}")
        return False


# ─── Dataset Download Functions ───────────────────────────────────────────

@register_dataset("librispeech", "Clean speech corpus (1000+ hours)", "~60 GB")
def download_librispeech(dest_root):
    """Download LibriSpeech train-clean-100 subset (6.3 GB)."""
    dest = Path(dest_root) / "librispeech"
    dest.mkdir(parents=True, exist_ok=True)

    # Check for partial download (train-clean-100 should have ~28,000 files)
    subset_dir = dest / "LibriSpeech" / "train-clean-100"
    if subset_dir.exists():
        file_count = len(list(subset_dir.rglob("*.flac")))
        if file_count > 25000:
            print(f"   Skip: LibriSpeech train-clean-100 already present ({file_count} files).")
            return True
        else:
            print(f"   Detect: Partial LibriSpeech found ({file_count} files).")
            # Rename as a workaround for WinError 5 locks
            old_path = dest.parent / f"librispeech_old_{int(time.time())}"
            try:
                dest.rename(old_path)
                print(f"   Renamed partial folder to {old_path.name}")
            except Exception:
                # If rename also fails, we just try to delete what we can and hope for the best
                shutil.rmtree(dest, ignore_errors=True)
            
            dest.mkdir(parents=True, exist_ok=True)

    # Download the train-clean-100 subset (6.3 GB)
    url = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
    archive = dest / "train-clean-100.tar.gz"

    print(f"   ℹ️  Downloading LibriSpeech train-clean-100 (~6.3 GB compressed).")
    ok = safe_download(url, archive, "LibriSpeech train-clean-100")
    if ok:
        ok = extract_archive(archive, dest, remove_after=True)
    return ok


@register_dataset("wham", "Urban noise recordings for mixing", "~17 GB")
def download_wham(dest_root):
    """Download WHAM! noise dataset."""
    dest = Path(dest_root) / "wham_noise"
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()) if dest.exists() else False:
        existing = list(dest.rglob("*.wav"))
        if len(existing) > 10:
            print(f"   ⏭️  WHAM! noise already present ({len(existing)} wav files).")
            return True

    # The WHAM! noise dataset
    url = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip"
    archive = dest / "wham_noise.zip"

    print("   ℹ️  WHAM! dataset (~17 GB). This is a large download.")
    print("   ℹ️  If the direct URL fails, download manually from: https://wham.whisper.ai")
    ok = safe_download(url, archive, "WHAM! noise dataset")
    if ok:
        ok = extract_archive(archive, dest, remove_after=True)
    return ok


@register_dataset("musan", "Music, Speech, and Noise corpus", "~11 GB")
def download_musan(dest_root):
    """Download MUSAN noise corpus."""
    dest = Path(dest_root) / "musan"
    dest.mkdir(parents=True, exist_ok=True)

    if (dest / "noise").exists() or (dest / "music").exists():
        print("   ⏭️  MUSAN already extracted.")
        return True

    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    archive = dest / "musan.tar.gz"

    ok = safe_download(url, archive, "MUSAN corpus (~11 GB)")
    if ok:
        ok = extract_archive(archive, dest, remove_after=True)
    return ok


@register_dataset("ami", "Multi-person meeting recordings", "~100 GB")
def download_ami(dest_root):
    """Download AMI Meeting Corpus headset mix audio."""
    dest = Path(dest_root) / "ami"
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.rglob("*.wav")):
        print("   ⏭️  AMI corpus already has audio files.")
        return True

    # AMI headset mix (the most useful single subset for diarization)
    print("   ℹ️  AMI Meeting Corpus is very large (~100 GB for full corpus).")
    print("   ℹ️  Downloading the headset-mix subset for diarization training.")
    print("   ℹ️  For the full corpus, visit: https://groups.inf.ed.ac.uk/ami/corpus/")

    # Use the Edinburgh mirror for individual meeting recordings
    # We download a manifest that lists all available files
    url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav"
    sample = dest / "ES2002a.Mix-Headset.wav"

    ok = safe_download(url, sample, "AMI sample meeting (ES2002a)")
    if ok:
        print("   ✅ Downloaded AMI sample meeting.")
        print("   ℹ️  For the full corpus, use the AMI download scripts from:")
        print("   ℹ️  https://github.com/pyannote/AMI-diarization-setup")
    return ok


@register_dataset("dns", "Microsoft Deep Noise Suppression dataset", "~500 GB+")
def download_dns(dest_root):
    """Clone DNS Challenge repository (scripts + small subset)."""
    dest = Path(dest_root) / "dns_challenge"

    if (dest / "README.md").exists():
        print("   ⏭️  DNS Challenge repo already cloned.")
        return True

    dest.mkdir(parents=True, exist_ok=True)

    print("   ℹ️  DNS Challenge is massive (~500 GB+ for full data).")
    print("   ℹ️  Cloning the repo with download scripts first...")

    # Clone just the repo (not the full dataset) — user can run their download scripts later
    ok = run_shell([
        "git", "clone", "--depth", "1",
        "https://github.com/microsoft/DNS-Challenge.git",
        str(dest)
    ], description="Clone DNS Challenge repo")

    if ok:
        print("   ✅ DNS Challenge repo cloned.")
        print("   ℹ️  To download the full dataset, run the scripts inside:")
        print(f"   ℹ️  {dest}")
    return ok


@register_dataset("librimix", "2-3 speaker mixtures for separation", "~430 GB (2-spk)")
def download_librimix(dest_root):
    """Clone LibriMix generation scripts."""
    dest = Path(dest_root) / "librimix"

    if (dest / "generate_librimix.sh").exists():
        print("   ⏭️  LibriMix repo already cloned.")
        return True

    dest.mkdir(parents=True, exist_ok=True)

    print("   ℹ️  LibriMix generates mixtures from LibriSpeech + WHAM! noise.")
    print("   ℹ️  Cloning the generation scripts...")

    ok = run_shell([
        "git", "clone", "--depth", "1",
        "https://github.com/JorisCos/LibriMix.git",
        str(dest)
    ], description="Clone LibriMix repo")

    if ok:
        print("   ✅ LibriMix repo cloned.")
        print("   ℹ️  To generate mixtures, run:")
        print(f"   ℹ️  cd {dest}")
        print("   ℹ️  bash generate_librimix.sh <librispeech_dir>")
    return ok


@register_dataset("whamr", "WHAM! + room reverberation", "~35 GB")
def download_whamr(dest_root):
    """Download WHAMR! (WHAM! with reverberation)."""
    dest = Path(dest_root) / "whamr"
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.rglob("*.wav")):
        print("   ⏭️  WHAMR! already has audio files.")
        return True

    print("   ℹ️  WHAMR! adds simulated room reverberation to WHAM! noise (~35 GB).")
    print("   ℹ️  If the direct download fails, visit: https://wham.whisper.ai")

    url = "https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/whamr_scripts.tar.gz"
    archive = dest / "whamr_scripts.tar.gz"

    ok = safe_download(url, archive, "WHAMR! scripts")
    if ok:
        ok = extract_archive(archive, dest, remove_after=True)
        print("   ℹ️  WHAMR! generation scripts downloaded.")
        print("   ℹ️  Run the scripts to generate reverberant mixtures from WHAM! data.")
    return ok


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for Speaker Diarization & Separation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_separation_datasets.py                     # Download all 7 datasets
  python download_separation_datasets.py --only musan        # Download just MUSAN
  python download_separation_datasets.py --list              # List available datasets
  python download_separation_datasets.py --dest "E:\\data"    # Custom destination
        """
    )
    parser.add_argument("--dest", type=str, default=DEFAULT_DEST,
                        help=f"Destination directory (default: {DEFAULT_DEST})")
    parser.add_argument("--only", type=str, default=None, nargs='+',
                        help="Download only these specific datasets")
    parser.add_argument("--list", action="store_true",
                        help="List all available datasets and exit")
    args = parser.parse_args()

    # List mode
    if args.list:
        print("\n📦 Available Datasets for Voice Separation:\n")
        print(f"  {'Name':<15} {'Size':<20} {'Description'}")
        print(f"  {'─'*15} {'─'*20} {'─'*40}")
        for name, info in DATASET_REGISTRY.items():
            auto_tag = "✅ Auto" if info["auto"] else "⚠️  Manual"
            print(f"  {name:<15} {info['size_hint']:<20} {info['description']}")
        print(f"\n  Total datasets: {len(DATASET_REGISTRY)}")
        print(f"  Default destination: {DEFAULT_DEST}")
        return

    # Validate --only
    if args.only:
        for t in args.only:
            if t not in DATASET_REGISTRY:
                print(f"❌ Unknown dataset: '{t}'")
                print(f"   Available: {', '.join(DATASET_REGISTRY.keys())}")
                return

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("📥 SPEAKER SEPARATION DATASET DOWNLOADER")
    print("=" * 60)
    print(f"  Destination: {dest_root}")
    print(f"  ⚠️  This is completely separate from your classification dataset!")
    print()

    # Determine which datasets to download
    if args.only:
        targets = {t: DATASET_REGISTRY[t] for t in args.only}
    else:
        targets = DATASET_REGISTRY

    results = {}
    total_start = time.time()

    for name, info in targets.items():
        print(f"\n{'─'*60}")
        print(f"  📦 [{name.upper()}] {info['description']} ({info['size_hint']})")
        print(f"{'─'*60}")

        start = time.time()
        try:
            ok = info["download_fn"](dest_root)
            elapsed = time.time() - start
            results[name] = ok
            status = "✅ Success" if ok else "❌ Failed"
            print(f"   {status} ({elapsed:.0f}s)")
        except KeyboardInterrupt:
            print(f"\n   ⛔ Interrupted by user.")
            results[name] = False
            break
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            results[name] = False

    # Summary
    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("📊 DOWNLOAD SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")

    succeeded = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {succeeded}/{total} datasets ready")
    print(f"  ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"  📁 Saved to: {dest_root}")

    if succeeded == total:
        print(f"\n  🎉 All datasets downloaded successfully!")
        print(f"  Next step: Run 'python src/preprocessing/preprocess_separation_data.py'")
    else:
        print(f"\n  ⚠️  Some datasets failed. Re-run the script to retry.")


if __name__ == "__main__":
    main()
