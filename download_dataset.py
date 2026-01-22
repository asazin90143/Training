"""
Automatic Dataset Downloader
Downloads audio samples from Freesound, HuggingFace, and other sources.
"""

import os
import argparse
import requests
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Try to import optional dependencies
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "dataset"

# Map internal class names to search queries
SEARCH_QUERIES = {
    "gunshot": ["gunshot", "gun fire", "shooting"],
    "glass_shatter": ["glass breaking", "window shatter", "glass crash"],
    "human_scream": ["human scream", "woman scream", "man scream"],
    "siren": ["police siren", "ambulance siren", "fire truck siren"],
    "car_alarm": ["car alarm", "vehicle alarm"],
    "explosion": ["explosion", "bomb blast", "fireworks"],
    "dog_bark": ["dog bark", "aggressive dog"],
    "power_tools": ["drill", "saw", "angle grinder"],
    "aggressive_shout": ["angry shout", "fighting voices"],
    "ambient": ["street noise", "wind", "rain", "park ambience"]
}

def download_file(url, target_path):
    """Download a file with progress."""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"    ❌ Error downloading {url}: {e}")
    return False

class FreesoundDownloader:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://freesound.org/apiv2"
    
    def search_and_download(self, query, target_dir, count=10):
        if not self.api_key:
            print("  ⚠️ No Freesound API Key provided. Skipping.")
            return 0
            
        print(f"  🎵 [Freesound] Searching for '{query}'...")
        
        # Search for high-quality, short duration sounds
        params = {
            "query": query,
            "token": self.api_key,
            "page_size": count,
            "filter": "duration:[0.5 TO 10.0]",
            "sort": "rating_desc",
            "fields": "id,name,previews,type"
        }
        
        try:
            resp = requests.get(f"{self.base_url}/search/text/", params=params)
            data = resp.json()
            
            if 'results' not in data:
                print(f"    ❌ API Error: {data}")
                return 0
                
            downloaded = 0
            for result in data['results']:
                name = "".join([c for c in result['name'] if c.isalpha() or c.isdigit()]).rstrip()
                filename = f"fs_{result['id']}_{name}.mp3" # Freesound previews are MP3
                target_path = target_dir / filename
                
                if target_path.exists():
                    continue
                    
                # Use high-quality preview
                preview_url = result['previews']['preview-hq-mp3']
                if download_file(preview_url, target_path):
                    downloaded += 1
                    time.sleep(0.5) # Be nice to API
            
            print(f"    ✅ Downloaded {downloaded} files")
            return downloaded
            
        except Exception as e:
            print(f"    ❌ Connection failed: {e}")
            return 0

class HuggingFaceDownloader:
    def __init__(self):
        pass
        
    def download_esc50(self):
        """Download relevant classes from ESC-50 dataset."""
        if not HF_AVAILABLE:
            print("  ⚠️ 'datasets' library not installed. Skipping HuggingFace.")
            print("     Run: pip install datasets")
            return
            
        print("  🤗 [HuggingFace] Loading ESC-50 dataset...")
        try:
            # ESC-50 contains: chainsaw, engine, glass_breaking, etc.
            dataset = load_dataset("ashraq/esc50", split="train")
            
            # Map ESC-50 categories to our classes
            # category column contains descriptive text
            mappings = {
                "glass_breaking": "glass_shatter",
                "siren": "siren",
                "dog": "dog_bark",
                "engine": "ambient", 
                "rain": "ambient",
                "wind": "ambient",
                "chainsaw": "power_tools",
                "fireworks": "explosion"
            }
            
            count = 0
            for item in dataset:
                category = item['category']
                if category in mappings:
                    target_class = mappings[category]
                    filename = item['filename']
                    audio_array = item['audio']['array']
                    sr = item['audio']['sampling_rate']
                    
                    # Save using soundfile (requires import inside mostly)
                    import soundfile as sf
                    
                    target_dir = DATASET_DIR / target_class
                    target_path = target_dir / f"esc50_{filename}"
                    
                    if not target_path.exists():
                        sf.write(str(target_path), audio_array, sr)
                        count += 1
            
            print(f"    ✅ Extracted {count} samples from ESC-50")
            
        except Exception as e:
            print(f"    ❌ HF Error: {e}")

class YouTubeDownloader:
    def __init__(self):
        pass
        
    def download_search(self, query, target_dir, count=5):
        if not YT_DLP_AVAILABLE:
            print("  ⚠️ yt-dlp not installed. Skipping YouTube.")
            print("     Run: pip install yt-dlp")
            return 0
            
        print(f"  📺 [YouTube] Searching for '{query}' sound effects...")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(target_dir / 'yt_%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'noplaylist': True,
            'quiet': True,
            'max_downloads': count,
            'match_filter': lambda info, **kwargs: None if info.get('duration', 0) < 60 else 'Video too long'
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Search and download top results
                ydl.download([f"ytsearch{count}:{query} sound effect"])
                print("    ✅ YouTube extraction complete")
                return count
        except Exception as e:
            print(f"    ❌ YouTube Error: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description="Auto Dataset Downloader")
    parser.add_argument("--freesound-key", help="Freesound.org API Key")
    parser.add_argument("--use-hf", action="store_true", help="Use HuggingFace datasets")
    parser.add_argument("--use-yt", action="store_true", help="Use YouTube Search")
    parser.add_argument("--count", type=int, default=10, help="Samples per class per source")
    
    args = parser.parse_args()
    
    print("="*50)
    print("⬇️ AUTOMATIC DATASET DOWNLOADER")
    print("="*50)
    
    # Initialize downloaders
    fs = FreesoundDownloader(args.freesound_key)
    hf = HuggingFaceDownloader()
    yt = YouTubeDownloader()
    
    # Create directories first
    DATASET_DIR.mkdir(exist_ok=True)
    for class_name in SEARCH_QUERIES.keys():
        (DATASET_DIR / class_name).mkdir(exist_ok=True)
    
    # 1. HuggingFace (General Purpose)
    if args.use_hf:
        hf.download_esc50()
    
    # 2. Iterate classes for search-based sources
    for class_name, queries in SEARCH_QUERIES.items():
        print(f"\n📂 Processing Class: {class_name.upper()}")
        target_dir = DATASET_DIR / class_name
        
        # Pick the best query (first one)
        primary_query = queries[0]
        
        # Freesound
        if args.freesound_key:
            fs.search_and_download(primary_query, target_dir, args.count)
            
        # YouTube
        if args.use_yt:
            yt.download_search(primary_query, target_dir, min(3, args.count))
            
    print("\n✅ Download complete!")
    print(f"Files saved in: {DATASET_DIR}")
    print("\nNext Step: Run 'python preprocess_audio.py' to clean and format the files.")

if __name__ == "__main__":
    main()
