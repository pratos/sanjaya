"""Download a few sample videos from LongVideoBench using yt-dlp.

Picks 5 diverse samples from the validation set (different durations/categories).
"""

import json
import subprocess
from pathlib import Path

DATA_DIR = Path("data/longvideobench")
META_DIR = DATA_DIR / "meta"
VIDEOS_DIR = DATA_DIR / "videos"


def load_validation_samples():
    """Load and parse validation samples from the parquet via the saved JSON."""
    # Use the HF API JSON we saved
    data = json.loads((META_DIR / "sample_rows_validation.json").read_text())
    rows = [r["row"] for r in data.get("rows", [])]
    return rows


def pick_diverse_samples(rows, n=5):
    """Pick n diverse samples across different categories and durations."""
    seen_videos = set()
    seen_categories = set()
    selected = []
    
    # Sort by duration group to get variety
    sorted_rows = sorted(rows, key=lambda r: r.get("duration", 0))
    
    for row in sorted_rows:
        vid = row["video_id"]
        cat = row.get("topic_category", "")
        if vid not in seen_videos and cat not in seen_categories:
            selected.append(row)
            seen_videos.add(vid)
            seen_categories.add(cat)
            if len(selected) >= n:
                break
    
    # If not enough diversity, fill remaining
    if len(selected) < n:
        for row in sorted_rows:
            vid = row["video_id"]
            if vid not in seen_videos:
                selected.append(row)
                seen_videos.add(vid)
                if len(selected) >= n:
                    break
    
    return selected


def download_video(video_id: str, output_dir: Path, max_duration: int = 600):
    """Download a YouTube video using yt-dlp."""
    output_path = output_dir / f"{video_id}.mp4"
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return True
    
    cmd = [
        "uvx", "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-o", str(output_path),
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--socket-timeout", "30",
    ]
    
    print(f"  Downloading {video_id}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
            print(f"  ✓ Downloaded: {output_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ Failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout downloading {video_id}")
        return False


def main():
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = load_validation_samples()
    print(f"Loaded {len(rows)} validation samples")
    
    samples = pick_diverse_samples(rows, n=5)
    
    print(f"\nSelected {len(samples)} diverse samples:")
    for s in samples:
        print(f"  {s['video_id']} | {s['duration']:.0f}s | {s['question_category']} / {s['topic_category']}")
        print(f"    Q: {s['question'][:100]}...")
    
    # Save selected samples metadata
    meta_path = DATA_DIR / "selected_samples.json"
    meta_path.write_text(json.dumps(samples, indent=2))
    print(f"\nSaved sample metadata to {meta_path}")
    
    # Download videos
    print("\n=== Downloading sample videos ===")
    results = []
    for s in samples:
        vid = s["video_id"]
        ok = download_video(vid, VIDEOS_DIR)
        results.append((vid, ok))
    
    print("\n=== Summary ===")
    for vid, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {vid}")
    
    success = sum(1 for _, ok in results if ok)
    print(f"\n{success}/{len(results)} videos downloaded to {VIDEOS_DIR}")


if __name__ == "__main__":
    main()
