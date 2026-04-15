#!/usr/bin/env python3
"""Download LongVideoBench and MLVU video benchmarks from HuggingFace.

Requires HF_READ_TOKEN in .env with access to:
- https://huggingface.co/datasets/longvideobench/LongVideoBench
- https://huggingface.co/datasets/MLVU/MVLU

Usage:
    uv run python scripts/download_video_benchmarks.py                    # Download both
    uv run python scripts/download_video_benchmarks.py --benchmark lvb    # LongVideoBench only
    uv run python scripts/download_video_benchmarks.py --benchmark mlvu   # MLVU only
    uv run python scripts/download_video_benchmarks.py --metadata-only    # Just JSON files (no videos)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"

# Dataset configs
DATASETS = {
    "lvb": {
        "name": "LongVideoBench",
        "repo_id": "longvideobench/LongVideoBench",
        "local_dir": BENCHMARKS_DIR / "longvideobench",
        "metadata_files": ["lvb_val.json", "lvb_test_wo_gt.json", "README.md"],
        "post_extract": [
            "cat videos.tar.part.* > videos.tar && tar -xvf videos.tar",
            "tar -xvf subtitles.tar",
        ],
    },
    "mlvu": {
        "name": "MLVU",
        "repo_id": "MLVU/MVLU",
        "local_dir": BENCHMARKS_DIR / "mlvu",
        "metadata_files": [
            "MLVU/json/1_plotQA.json",
            "MLVU/json/2_needle.json",
            "MLVU/json/3_ego.json",
            "MLVU/json/4_count.json",
            "MLVU/json/5_order.json",
            "MLVU/json/6_anomaly_reco.json",
            "MLVU/json/7_topic_reasoning.json",
            "MLVU/json/8_sub_scene.json",
            "MLVU/json/9_summary.json",
        ],
        "post_extract": [],
    },
}


def download_metadata(repo_id: str, files: list[str], local_dir: Path, token: str) -> bool:
    """Download only metadata/JSON files."""
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    success = True

    for filename in files:
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=token,
            )
            print(f"  ✓ {filename}")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
            success = False

    return success


def download_full(repo_id: str, local_dir: Path, token: str) -> bool:
    """Download entire dataset using huggingface_hub Python API."""
    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {repo_id} to {local_dir}...")
    print(f"  This may take a while for large datasets...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=token,
        )
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def run_post_extract(commands: list[str], cwd: Path) -> None:
    """Run post-download extraction commands."""
    for cmd in commands:
        print(f"  Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, cwd=str(cwd), check=True)
            print(f"  ✓ Done")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download video benchmarks from HuggingFace")
    parser.add_argument(
        "--benchmark",
        choices=["lvb", "mlvu", "both"],
        default="both",
        help="Which benchmark to download (default: both)",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only metadata/JSON files (no videos)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip post-download extraction (for LongVideoBench tar files)",
    )
    args = parser.parse_args()

    # Get token
    token = os.getenv("HF_READ_TOKEN") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: No HuggingFace token found.")
        print("Set HF_READ_TOKEN in .env or run: huggingface-cli login")
        sys.exit(1)

    # Determine which datasets to download
    if args.benchmark == "both":
        datasets_to_download = ["lvb", "mlvu"]
    else:
        datasets_to_download = [args.benchmark]

    print("=" * 60)
    print("Video Benchmark Downloader")
    print("=" * 60)

    for key in datasets_to_download:
        config = DATASETS[key]
        print(f"\n{'=' * 60}")
        print(f"Downloading: {config['name']}")
        print(f"Repo: {config['repo_id']}")
        print(f"Local: {config['local_dir']}")
        print("=" * 60)

        if args.metadata_only:
            print("\nDownloading metadata only...")
            success = download_metadata(
                repo_id=config["repo_id"],
                files=config["metadata_files"],
                local_dir=config["local_dir"],
                token=token,
            )
        else:
            print("\nDownloading full dataset (this may take a while)...")
            success = download_full(
                repo_id=config["repo_id"],
                local_dir=config["local_dir"],
                token=token,
            )

            # Run post-extraction if needed
            if success and not args.skip_extract and config["post_extract"]:
                print("\nExtracting archives...")
                run_post_extract(config["post_extract"], config["local_dir"])

        if success:
            print(f"\n✓ {config['name']} download complete!")
        else:
            print(f"\n✗ {config['name']} download had errors")

    # Summary
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    for key in datasets_to_download:
        config = DATASETS[key]
        local_dir = config["local_dir"]
        if local_dir.exists():
            files = list(local_dir.rglob("*"))
            json_files = [f for f in files if f.suffix == ".json"]
            video_files = [f for f in files if f.suffix in (".mp4", ".avi", ".mkv", ".webm")]
            print(f"\n{config['name']}:")
            print(f"  Location: {local_dir}")
            print(f"  JSON files: {len(json_files)}")
            print(f"  Video files: {len(video_files)}")
        else:
            print(f"\n{config['name']}: Not downloaded")


if __name__ == "__main__":
    main()
