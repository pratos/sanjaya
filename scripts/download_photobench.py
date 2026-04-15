"""Download PhotoBench dataset.

PhotoBench: Personal photo album retrieval + reasoning benchmark.
Repo: https://github.com/LaVieEnRose365/PhotoBench
Data: https://gitee.com/sorrowtea/PhotoBench (complete images)
      https://huggingface.co/datasets/LaVieEnRose365/PhotoBench (HF mirror)

Usage:
    uv run python scripts/download_photobench.py
    uv run python scripts/download_photobench.py --source github   # clone from GitHub (samples only)
    uv run python scripts/download_photobench.py --source hf       # try HuggingFace
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "photobench"


def download_from_github() -> None:
    """Clone PhotoBench from GitHub (includes sample images + scripts)."""
    repo_dir = BENCH_DIR / "repo"

    if repo_dir.exists():
        print(f"  Repo already exists at {repo_dir}, pulling latest...")
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)
    else:
        print("  Cloning LaVieEnRose365/PhotoBench from GitHub...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "https://github.com/LaVieEnRose365/PhotoBench.git", str(repo_dir)],
            check=True,
        )

    # Explore what we got
    print(f"  Repo cloned to: {repo_dir}")
    for p in sorted(repo_dir.rglob("*.json"))[:10]:
        print(f"    {p.relative_to(repo_dir)}")
    for p in sorted(repo_dir.rglob("*.py"))[:10]:
        print(f"    {p.relative_to(repo_dir)}")

    # Look for dataset files
    dataset_dir = repo_dir / "dataset"
    if dataset_dir.exists():
        # Check if the actual data (albums + JSON queries) exists
        has_albums = any(
            d.is_dir() and d.name.startswith("album") for d in dataset_dir.iterdir()
        )
        has_queries = (dataset_dir / "train.json").exists() or (dataset_dir / "test.json").exists()

        if has_albums or has_queries:
            _process_dataset_dir(dataset_dir)
        else:
            print("\n  GitHub repo cloned but dataset images/queries not included.")
            print("  The actual data must be downloaded separately.")
            _print_manual_download_instructions()
    else:
        print("  WARN: No dataset/ directory found.")
        _print_manual_download_instructions()


def download_from_hf() -> None:
    """Download PhotoBench from HuggingFace datasets."""
    print("PhotoBench is NOT on HuggingFace. The dataset must be downloaded manually.")
    print()
    _print_manual_download_instructions()


def _print_manual_download_instructions() -> None:
    """Print instructions for manually downloading the PhotoBench dataset."""
    dataset_dir = BENCH_DIR / "repo" / "dataset"
    print("=" * 60)
    print("PhotoBench dataset requires manual download:")
    print()
    print("  Option 1 (Google Drive):")
    print("    https://drive.google.com/drive/folders/1ODJqgbC9Hu_EfP9m4xP31DLb84ZDJqCC")
    print()
    print("  Option 2 (OAS Box, password: Oppo2026):")
    print("    https://sbox.myoas.com/l/B10d84e1cfa514920")
    print()
    print("  After downloading, place the album folders and JSON files in:")
    print(f"    {dataset_dir}/")
    print()
    print("  Expected structure:")
    print(f"    {dataset_dir}/album1/  (photos)")
    print(f"    {dataset_dir}/album2/  (photos)")
    print(f"    {dataset_dir}/album3/  (photos)")
    print(f"    {dataset_dir}/train.json")
    print(f"    {dataset_dir}/test.json")
    print()
    print("  Then re-run this script to process the data.")
    print("=" * 60)


def _process_dataset_dir(dataset_dir: Path) -> None:
    """Process a downloaded PhotoBench dataset directory."""
    print(f"\n  Processing dataset at: {dataset_dir}")

    # Find query files
    train_file = dataset_dir / "train.json"
    test_file = dataset_dir / "test.json"

    manifest = {"benchmark": "PhotoBench", "albums": [], "queries": {"train": [], "test": []}}

    # Find albums
    album_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("album")])
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    for album_dir in album_dirs:
        images = [p for p in album_dir.rglob("*") if p.suffix.lower() in image_exts]
        manifest["albums"].append({
            "name": album_dir.name,
            "path": str(album_dir),
            "image_count": len(images),
        })
        print(f"    Album '{album_dir.name}': {len(images)} images")

    # Process query files
    for label, qfile in [("train", train_file), ("test", test_file)]:
        if qfile.exists():
            try:
                data = json.loads(qfile.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    manifest["queries"][label] = data
                    print(f"    {label}.json: {len(data)} queries")
                elif isinstance(data, dict):
                    manifest["queries"][label] = data.get("queries", data.get("results", [data]))
                    print(f"    {label}.json: dict with keys {list(data.keys())}")
            except Exception as e:
                print(f"    WARN: Failed to parse {label}.json: {e}")

    # Also check for samples directory
    samples_dir = dataset_dir / "samples"
    if samples_dir and samples_dir.exists():
        sample_albums = sorted([d for d in samples_dir.iterdir() if d.is_dir()])
        for sd in sample_albums:
            images = [p for p in sd.rglob("*") if p.suffix.lower() in image_exts]
            print(f"    Samples '{sd.name}': {len(images)} images")

    # Save manifest
    manifest_path = BENCH_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    summary = {
        "benchmark": "PhotoBench",
        "total_albums": len(manifest["albums"]),
        "total_images": sum(a["image_count"] for a in manifest["albums"]),
        "train_queries": len(manifest["queries"]["train"]),
        "test_queries": len(manifest["queries"]["test"]),
    }
    summary_path = BENCH_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("PhotoBench processed:")
    print(f"  Albums: {summary['total_albums']}")
    print(f"  Total images: {summary['total_images']}")
    print(f"  Train queries: {summary['train_queries']}")
    print(f"  Test queries: {summary['test_queries']}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Download PhotoBench dataset")
    parser.add_argument(
        "--source",
        choices=["github", "hf"],
        default="github",
        help="Download source (default: github)",
    )
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        download_from_hf()
    else:
        download_from_github()


if __name__ == "__main__":
    main()
