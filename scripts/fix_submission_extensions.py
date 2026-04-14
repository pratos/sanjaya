#!/usr/bin/env python3
"""Fix submission file predictions by matching to actual album filenames."""

import json
import os
from pathlib import Path


def load_album_filenames(album_path: Path) -> dict[str, str]:
    """Load all image filenames from an album, creating a lookup by stem."""
    images_path = album_path / "images"
    if not images_path.exists():
        images_path = album_path  # fallback if no images/ subdir
    
    lookup = {}
    for f in images_path.iterdir():
        if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.mpo'}:
            # Store by stem (without extension) and by full name
            lookup[f.stem.lower()] = f.name
            lookup[f.name.lower()] = f.name
    return lookup


def fix_prediction(pred: str, lookup: dict[str, str]) -> str:
    """Fix a single prediction by finding the matching filename."""
    # Try exact match first (case-insensitive)
    if pred.lower() in lookup:
        return lookup[pred.lower()]
    
    # Try without extension
    stem = Path(pred).stem
    if stem.lower() in lookup:
        return lookup[stem.lower()]
    
    # Try the prediction as-is but with common extensions
    for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
        candidate = stem + ext
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    
    # If still not found, return original (will fail evaluation but preserves data)
    print(f"  WARNING: Could not match '{pred}'")
    return pred


def fix_submission(submission_path: Path, albums_base: Path, output_path: Path):
    """Fix all predictions in a submission file."""
    with open(submission_path) as f:
        data = json.load(f)
    
    # Preload all album lookups
    album_lookups = {}
    for album_name in ["album1", "album2", "album3"]:
        album_path = albums_base / album_name
        if album_path.exists():
            album_lookups[album_name] = load_album_filenames(album_path)
            print(f"Loaded {len(album_lookups[album_name])} files from {album_name}")
    
    # Fix each result
    fixed_count = 0
    for result in data["results"]:
        query_id = result["query_id"]
        album_name = query_id.rsplit("_", 1)[0]  # e.g., "album2_1" -> "album2"
        
        if album_name not in album_lookups:
            print(f"  Skipping {query_id}: unknown album {album_name}")
            continue
        
        lookup = album_lookups[album_name]
        fixed_predictions = []
        
        for pred in result["predictions"]:
            fixed = fix_prediction(pred, lookup)
            if fixed != pred:
                fixed_count += 1
            fixed_predictions.append(fixed)
        
        result["predictions"] = fixed_predictions
    
    # Write fixed submission
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nFixed {fixed_count} predictions")
    print(f"Output: {output_path}")


def main():
    base = Path("data/benchmark_results/photobench")
    albums_base = Path("data/benchmarks/photobench/repo/dataset")
    
    submissions = [
        ("photobench_20260414_111313", "album1"),
        ("photobench-album2", "album2"),
        ("photobench-album3", "album3"),
    ]
    
    all_results = []
    
    for run_name, album_name in submissions:
        submission_path = base / run_name / "submission.json"
        if not submission_path.exists():
            print(f"Skipping {run_name}: no submission.json")
            continue
        
        print(f"\n=== Fixing {run_name} ===")
        
        # Load and fix
        with open(submission_path) as f:
            data = json.load(f)
        
        album_lookup = load_album_filenames(albums_base / album_name)
        
        fixed_count = 0
        for result in data["results"]:
            fixed_predictions = []
            for pred in result["predictions"]:
                fixed = fix_prediction(pred, album_lookup)
                if fixed != pred:
                    fixed_count += 1
                    print(f"  {pred} -> {fixed}")
                fixed_predictions.append(fixed)
            result["predictions"] = fixed_predictions
        
        print(f"Fixed {fixed_count} predictions in {run_name}")
        all_results.extend(data["results"])
    
    # Create combined submission
    combined = {
        "model_name": "sanjaya-rlm",
        "language": "en",
        "results": all_results
    }
    
    output_path = base / "combined_submission.json"
    with open(output_path, "w") as f:
        json.dump(combined, f, indent=2)
    
    print(f"\n=== Combined submission saved to {output_path} ===")
    print(f"Total queries: {len(all_results)}")


if __name__ == "__main__":
    main()
