"""Download MUIRBench dataset from HuggingFace.

MUIRBench: 2,600 MCQ across 12 multi-image understanding tasks.
Dataset: https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH
Paper: https://arxiv.org/abs/2406.09411

Usage:
    uv run python scripts/download_muirbench.py
    uv run python scripts/download_muirbench.py --limit 50       # download first 50 for testing
    uv run python scripts/download_muirbench.py --tasks "Visual Retrieval" "Scene Understanding"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "muirbench"
IMAGES_DIR = BENCH_DIR / "images"


def download(limit: int | None = None, tasks: list[str] | None = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: datasets library required. Install with: uv add datasets")
        sys.exit(1)

    print("Downloading MUIRBENCH from HuggingFace...")
    ds = load_dataset("MUIRBENCH/MUIRBENCH", split="test")
    print(f"  Total samples: {len(ds)}")

    # Explore dataset structure
    sample = ds[0]
    print(f"  Columns: {list(sample.keys())}")
    print(f"  Tasks: {sorted(set(ds['task']))}")

    # Filter by task if requested
    if tasks:
        ds = ds.filter(lambda x: x["task"] in tasks)
        print(f"  Filtered to {len(ds)} samples for tasks: {tasks}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"  Limited to {len(ds)} samples")

    # Create output directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Process and save
    manifest = []
    total = len(ds)

    for i, sample in enumerate(ds):
        idx = sample.get("idx", i)
        task = sample["task"]
        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]
        image_relation = sample.get("image_relation", "")
        image_type = sample.get("image_type", "")
        counterpart_idx = sample.get("counterpart_idx", "")

        # Save images from image_list
        image_paths = []
        image_list = sample.get("image_list", [])

        if image_list:
            sample_dir = IMAGES_DIR / f"sample_{idx}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            for j, img in enumerate(image_list):
                if img is None:
                    image_paths.append(None)
                    continue

                img_path = sample_dir / f"img_{j}.jpg"
                try:
                    img.save(str(img_path))
                    image_paths.append(str(img_path))
                except Exception as e:
                    print(f"  WARN: Failed to save image {j} for sample {idx}: {e}")
                    image_paths.append(None)

        # Build choices text (some options reference <image>)
        choices = []
        for k, opt in enumerate(options):
            choices.append({
                "label": chr(ord("A") + k),
                "text": opt,
                "is_image_ref": opt == "<image>",
            })

        entry = {
            "idx": idx,
            "task": task,
            "image_relation": image_relation,
            "image_type": image_type,
            "question": question,
            "options": options,
            "choices": choices,
            "answer": answer,
            "image_paths": image_paths,
            "counterpart_idx": counterpart_idx,
        }
        manifest.append(entry)

        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  Processed {i + 1}/{total}")

    # Save manifest
    manifest_path = BENCH_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    # Save task distribution summary
    task_counts: dict[str, int] = {}
    for entry in manifest:
        t = entry["task"]
        task_counts[t] = task_counts.get(t, 0) + 1

    summary = {
        "benchmark": "MUIRBench",
        "total_samples": len(manifest),
        "tasks": task_counts,
        "image_relations": sorted(set(e["image_relation"] for e in manifest if e["image_relation"])),
        "image_types": sorted(set(e["image_type"] for e in manifest if e["image_type"])),
        "avg_images_per_sample": round(
            sum(len([p for p in e["image_paths"] if p]) for e in manifest) / max(len(manifest), 1), 1
        ),
        "avg_options_per_sample": round(
            sum(len(e["options"]) for e in manifest) / max(len(manifest), 1), 1
        ),
    }
    summary_path = BENCH_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"MUIRBench downloaded: {len(manifest)} samples")
    print(f"  Tasks: {len(task_counts)}")
    for t, c in sorted(task_counts.items()):
        print(f"    {t}: {c}")
    print(f"  Avg images/sample: {summary['avg_images_per_sample']}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Images: {IMAGES_DIR}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Download MUIRBench dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Filter to specific tasks")
    args = parser.parse_args()

    download(limit=args.limit, tasks=args.tasks)


if __name__ == "__main__":
    main()
