"""Run PhotoBench evaluation using sanjaya.

Evaluates the RLM agent on PhotoBench photo album retrieval + reasoning.
Sanjaya's ImageToolkit is used to search/reason over album images.

Unlike MUIRBench (MCQ), PhotoBench is a retrieval task:
  Input:  a query + an album of images
  Output: ranked list of image filenames matching the query

Usage:
    uv run python scripts/run_photobench.py                         # full benchmark
    uv run python scripts/run_photobench.py --limit 10              # first 10 queries
    uv run python scripts/run_photobench.py --album album1          # single album
    uv run python scripts/run_photobench.py --fast                  # reduced iterations
    uv run python scripts/run_photobench.py --max-album-images 50   # cap images per album

Requires: uv run python scripts/download_photobench.py first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "photobench"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_results" / "photobench"


def find_album_images(album_dir: Path, max_images: int | None = None) -> list[str]:
    """Find all image files in an album directory."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    images = sorted(
        [str(p) for p in album_dir.rglob("*") if p.suffix.lower() in extensions]
    )
    if max_images and len(images) > max_images:
        print(f"    Capping album from {len(images)} to {max_images} images")
        images = images[:max_images]
    return images


def run_query(
    agent_cls,
    agent_kwargs: dict,
    query: dict,
    album_images: list[str],
    query_idx: int,
    total: int,
) -> dict:
    """Run a single PhotoBench query through the agent.

    The agent gets all album images and a natural-language query.
    It should return a ranked list of matching image filenames.
    """
    query_text = query.get("query", query.get("query_cn", query.get("query_en", "")))
    query_id = query.get("query_id", str(query_idx))
    ground_truth = query.get("ground_truth", query.get("gt", []))

    # Build the retrieval prompt
    prompt = f"""You are searching through a photo album to find images matching this query:

"{query_text}"

You have {len(album_images)} images loaded. Use the available tools to:
1. Start with list_images() to see all loaded images
2. Use search_images(query) to search by caption keywords (this captions all images first)
3. Use vision_query(prompt, image_id=...) to inspect promising candidates
4. Use compare_images() if you need to compare candidates

Return a JSON list of the image filenames (not full paths) that match the query,
ranked by relevance (most relevant first). Return ONLY filenames, not image IDs.

If no images match, return an empty list.
"""

    print(f"\n[{query_idx + 1}/{total}] Query: {query_text[:100]}...")
    print(f"  Album: {len(album_images)} images")
    if ground_truth:
        print(f"  GT: {len(ground_truth)} matches")

    start = time.time()
    try:
        # Create fresh agent per query
        agent = agent_cls(**agent_kwargs)
        answer = agent.ask(prompt, image=album_images)
        elapsed = time.time() - start

        raw_answer = answer.text or ""

        # Try to parse predictions from the answer
        predictions = _parse_predictions(raw_answer, answer.data)

        # Compute retrieval metrics if ground truth exists
        metrics = {}
        if ground_truth:
            metrics = _compute_retrieval_metrics(predictions, ground_truth)

        result = {
            "query_id": query_id,
            "query": query_text,
            "predictions": predictions,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "raw_answer": raw_answer[:1000],
            "iterations": answer.iterations,
            "cost_usd": answer.cost_usd,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "wall_time_s": round(elapsed, 2),
            "n_album_images": len(album_images),
        }

        recall = metrics.get("recall", "N/A")
        precision = metrics.get("precision", "N/A")
        print(f"  Predictions: {len(predictions)} | R={recall} | P={precision}")
        print(f"  ${answer.cost_usd or 0:.4f} | {elapsed:.1f}s | {answer.iterations} iter")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ ERROR: {e}")
        result = {
            "query_id": query_id,
            "query": query_text,
            "predictions": [],
            "ground_truth": ground_truth,
            "metrics": {},
            "raw_answer": str(e)[:500],
            "iterations": 0,
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": round(elapsed, 2),
            "n_album_images": len(album_images),
            "error": str(e),
        }

    return result


def _parse_predictions(raw_answer: str, answer_data: dict | None) -> list[str]:
    """Extract predicted filenames from agent response."""
    # Try structured data first
    if answer_data:
        for key in ["predictions", "filenames", "matches", "results", "images"]:
            if key in answer_data and isinstance(answer_data[key], list):
                return [str(f) for f in answer_data[key]]

    # Try to parse JSON from raw text
    import re

    # Look for JSON array in the response
    json_match = re.search(r'\[.*?\]', raw_answer, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(f) for f in parsed]
        except json.JSONDecodeError:
            pass

    # Look for filenames with image extensions
    filenames = re.findall(r'[\w\-\.]+\.(?:jpg|jpeg|png|webp)', raw_answer, re.IGNORECASE)
    return list(dict.fromkeys(filenames))  # deduplicate preserving order


def _compute_retrieval_metrics(predictions: list[str], ground_truth: list[str]) -> dict:
    """Compute retrieval metrics: precision, recall, F1, MRR."""
    if not ground_truth:
        return {}

    gt_set = set(str(g) for g in ground_truth)
    pred_set = set(str(p) for p in predictions)

    tp = len(gt_set & pred_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gt_set), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    # Mean Reciprocal Rank
    mrr = 0.0
    for i, pred in enumerate(predictions):
        if str(pred) in gt_set:
            mrr = 1.0 / (i + 1)
            break

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mrr": round(mrr, 4),
        "tp": tp,
        "fp": len(pred_set) - tp,
        "fn": len(gt_set) - tp,
    }


def main():
    parser = argparse.ArgumentParser(description="Run PhotoBench evaluation with sanjaya")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--album", type=str, default=None, help="Specific album name")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer iterations")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--max-budget-usd", type=float, default=None)
    parser.add_argument("--max-album-images", type=int, default=100, help="Cap images per album (default: 100)")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    # Load manifest
    manifest_path = BENCH_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run: uv run python scripts/download_photobench.py")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Load queries
    queries = manifest.get("queries", {}).get(args.split, [])
    if not queries:
        # Try loading from file directly
        query_file = BENCH_DIR / "repo" / "dataset" / f"{args.split}.json"
        if query_file.exists():
            queries = json.loads(query_file.read_text(encoding="utf-8"))
            if isinstance(queries, dict):
                queries = queries.get("queries", queries.get("data", [queries]))

    if not queries:
        print(f"ERROR: No queries found for split '{args.split}'")
        print("Check that PhotoBench dataset was downloaded correctly.")
        sys.exit(1)

    print(f"Loaded {len(queries)} PhotoBench queries ({args.split} split)")

    # Find album images
    albums: dict[str, list[str]] = {}
    album_infos = manifest.get("albums", [])

    # Also check repo/dataset for albums
    repo_dataset = BENCH_DIR / "repo" / "dataset"
    if repo_dataset.exists():
        for d in sorted(repo_dataset.iterdir()):
            if d.is_dir() and d.name.startswith("album"):
                imgs = find_album_images(d, max_images=args.max_album_images)
                if imgs:
                    albums[d.name] = imgs
                    print(f"  Album '{d.name}': {len(imgs)} images")

    if not albums:
        for info in album_infos:
            p = Path(info["path"])
            if p.exists():
                imgs = find_album_images(p, max_images=args.max_album_images)
                if imgs:
                    albums[info["name"]] = imgs

    if not albums:
        print("ERROR: No album images found. Check download.")
        sys.exit(1)

    # Filter
    if args.album:
        albums = {k: v for k, v in albums.items() if k == args.album}
        if not albums:
            print(f"ERROR: Album '{args.album}' not found")
            sys.exit(1)

    if args.limit:
        queries = queries[:args.limit]

    # Setup
    run_name = args.run_name or f"photobench_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.fast:
        max_iter = args.max_iterations or 5
        max_budget = args.max_budget_usd or 2.00
        critic_model = None
    else:
        max_iter = args.max_iterations or 8
        max_budget = args.max_budget_usd or 5.00
        critic_model = "openrouter:qwen/qwen3-30b-a3b-thinking-2507"

    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from sanjaya import Agent

    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
    agent_kwargs = {
        "model": "openrouter:openai/gpt-5.3-codex",
        "sub_model": "openai/gpt-4.1-mini",
        "critic_model": critic_model,
        "provider": provider,
        "max_iterations": max_iter,
        "max_budget_usd": max_budget,
        "tracing": False,
    }

    # Save config
    run_config = {
        "benchmark": "PhotoBench",
        "run_name": run_name,
        "split": args.split,
        "total_queries": len(queries),
        "albums": {k: len(v) for k, v in albums.items()},
        "max_iterations": max_iter,
        "max_budget_usd": max_budget,
        "max_album_images": args.max_album_images,
        "fast_mode": args.fast,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Run queries
    results: list[dict] = []
    checkpoint_path = run_dir / "checkpoint.jsonl"

    for i, query in enumerate(queries):
        # Determine which album this query belongs to
        query_album = query.get("album", query.get("album_name", ""))
        album_images = None
        if query_album and query_album in albums:
            album_images = albums[query_album]
        else:
            # Use first album as fallback
            album_images = list(albums.values())[0]

        result = run_query(
            agent_cls=Agent,
            agent_kwargs=agent_kwargs,
            query=query,
            album_images=album_images,
            query_idx=i,
            total=len(queries),
        )
        results.append(result)

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")

    # ── Summary ──────────────────────────────────────────────

    total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)
    total_time = sum(r.get("wall_time_s", 0) or 0 for r in results)
    avg_iter = sum(r.get("iterations", 0) for r in results) / max(len(results), 1)

    # Average retrieval metrics
    metrics_with_gt = [r["metrics"] for r in results if r.get("metrics")]
    avg_metrics = {}
    if metrics_with_gt:
        for key in ["precision", "recall", "f1", "mrr"]:
            vals = [m[key] for m in metrics_with_gt if key in m]
            if vals:
                avg_metrics[f"avg_{key}"] = round(sum(vals) / len(vals), 4)

    summary = {
        "benchmark": "PhotoBench",
        "run_name": run_name,
        "total_queries": len(results),
        "total_cost_usd": round(total_cost, 4),
        "total_wall_time_s": round(total_time, 1),
        "avg_cost_per_query": round(total_cost / max(len(results), 1), 4),
        "avg_iterations": round(avg_iter, 1),
        "retrieval_metrics": avg_metrics,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Build submission file (for PhotoBench eval website)
    submission = {
        "model_name": "sanjaya-rlm",
        "language": "en",
        "results": [
            {"query_id": r["query_id"], "predictions": r["predictions"]}
            for r in results
        ],
    }
    (run_dir / "submission.json").write_text(json.dumps(submission, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"PhotoBench Results: {run_name}")
    print(f"{'=' * 70}")
    print(f"  Queries: {len(results)}")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"  Avg Cost/Query: ${total_cost / max(len(results), 1):.4f}")
    print(f"  Avg Iterations: {avg_iter:.1f}")
    if avg_metrics:
        print("  Retrieval Metrics:")
        for k, v in avg_metrics.items():
            print(f"    {k}: {v:.4f}")
    print(f"\n  Results: {run_dir}")
    print(f"  Submission file: {run_dir / 'submission.json'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
