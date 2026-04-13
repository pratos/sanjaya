"""Run MUIRBench evaluation using sanjaya.

Evaluates the RLM agent on MUIRBench multi-image MCQ tasks.
Supports full benchmark, task-specific subsets, and fast mode.

Usage:
    uv run python scripts/run_muirbench.py                          # full benchmark
    uv run python scripts/run_muirbench.py --limit 20               # first 20 samples
    uv run python scripts/run_muirbench.py --tasks "Visual Retrieval"
    uv run python scripts/run_muirbench.py --fast                   # reduced iterations, no critic
    uv run python scripts/run_muirbench.py --resume                 # resume from last checkpoint

Requires: uv run python scripts/download_muirbench.py first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BENCH_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks" / "muirbench"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmark_results" / "muirbench"


# ── MUIRBench answer parsing (from their postprocess.py) ─────


def parse_mcq_response(response: str, all_choices: list[str], index2ans: dict[str, str]) -> str:
    """Parse the predicted choice letter from a free-text response.

    Adapted from MUIRBENCH/eval/utils/postprocess.py
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates = []
    ans_with_brack = False

    # Try (A) (B) etc.
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    # Try standalone A B C D
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Try content matching
    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)

    if not candidates:
        return random.choice(all_choices)

    if len(candidates) > 1:
        start_indexes = []
        if ans_with_brack:
            for can in candidates:
                start_indexes.append(response.rfind(f"({can})"))
        else:
            for can in candidates:
                idx = response.rfind(f" {can} ")
                start_indexes.append(idx)
        return candidates[int(np.argmax(start_indexes))]

    return candidates[0]


# ── Prompt builder (from their preprocess.py) ────────────────


def build_mcq_prompt(question: str, options: list[str]) -> str:
    """Build MCQ prompt text from question and options."""
    parts = [f"Question: {question}", "Choices:"]
    for i, opt in enumerate(options):
        label = chr(ord("A") + i)
        if opt == "<image>":
            parts.append(f"({label}) [See image {i + 1}]")
        else:
            parts.append(f"({label}) {opt}")
    parts.append("Hint: Please provide the correct option letter, such as A, B, C, D, directly.")
    parts.append("Answer:")
    return "\n".join(parts)


# ── Main runner ──────────────────────────────────────────────


def run_sample(
    agent,
    sample: dict,
    sample_idx: int,
    total: int,
) -> dict:
    """Run a single MUIRBench sample through the agent."""
    question = sample["question"]
    options = sample["options"]
    answer_gt = sample["answer"]
    image_paths = [p for p in sample["image_paths"] if p is not None]

    # Build MCQ prompt
    prompt = build_mcq_prompt(question, options)

    # Build choice mapping
    all_choices = [chr(ord("A") + i) for i in range(len(options))]
    index2ans = {chr(ord("A") + i): opt for i, opt in enumerate(options)}

    print(f"\n[{sample_idx + 1}/{total}] Task: {sample['task']}")
    print(f"  Q: {question[:100]}...")
    print(f"  Images: {len(image_paths)}")
    print(f"  GT: {answer_gt}")

    start = time.time()
    try:
        image_arg: str | list[str] = image_paths[0] if len(image_paths) == 1 else image_paths
        answer = agent.ask(prompt, image=image_arg)
        elapsed = time.time() - start

        # Parse predicted choice
        raw_answer = answer.text or ""
        predicted = parse_mcq_response(raw_answer, all_choices, index2ans)
        correct = predicted == answer_gt

        result = {
            "idx": sample["idx"],
            "task": sample["task"],
            "image_relation": sample.get("image_relation", ""),
            "question": question,
            "answer_gt": answer_gt,
            "predicted": predicted,
            "correct": correct,
            "raw_answer": raw_answer[:500],
            "iterations": answer.iterations,
            "cost_usd": answer.cost_usd,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "wall_time_s": round(elapsed, 2),
            "n_images": len(image_paths),
        }

        status = "✅" if correct else "❌"
        print(f"  {status} Predicted: {predicted} | GT: {answer_gt} | ${answer.cost_usd or 0:.4f} | {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ ERROR: {e}")
        result = {
            "idx": sample["idx"],
            "task": sample["task"],
            "image_relation": sample.get("image_relation", ""),
            "question": question,
            "answer_gt": answer_gt,
            "predicted": "ERROR",
            "correct": False,
            "raw_answer": str(e)[:500],
            "iterations": 0,
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": round(elapsed, 2),
            "n_images": len(image_paths),
            "error": str(e),
        }

    return result


def main():
    parser = argparse.ArgumentParser(description="Run MUIRBench evaluation with sanjaya")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="Filter to specific tasks")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer iterations, no critic")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--max-budget-usd", type=float, default=None, help="Override max budget per sample")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample order")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Name for this run (default: auto-generated)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load manifest
    manifest_path = BENCH_DIR / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run: uv run python scripts/download_muirbench.py")
        sys.exit(1)

    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(samples)} MUIRBench samples")

    # Filter
    if args.tasks:
        samples = [s for s in samples if s["task"] in args.tasks]
        print(f"  Filtered to {len(samples)} for tasks: {args.tasks}")

    if args.shuffle:
        random.shuffle(samples)

    if args.limit:
        samples = samples[:args.limit]
        print(f"  Limited to {len(samples)} samples")

    # Set up run
    run_name = args.run_name or f"muirbench_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Resume support
    completed_idxs: set = set()
    results: list[dict] = []
    if args.resume:
        checkpoint_path = run_dir / "checkpoint.jsonl"
        if checkpoint_path.exists():
            for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    completed_idxs.add(r["idx"])
            print(f"  Resumed: {len(results)} completed samples")
            samples = [s for s in samples if s["idx"] not in completed_idxs]

    # Create agent
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    from sanjaya import Agent
    from sanjaya.prompts import PromptConfig

    if args.fast:
        max_iter = args.max_iterations or 3
        max_budget = args.max_budget_usd or 0.50
        critic_model = None
        print(f"  FAST mode: max_iterations={max_iter}, no critic, budget=${max_budget}")
    else:
        max_iter = args.max_iterations or 6
        max_budget = args.max_budget_usd or 1.00
        critic_model = "openrouter:qwen/qwen3-30b-a3b-thinking-2507"
        print(f"  FULL mode: max_iterations={max_iter}, with critic, budget=${max_budget}")

    # MCQ answer schema — force a simple answer format
    answer_schema = {
        "answer_letter": {
            "type": "string",
            "description": "The correct option letter (A, B, C, or D)",
            "required": True,
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning for the chosen answer",
            "required": True,
        },
    }

    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))

    # Save run config
    run_config = {
        "benchmark": "MUIRBench",
        "run_name": run_name,
        "total_samples": len(samples) + len(results),
        "remaining_samples": len(samples),
        "max_iterations": max_iter,
        "max_budget_usd": max_budget,
        "fast_mode": args.fast,
        "tasks_filter": args.tasks,
        "limit": args.limit,
        "seed": args.seed,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # Run
    checkpoint_path = run_dir / "checkpoint.jsonl"
    total = len(samples)

    for i, sample in enumerate(samples):
        # Create a fresh agent per sample to reset budget
        agent = Agent(
            model="openrouter:z-ai/glm-5.1",
            sub_model="openai/gpt-4.1-mini",
            critic_model=critic_model,
            provider=provider,
            prompts=PromptConfig(answer_schema=answer_schema),
            max_iterations=max_iter,
            max_budget_usd=max_budget,
            tracing=False,
        )

        result = run_sample(agent, sample, len(results) + i, len(results) + total)
        results.append(result)

        # Checkpoint: append to JSONL
        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, default=str) + "\n")

    # ── Final summary ────────────────────────────────────────

    correct = sum(1 for r in results if r.get("correct", False))
    total_all = len(results)
    accuracy = correct / total_all if total_all > 0 else 0
    total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)
    total_time = sum(r.get("wall_time_s", 0) or 0 for r in results)
    avg_iterations = sum(r.get("iterations", 0) for r in results) / max(total_all, 1)

    # Per-task breakdown
    task_stats: dict[str, dict] = {}
    for r in results:
        t = r["task"]
        if t not in task_stats:
            task_stats[t] = {"correct": 0, "total": 0, "cost": 0.0}
        task_stats[t]["total"] += 1
        task_stats[t]["cost"] += r.get("cost_usd", 0) or 0
        if r.get("correct"):
            task_stats[t]["correct"] += 1

    for t in task_stats:
        s = task_stats[t]
        s["accuracy"] = round(s["correct"] / max(s["total"], 1), 4)
        s["cost"] = round(s["cost"], 4)

    summary = {
        "benchmark": "MUIRBench",
        "run_name": run_name,
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total_all,
        "total_cost_usd": round(total_cost, 4),
        "total_wall_time_s": round(total_time, 1),
        "avg_cost_per_sample": round(total_cost / max(total_all, 1), 4),
        "avg_wall_time_per_sample": round(total_time / max(total_all, 1), 1),
        "avg_iterations": round(avg_iterations, 1),
        "task_breakdown": task_stats,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also save full results
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'=' * 70}")
    print(f"MUIRBench Results: {run_name}")
    print(f"{'=' * 70}")
    print(f"  Overall Accuracy: {accuracy:.1%} ({correct}/{total_all})")
    print(f"  Total Cost: ${total_cost:.4f}")
    print(f"  Avg Cost/Sample: ${total_cost / max(total_all, 1):.4f}")
    print(f"  Total Time: {total_time:.0f}s")
    print(f"  Avg Iterations: {avg_iterations:.1f}")
    print("\n  Per-Task Accuracy:")
    for t, s in sorted(task_stats.items(), key=lambda x: -x[1]["accuracy"]):
        print(f"    {t:40s} {s['accuracy']:.1%} ({s['correct']}/{s['total']}) ${s['cost']:.4f}")
    print(f"\n  Results: {run_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
