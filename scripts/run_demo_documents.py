"""Run demo prompts against text documents (EPUB/PDF/MD/TXT) and dump results.

Usage:
    uv run python scripts/run_demo_documents.py
    uv run python scripts/run_demo_documents.py --prompt 1
    uv run python scripts/run_demo_documents.py --prompt 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so OPENROUTER_API_KEY is available.
# override=True because the parent shell may export an empty key.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from pydantic_ai.providers.openrouter import OpenRouterProvider  # noqa: E402

from sanjaya import Agent  # noqa: E402
from sanjaya.prompts import PromptConfig  # noqa: E402
from sanjaya.tools.document import DocumentToolkit  # noqa: E402

# ── Document collections ────────────────────────────────────

GRRM_DIR = Path("/tmp/grrm")

COLLECTIONS: dict[str, list[str]] = {
    "asoiaf_main": [
        str(GRRM_DIR / "A Game Of Thrones.epub"),
        str(GRRM_DIR / "A Clash of Kings.epub"),
        str(GRRM_DIR / "A Storm of Swords.epub"),
        str(GRRM_DIR / "A Feast for Crows.epub"),
        str(GRRM_DIR / "A Dance With Dragons.epub"),
    ],
    "asoiaf_all": [
        str(GRRM_DIR / "A Game Of Thrones.epub"),
        str(GRRM_DIR / "A Clash of Kings.epub"),
        str(GRRM_DIR / "A Storm of Swords.epub"),
        str(GRRM_DIR / "A Feast for Crows.epub"),
        str(GRRM_DIR / "A Dance With Dragons.epub"),
        str(GRRM_DIR / "The Tales of Dunk & Egg.epub"),
    ],
}

# ── Prompts ──────────────────────────────────────────────────

PROMPTS: list[dict] = [
    {
        "id": 1,
        "name": "prince_that_was_promised",
        "collection": "asoiaf_main",
        "question": (
            "Who's the prince that was promised? Give me plausible characters from the novels. "
            "For each candidate, cite specific prophecies, passages, or hints from the text that "
            "support them (with book title and chapter if possible)."
        ),
    },
]

# ── Runner ───────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_document_results"

# Override with env var for versioned output
import os as _os

if _os.getenv("DOC_RESULTS_DIR"):
    RESULTS_DIR = Path(_os.getenv("DOC_RESULTS_DIR"))


def run_prompt(
    prompt: dict,
    max_iterations: int = 20,
    max_budget_usd: float | None = 5.0,
    prompt_config: PromptConfig | None = None,
) -> dict:
    """Run a single prompt against a document collection and return results."""
    collection = COLLECTIONS[prompt["collection"]]
    missing = [p for p in collection if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing documents: {missing}")

    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
    agent = Agent(
        model="z-ai/glm-5.1",
        sub_model="openai/gpt-4.1-mini",
        provider=provider,
        prompts=prompt_config,
        max_iterations=max_iterations,
        max_budget_usd=max_budget_usd,
        tracing=True,
    )

    dt = DocumentToolkit()
    agent.use(dt)

    print(f"\n{'=' * 60}")
    print(f"PROMPT {prompt['id']}: {prompt['name']}")
    print(f"Collection: {prompt['collection']} ({len(collection)} documents)")
    for p in collection:
        print(f"  - {Path(p).name}")
    print(f"Question: {prompt['question'][:100]}...")
    print(f"{'=' * 60}\n")

    start = time.time()
    answer = agent.ask(
        prompt["question"],
        document=collection,
    )
    elapsed = time.time() - start

    # Collect trace events
    trace_events = []
    if hasattr(agent, "_tracer") and agent._tracer is not None:
        trace_events = agent._tracer.dump_events()

    result = {
        "prompt_id": prompt["id"],
        "prompt_name": prompt["name"],
        "collection": prompt["collection"],
        "document_paths": collection,
        "question": prompt["question"],
        "prompt_config": prompt_config.to_dict() if prompt_config else None,
        "answer_text": answer.text,
        "answer_data": answer.data,
        "iterations": answer.iterations,
        "cost_usd": answer.cost_usd,
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
        "wall_time_s": round(elapsed, 2),
        "evidence_count": len(answer.evidence),
        "evidence_sources": [e.source for e in answer.evidence],
        "trace_events": trace_events,
    }

    print("\n--- Result ---")
    print(f"Answer: {answer.text[:400]}...")
    print(f"Iterations: {answer.iterations}")
    if answer.cost_usd is not None:
        print(f"Cost: ${answer.cost_usd:.6f}")
    print(f"Tokens: {answer.input_tokens} in / {answer.output_tokens} out")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Evidence items: {len(answer.evidence)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run sanjaya demo prompts against documents")
    parser.add_argument(
        "--prompt",
        type=int,
        nargs="*",
        help="Prompt ID(s) to run (default: all)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Max iterations per prompt (default: 20)",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=5.0,
        help="Budget cap per prompt in USD (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data/demo_document_results)",
    )
    parser.add_argument(
        "--prompts-yaml",
        type=str,
        default=None,
        help="Path to a PromptConfig YAML file (overrides strategy/critic prompts)",
    )
    args = parser.parse_args()

    prompt_config = None
    if args.prompts_yaml:
        prompt_config = PromptConfig.from_yaml(args.prompts_yaml)
        print(f"Loaded PromptConfig from {args.prompts_yaml}")
        print(f"  Overrides: {list(prompt_config.to_dict().keys())}")

    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    prompts_to_run = PROMPTS
    if args.prompt:
        prompts_to_run = [p for p in PROMPTS if p["id"] in args.prompt]
        if not prompts_to_run:
            print(f"No prompts found with IDs: {args.prompt}")
            return

    all_results = []
    for prompt in prompts_to_run:
        try:
            result = run_prompt(
                prompt,
                max_iterations=args.max_iterations,
                max_budget_usd=args.max_budget_usd,
                prompt_config=prompt_config,
            )
            all_results.append(result)

            out_path = results_dir / f"prompt_{prompt['id']:02d}_{prompt['name']}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"\nERROR on prompt {prompt['id']} ({prompt['name']}): {e}")
            import traceback
            traceback.print_exc()
            all_results.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_name": prompt["name"],
                    "error": str(e),
                }
            )

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"DONE — {len(all_results)} prompts run")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")

    total_cost = sum(r.get("cost_usd", 0) or 0 for r in all_results)
    total_time = sum(r.get("wall_time_s", 0) or 0 for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
