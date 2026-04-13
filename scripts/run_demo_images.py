"""Run demo prompts against images and dump results.

Usage:
    uv run python scripts/run_demo_images.py
    uv run python scripts/run_demo_images.py --prompt 1
    uv run python scripts/run_demo_images.py --image /path/to/image.jpg --question "Describe this"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from pydantic_ai.providers.openrouter import OpenRouterProvider  # noqa: E402

from sanjaya import Agent  # noqa: E402
from sanjaya.prompts import PromptConfig  # noqa: E402

# ── Demo images directory ────────────────────────────────────

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_images"

# ── Built-in prompts (used when demo images exist) ───────────

PROMPTS: list[dict] = [
    {
        "id": 1,
        "name": "describe_single",
        "images": [str(DEMO_DIR / "sample.jpg")],
        "question": "Describe everything visible in this image in detail.",
    },
    {
        "id": 2,
        "name": "text_extraction",
        "images": [str(DEMO_DIR / "screenshot.png")],
        "question": "Read all text visible in this screenshot and list it.",
    },
    {
        "id": 3,
        "name": "compare_two",
        "images": [
            str(DEMO_DIR / "chart_q1.png"),
            str(DEMO_DIR / "chart_q2.png"),
        ],
        "question": "Compare these two charts and summarize the key differences.",
    },
    {
        "id": 4,
        "name": "crop_and_zoom",
        "images": [str(DEMO_DIR / "document.jpg")],
        "question": "What does the small label in the bottom-right corner say?",
    },
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_image_results"


def run_prompt(
    images: list[str],
    question: str,
    name: str = "adhoc",
    max_iterations: int = 10,
    max_budget_usd: float = 2.0,
    prompt_config: PromptConfig | None = None,
) -> dict:
    """Run a single image prompt and return results."""
    missing = [p for p in images if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing images: {missing}")

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

    image_arg: str | list[str] = images[0] if len(images) == 1 else images

    print(f"\n{'=' * 60}")
    print(f"PROMPT: {name}")
    print(f"Images: {len(images)}")
    for p in images:
        print(f"  - {Path(p).name}")
    print(f"Question: {question[:100]}...")
    print(f"{'=' * 60}\n")

    start = time.time()
    answer = agent.ask(question, image=image_arg)
    elapsed = time.time() - start

    result = {
        "name": name,
        "image_paths": images,
        "question": question,
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
    }

    print("\n--- Result ---")
    print(f"Answer: {answer.text[:400]}...")
    print(f"Iterations: {answer.iterations}")
    if answer.cost_usd is not None:
        print(f"Cost: ${answer.cost_usd:.6f}")
    print(f"Tokens: {answer.input_tokens} in / {answer.output_tokens} out")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Evidence: {len(answer.evidence)}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run sanjaya image analysis demos")
    parser.add_argument("--prompt", type=int, nargs="*", help="Prompt ID(s) to run")
    parser.add_argument("--image", type=str, nargs="+", help="Ad-hoc image path(s)")
    parser.add_argument("--question", type=str, help="Ad-hoc question")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--max-budget-usd", type=float, default=2.0)
    parser.add_argument("--output-dir", type=str, default=None)
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

    all_results = []

    # Ad-hoc mode: user provides --image and --question
    if args.image:
        if not args.question:
            parser.error("--question is required when using --image")
        result = run_prompt(
            images=args.image,
            question=args.question,
            name="adhoc",
            max_iterations=args.max_iterations,
            max_budget_usd=args.max_budget_usd,
            prompt_config=prompt_config,
        )
        all_results.append(result)
    else:
        # Run built-in prompts
        prompts_to_run = PROMPTS
        if args.prompt:
            prompts_to_run = [p for p in PROMPTS if p["id"] in args.prompt]

        for prompt in prompts_to_run:
            try:
                result = run_prompt(
                    images=prompt["images"],
                    question=prompt["question"],
                    name=prompt["name"],
                    max_iterations=args.max_iterations,
                    max_budget_usd=args.max_budget_usd,
                    prompt_config=prompt_config,
                )
                all_results.append(result)

                out_path = results_dir / f"{prompt['name']}.json"
                out_path.write_text(
                    json.dumps(result, indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"Saved: {out_path}")
            except FileNotFoundError as e:
                print(f"\nSKIPPED {prompt['name']}: {e}")
            except Exception as e:
                print(f"\nERROR on {prompt['name']}: {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        summary_path = results_dir / "summary.json"
        summary_path.write_text(
            json.dumps(all_results, indent=2, default=str),
            encoding="utf-8",
        )

        total_cost = sum(r.get("cost_usd", 0) or 0 for r in all_results)
        total_time = sum(r.get("wall_time_s", 0) or 0 for r in all_results)
        print(f"\n{'=' * 60}")
        print(f"DONE — {len(all_results)} prompts run")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Total time: {total_time:.0f}s")
        print(f"Summary: {summary_path}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
