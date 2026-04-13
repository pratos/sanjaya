"""Run demo prompts against test videos and dump results.

Usage:
    uv run python scripts/run_demo_prompts.py
    uv run python scripts/run_demo_prompts.py --prompt 2      # run a single prompt
    uv run python scripts/run_demo_prompts.py --prompt 2 4 9  # run specific prompts
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from pydantic_ai.providers.openrouter import OpenRouterProvider

from sanjaya import Agent
from sanjaya.prompts import PromptConfig
from sanjaya.tools.video import VideoToolkit

# ── Video paths ──────────────────────────────────────────────

DATA = Path(__file__).resolve().parent.parent / "data" / "youtube"

VIDEOS = {
    "podcast": {
        "video": str(DATA / "5RAFKES5J6E.mp4"),
        "subtitle": str(DATA / "meta" / "5RAFKES5J6E_en.json"),
    },
    "curry": {
        "video": str(DATA / "zVg3FvMuDlw.mp4"),
    },
    "mkbhd": {
        "video": str(DATA / "rng_yUSwrgU.mp4"),
        "subtitle": str(DATA / "meta" / "rng_yUSwrgU_en.json"),
    },
    "football": {
        "video": str(DATA / "9gyv2xh7qQw.mp4"),
    },
    "tech_talk": {
        "video": str(DATA / "qdfwmYTO0Aw.mp4"),
        "subtitle": str(DATA / "meta" / "qdfwmYTO0Aw_en.json"),
    },
}

# ── Prompts mapped to videos ────────────────────────────────

PROMPTS: list[dict] = [
    {
        "id": 1,
        "name": "long_video_summarization",
        "video_key": "podcast",
        "question": (
            "Summarize this video with timestamped key points, main arguments, and any disagreements between speakers"
        ),
    },
    {
        "id": 2,
        "name": "sports_moment_finding",
        "video_key": "curry",
        "question": (
            "Find every 3-pointer made in this video. For each one, give me "
            "the timestamp, the player's jersey number, and whether it was contested"
        ),
    },
    {
        "id": 3,
        "name": "general_moment_finding",
        "video_key": "tech_talk",
        "question": (
            "Find every time the presenter makes a mistake or corrects themselves. "
            "For each, give the timestamp, what they said, and what the correction was"
        ),
    },
    {
        "id": 4,
        "name": "podcast_highlights",
        "video_key": "podcast",
        "question": (
            "Identify the 3 most viral-worthy moments from this podcast. "
            "For each, give the exact start and end timestamps for a 30-60 second clip, "
            "a suggested caption, and why it would perform well on short-form platforms"
        ),
    },
    {
        "id": 5,
        "name": "quote_mining",
        "video_key": "podcast",
        "question": (
            "Find standalone quotes or hot takes from this conversation that would work "
            "as text-overlay clips for Instagram Reels. Give me 5 candidates with "
            "exact timestamps, the verbatim quote, and a suggested text overlay"
        ),
    },
    {
        "id": 6,
        "name": "multi_speaker_attribution",
        "video_key": "podcast",
        "question": (
            "How many distinct speakers are in this video? For each speaker, "
            "give a description of their appearance, their approximate total speaking time, "
            "and the 3 most important points they made with timestamps"
        ),
    },
    {
        "id": 7,
        "name": "fact_checking_visuals",
        "video_key": "tech_talk",
        "question": (
            "The speaker makes several claims and references visuals on screen. "
            "List each claim with its timestamp, whether a chart or diagram was shown "
            "to support it, and whether the visual actually matches what was said"
        ),
    },
    {
        "id": 8,
        "name": "match_progression",
        "video_key": "football",
        "question": (
            "Describe the progression of this football match. Identify the before state "
            "and final state with timestamps, list each goal scored with the timestamp "
            "and scorer, and describe how the momentum shifted"
        ),
    },
    {
        "id": 9,
        "name": "ad_sponsorship_detection",
        "video_key": "mkbhd",
        "question": (
            "Find every sponsored segment, ad read, or product placement in this video. "
            "List the start and end timestamps, the brand or product mentioned, "
            "and whether it was a verbal mention, visual placement, or both"
        ),
    },
    {
        "id": 10,
        "name": "lecture_notes",
        "video_key": "tech_talk",
        "question": (
            "Generate study notes from this video. Include key concepts with definitions, "
            "any diagrams or code shown on screen with their timestamps, "
            "and a list of topics the presenter emphasized or repeated"
        ),
    },
    {
        "id": 11,
        "name": "product_analysis",
        "video_key": "mkbhd",
        "question": (
            "This is a product review video. Extract every feature discussed "
            "with its timestamp, any pricing mentioned, "
            "and any competitor comparisons made by the reviewer"
        ),
    },
    {
        "id": 12,
        "name": "football_event_detection",
        "video_key": "football",
        "question": (
            "List every notable event in this match: goals, fouls, cards, "
            "substitutions, and VAR decisions. For each, give the timestamp, "
            "what happened, and which players were involved"
        ),
    },
]

# ── Runner ───────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_results"


def _check_subtitle_exists(video_path: str) -> bool:
    """Check if a subtitle sidecar already exists for this video."""
    src = Path(video_path)
    stem = src.stem
    candidates = [
        src.with_name(f"{stem}_en.json"),
        src.parent / "meta" / f"{stem}_en.json",
        src.parent.parent / "meta" / f"{stem}_en.json",
    ]
    return any(c.exists() for c in candidates)


def run_prompt(
    prompt: dict,
    max_iterations: int = 20,
    prompt_config: PromptConfig | None = None,
) -> dict:
    """Run a single prompt and return results."""
    video_cfg = VIDEOS[prompt["video_key"]]

    provider = OpenRouterProvider(api_key=os.getenv("OPENROUTER_API_KEY"))
    agent = Agent(
        model="z-ai/glm-5.1",
        sub_model="openai/gpt-4.1-mini",
        vision_model="moondream-station:moondream3-preview",
        caption_model="moondream-station:moondream3-preview",
        provider=provider,
        prompts=prompt_config,
        max_iterations=max_iterations,
        tracing=True,
    )

    vt = VideoToolkit(max_frames_per_clip=8)
    agent.use(vt)

    print(f"\n{'=' * 60}")
    print(f"PROMPT {prompt['id']}: {prompt['name']}")
    print(f"Video: {prompt['video_key']}")
    print(f"Question: {prompt['question'][:80]}...")
    print(f"{'=' * 60}\n")

    # Check if subtitle generation will be needed
    has_explicit_sub = video_cfg.get("subtitle") is not None
    has_existing_sub = has_explicit_sub or _check_subtitle_exists(video_cfg["video"])

    subtitle_path = video_cfg.get("subtitle")
    if subtitle_path and not Path(subtitle_path).exists():
        print(f"⚠️  Configured subtitle not found: {subtitle_path}")
        subtitle_path = None

    start = time.time()
    answer = agent.ask(
        prompt["question"],
        video=video_cfg["video"],
        subtitle=subtitle_path,
    )
    elapsed = time.time() - start

    # Detect subtitle generation cost (time-based estimate)
    subtitle_generated = not has_existing_sub and _check_subtitle_exists(video_cfg["video"])
    subtitle_info = {
        "had_existing_subtitle": has_existing_sub,
        "subtitle_generated": subtitle_generated,
        "subtitle_source": "existing" if has_existing_sub else ("whisper_local" if subtitle_generated else "none"),
    }

    # Collect trace events from the agent's tracer
    trace_events = []
    if hasattr(agent, "_tracer") and agent._tracer is not None:
        trace_events = agent._tracer.dump_events()

    result = {
        "prompt_id": prompt["id"],
        "prompt_name": prompt["name"],
        "video_key": prompt["video_key"],
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
        "subtitle": subtitle_info,
        "trace_events": trace_events,
    }

    print("\n--- Result ---")
    print(f"Answer: {answer.text[:200]}...")
    print(f"Iterations: {answer.iterations}")
    print(f"Cost: ${answer.cost_usd:.6f}")
    print(f"Tokens: {answer.input_tokens} in / {answer.output_tokens} out")
    print(f"Wall time: {elapsed:.1f}s")
    print(f"Evidence clips: {len(answer.evidence)}")
    print(f"Subtitle: {subtitle_info['subtitle_source']}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run sanjaya demo prompts")
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
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: data/demo_results)",
    )
    parser.add_argument(
        "--prompts-yaml",
        type=str,
        default=None,
        help="Path to a PromptConfig YAML file (overrides strategy/critic prompts)",
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    prompt_config = None
    if args.prompts_yaml:
        prompt_config = PromptConfig.from_yaml(args.prompts_yaml)
        print(f"Loaded PromptConfig from {args.prompts_yaml}")
        print(f"  Overrides: {list(prompt_config.to_dict().keys())}")

    prompts_to_run = PROMPTS
    if args.prompt:
        prompts_to_run = [p for p in PROMPTS if p["id"] in args.prompt]
        if not prompts_to_run:
            print(f"No prompts found with IDs: {args.prompt}")
            return

    all_results = []
    for prompt in prompts_to_run:
        try:
            result = run_prompt(prompt, max_iterations=args.max_iterations, prompt_config=prompt_config)
            all_results.append(result)

            # Save individual result
            out_path = results_dir / f"prompt_{prompt['id']:02d}_{prompt['name']}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            print(f"Saved: {out_path}")
        except Exception as e:
            print(f"\nERROR on prompt {prompt['id']} ({prompt['name']}): {e}")
            all_results.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt_name": prompt["name"],
                    "error": str(e),
                }
            )

    # Save summary
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"DONE — {len(all_results)} prompts run")
    print(f"Summary: {summary_path}")
    print(f"{'=' * 60}")

    # Print cost summary
    total_cost = sum(r.get("cost_usd", 0) or 0 for r in all_results)
    total_time = sum(r.get("wall_time_s", 0) or 0 for r in all_results)
    errors = sum(1 for r in all_results if "error" in r)
    print(f"\nTotal cost: ${total_cost:.4f}")
    print(f"Total time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
