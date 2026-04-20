#!/usr/bin/env python3
"""Run video benchmarks with OpenAI models via OpenRouter.

Includes:
  - 12 original demo prompts (open-ended, 5 YouTube videos)
  - 9 LongVideoBench MCQ prompts (diverse categories, 9 YouTube videos)

Root LLM:  openai/gpt-5.3-codex (OpenRouter)
Sub LLM:   openai/gpt-4.1-mini (OpenRouter)
Vision:    openai/gpt-4.1-mini (OpenRouter)
Caption:   openai/gpt-4.1-mini (OpenRouter)

Uses multiprocessing (default 6 workers) to run prompts in parallel.

Usage:
    uv run python scripts/run_video_benchmarks.py                      # all 20 prompts
    uv run python scripts/run_video_benchmarks.py --prompt 2 4 9       # specific prompts
    uv run python scripts/run_video_benchmarks.py --prompt 13 14 15    # LVB prompts only
    uv run python scripts/run_video_benchmarks.py --workers 4          # 4 parallel workers
    uv run python scripts/run_video_benchmarks.py --fast               # fewer iterations
    uv run python scripts/run_video_benchmarks.py --download-lvb       # download LVB videos first
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

# ── Models ───────────────────────────────────────────────────

ROOT_MODEL = "openrouter:openai/gpt-5.3-codex"
SUB_MODEL = "openrouter:openai/gpt-4.1-mini"
VISION_MODEL = "openrouter:openai/gpt-4.1-mini"
CAPTION_MODEL = "openrouter:openai/gpt-4.1-mini"
CRITIC_MODEL = None  # skip critic for benchmark runs

# ── Paths ────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "youtube"
LVB_DIR = PROJECT_ROOT / "data" / "longvideobench" / "videos"
RESULTS_DIR = PROJECT_ROOT / "data" / "demo_results_codex53"

# ── Video paths ──────────────────────────────────────────────

VIDEOS = {
    "podcast": {
        "video": str(DATA / "5RAFKES5J6E.mp4"),
        "subtitle": str(DATA / "meta" / "5RAFKES5J6E_en.json"),
    },
    "curry": {
        "video": str(DATA / "zVg3FvMuDlw.mp4"),
        "subtitle": str(DATA / "meta" / "zVg3FvMuDlw_en.json"),
    },
    "mkbhd": {
        "video": str(DATA / "rng_yUSwrgU.mp4"),
        "subtitle": str(DATA / "meta" / "rng_yUSwrgU_en.json"),
    },
    "football": {
        "video": str(DATA / "9gyv2xh7qQw.mp4"),
        "subtitle": str(DATA / "meta" / "9gyv2xh7qQw_en.json"),
    },
    "tech_talk": {
        "video": str(DATA / "qdfwmYTO0Aw.mp4"),
        "subtitle": str(DATA / "meta" / "qdfwmYTO0Aw_en.json"),
    },
    # ── LongVideoBench videos ────────────────────────────────
    "lvb_cooking": {
        "video": str(LVB_DIR / "1R5uPaL0V-0.mp4"),
        "youtube_id": "1R5uPaL0V-0",
        "category": "Cooking",
        "duration_min": 16.8,
    },
    "lvb_movie": {
        "video": str(LVB_DIR / "N7RTTiHsSjI.mp4"),
        "youtube_id": "N7RTTiHsSjI",
        "category": "Movie Recap",
        "duration_min": 8.0,
    },
    "lvb_travel": {
        "video": str(LVB_DIR / "kOZnpwI2hIM.mp4"),
        "youtube_id": "kOZnpwI2hIM",
        "category": "Travel",
        "duration_min": 16.7,
    },
    "lvb_history": {
        "video": str(LVB_DIR / "fvCrE5NCsts.mp4"),
        "youtube_id": "fvCrE5NCsts",
        "category": "History",
        "duration_min": 8.5,
    },
    "lvb_art": {
        "video": str(LVB_DIR / "fZBC3nmvJb8.mp4"),
        "youtube_id": "fZBC3nmvJb8",
        "category": "Art",
        "duration_min": 20.1,
    },
    "lvb_geography": {
        "video": str(LVB_DIR / "lzAESaVqix0.mp4"),
        "youtube_id": "lzAESaVqix0",
        "category": "Geography",
        "duration_min": 19.8,
    },
    "lvb_stem": {
        "video": str(LVB_DIR / "zda-T6wrEhs.mp4"),
        "youtube_id": "zda-T6wrEhs",
        "category": "STEM",
        "duration_min": 8.5,
    },
    "lvb_vlog": {
        "video": str(LVB_DIR / "Jfp1Ks7Hh1E.mp4"),
        "youtube_id": "Jfp1Ks7Hh1E",
        "category": "Life Vlog",
        "duration_min": 15.2,
    },
    "lvb_napoleon": {
        "video": str(LVB_DIR / "P9hDA0u6FO0.mp4"),
        "youtube_id": "P9hDA0u6FO0",
        "category": "History (long)",
        "duration_min": 33.3,
    },
}

# ── LongVideoBench MCQ question data ────────────────────────
# One question per video, picked for visual+transcript balance

LVB_QUESTIONS: dict[int, dict] = {
    13: {
        "video_key": "lvb_cooking",
        "question": (
            "The person on the screen is a black-haired man wearing a T-shirt and a black scarf. "
            "He is in a white kitchen, with a stove behind him. On the stove, there is a pot. "
            "The man is using a spoon to stir the contents of the pot. "
            "What is the material of the spoon he is using?"
        ),
        "candidates": ["plastic", "wood", "ceramic", "stainless steel", "glass"],
        "gt_answer": "wood",
    },
    14: {
        "video_key": "lvb_movie",
        "question": (
            "On the exterior facade of a building, there are three windows in the middle, "
            "a black pipe on the left side, and a white air conditioner unit on the right side. "
            "What is the shape of the air conditioner unit?"
        ),
        "candidates": ["Cuboid", "Spherical", "Cube", "Cylindrical", "Conical"],
        "gt_answer": "Cuboid",
    },
    15: {
        "video_key": "lvb_travel",
        "question": (
            "Inside the airport, there is a dense crowd. On the left side of the screen, "
            "there's an area surrounded by blue lines and silver pillars, with white stuff on the ground. "
            "After this scene, what appears on the screen?"
        ),
        "candidates": [
            "A woman in blue with a work badge",
            "A woman in red with a work badge",
            "A man in red with a work badge",
            "A potted plant",
            "An umbrella",
        ],
        "gt_answer": "A woman in red with a work badge",
    },
    16: {
        "video_key": "lvb_history",
        "question": (
            "In the black-and-white scene, there is a grass hut. Inside the grass hut, "
            "there is a person holding a long gun without revealing their body. "
            "When the subtitle mentions 'fired a shot that was heard around the world', "
            "what happened next in the video?"
        ),
        "candidates": [
            "The gun fired",
            "A bird landed on the ground",
            "The gun was placed on the ground",
            "The person stood up from the grass hut",
            "The gun was thrown out",
        ],
        "gt_answer": "The gun fired",
    },
    17: {
        "video_key": "lvb_art",
        "question": (
            "In a room, there is a white piece of furniture in the back. Next to the furniture, "
            "there is a wooden object. On the right side, there are various bottles. "
            "A person is sitting in a chair, holding something in their hand. "
            "What are they holding?"
        ),
        "candidates": ["Black marker", "Rag", "Small brush", "Brush", "Black crayon"],
        "gt_answer": "Small brush",
    },
    18: {
        "video_key": "lvb_geography",
        "question": (
            "After a man wearing a red short-sleeved shirt and a black hat finished speaking "
            "in front of a black background, what did this man do?"
        ),
        "candidates": [
            "picked up a basketball",
            "picked up a stick",
            "picked up a soccer ball",
            "picked up a painting",
            "picked up a pot of flowers",
        ],
        "gt_answer": "picked up a basketball",
    },
    19: {
        "video_key": "lvb_stem",
        "question": (
            "When the video shows a room with black and orange colors, there is a bald man "
            "wearing a floral shirt sitting on a sofa. What is this man doing?"
        ),
        "candidates": [
            "Both hands are raised",
            "One hand is holding a pen and the other hand is holding a book",
            "One hand is placed on his knee",
            "Both hands are placed on his knees and he is clenching his fists",
        ],
        "gt_answer": "Both hands are placed on his knees and he is clenching his fists",
    },
    20: {
        "video_key": "lvb_vlog",
        "question": (
            "Who is the first person to appear in the video?"
        ),
        "candidates": [
            "The man sitting in front of a desk with a projector and computer in a room with hanging lights, wearing a black hoodie and jeans",
            "The man sitting on the bed in a room with hanging lights, wearing a black short-sleeve shirt and black pants",
            "The woman with black hair, wearing a black and white coat with a white top, a white headset around her neck, opening a door",
            "The woman walking outdoors at night in front of a wall with English writing, wearing a black short-sleeve shirt and shorts, and carrying a paper bag while wearing headphones",
            "The man in a dimly lit room, sitting in front of a computer, wearing a white hoodie and facing a mirror",
        ],
        "gt_answer": "The man sitting on the bed in a room with hanging lights, wearing a black short-sleeve shirt and black pants",
    },
    21: {
        "video_key": "lvb_napoleon",
        "question": (
            "Which of the following sequences of scenes is correct?"
        ),
        "candidates": [
            "A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame. On the right, a man in a dark blue military uniform with a white flower in his hair and medals on his chest. On the left, a yellow paper sheet. Under a blue sky, a man in a white short sleeve shirt holding a camera stands by a yellow rock.",
            "On the right, a man in a dark blue military uniform with a white flower in his hair and medals on his chest. On the left, a yellow paper sheet. A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame. Under a blue sky, a man in a white short sleeve shirt holding a camera stands by a yellow rock.",
            "A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame. Under a blue sky, a man in a white short sleeve shirt holding a camera stands by a yellow rock. On the right, a man in a dark blue military uniform with a white flower in his hair and medals on his chest. On the left, a yellow paper sheet.",
            "On the right, a man in a dark blue military uniform with a white flower in his hair and medals on his chest. On the left, a yellow paper sheet. Under a blue sky, a man in a white short sleeve shirt holding a camera stands by a yellow rock. A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame.",
        ],
        "gt_answer": "A man wearing a military hat, a long coat, and long boots, with white and red English words in the frame. On the right, a man in a dark blue military uniform with a white flower in his hair and medals on his chest. On the left, a yellow paper sheet. Under a blue sky, a man in a white short sleeve shirt holding a camera stands by a yellow rock.",
    },
}


def _format_mcq_question(question: str, candidates: list[str]) -> str:
    """Format an MCQ question with lettered options."""
    letters = "ABCDEFGHIJ"
    options = "\n".join(
        f"  {letters[i]}. {c}" for i, c in enumerate(candidates) if i < len(letters)
    )
    return (
        f"{question}\n\n"
        f"**Options:**\n{options}\n\n"
        f"Choose the EXACT text of the correct option. "
        f"Use vision tools to examine the relevant moment in the video."
    )


# ── 20 Prompts ───────────────────────────────────────────────

PROMPTS: list[dict] = [
    # ── Original 12 demo prompts ─────────────────────────────
    {"id": 1, "name": "long_video_summarization", "video_key": "podcast", "question": "Summarize this video with timestamped key points, main arguments, and any disagreements between speakers"},
    {"id": 2, "name": "sports_moment_finding", "video_key": "curry", "question": "Find every 3-pointer made in this video. For each one, give me the timestamp, the player's jersey number, and whether it was contested"},
    {"id": 3, "name": "general_moment_finding", "video_key": "tech_talk", "question": "Find every time the presenter makes a mistake or corrects themselves. For each, give the timestamp, what they said, and what the correction was"},
    {"id": 4, "name": "podcast_highlights", "video_key": "podcast", "question": "Identify the 3 most viral-worthy moments from this podcast. For each, give the exact start and end timestamps for a 30-60 second clip, a suggested caption, and why it would perform well on short-form platforms"},
    {"id": 5, "name": "quote_mining", "video_key": "podcast", "question": "Find standalone quotes or hot takes from this conversation that would work as text-overlay clips for Instagram Reels. Give me 5 candidates with exact timestamps, the verbatim quote, and a suggested text overlay"},
    {"id": 6, "name": "multi_speaker_attribution", "video_key": "podcast", "question": "How many distinct speakers are in this video? For each speaker, give a description of their appearance, their approximate total speaking time, and the 3 most important points they made with timestamps"},
    {"id": 7, "name": "fact_checking_visuals", "video_key": "tech_talk", "question": "The speaker makes several claims and references visuals on screen. List each claim with its timestamp, whether a chart or diagram was shown to support it, and whether the visual actually matches what was said"},
    {"id": 8, "name": "match_progression", "video_key": "football", "question": "Describe the progression of this football match. Identify the before state and final state with timestamps, list each goal scored with the timestamp and scorer, and describe how the momentum shifted"},
    {"id": 9, "name": "ad_sponsorship_detection", "video_key": "mkbhd", "question": "Find every sponsored segment, ad read, or product placement in this video. List the start and end timestamps, the brand or product mentioned, and whether it was a verbal mention, visual placement, or both"},
    {"id": 10, "name": "lecture_notes", "video_key": "tech_talk", "question": "Generate study notes from this video. Include key concepts with definitions, any diagrams or code shown on screen with their timestamps, and a list of topics the presenter emphasized or repeated"},
    {"id": 11, "name": "product_analysis", "video_key": "mkbhd", "question": "This is a product review video. Extract every feature discussed with its timestamp, any pricing mentioned, and any competitor comparisons made by the reviewer"},
    {"id": 12, "name": "football_event_detection", "video_key": "football", "question": "List every notable event in this match: goals, fouls, cards, substitutions, and VAR decisions. For each, give the timestamp, what happened, and which players were involved"},
    # ── LongVideoBench MCQ prompts ───────────────────────────
    {"id": 13, "name": "lvb_cooking_material", "video_key": "lvb_cooking", "question": "MCQ", "is_mcq": True},
    {"id": 14, "name": "lvb_movie_action", "video_key": "lvb_movie", "question": "MCQ", "is_mcq": True},
    {"id": 15, "name": "lvb_travel_observation", "video_key": "lvb_travel", "question": "MCQ", "is_mcq": True},
    {"id": 16, "name": "lvb_history_event", "video_key": "lvb_history", "question": "MCQ", "is_mcq": True},
    {"id": 17, "name": "lvb_art_object", "video_key": "lvb_art", "question": "MCQ", "is_mcq": True},
    {"id": 18, "name": "lvb_geography_action", "video_key": "lvb_geography", "question": "MCQ", "is_mcq": True},
    {"id": 19, "name": "lvb_stem_body_lang", "video_key": "lvb_stem", "question": "MCQ", "is_mcq": True},
    {"id": 20, "name": "lvb_vlog_first_person", "video_key": "lvb_vlog", "question": "MCQ", "is_mcq": True},
    {"id": 21, "name": "lvb_napoleon_sequence", "video_key": "lvb_napoleon", "question": "MCQ", "is_mcq": True},
]


# ── Video download helper ────────────────────────────────────

def download_lvb_videos() -> None:
    """Download LongVideoBench videos using yt-dlp."""
    LVB_DIR.mkdir(parents=True, exist_ok=True)

    lvb_video_keys = [v for k, v in VIDEOS.items() if k.startswith("lvb_")]
    to_download = []
    for cfg in lvb_video_keys:
        vid_path = Path(cfg["video"])
        if not vid_path.exists():
            to_download.append(cfg)

    if not to_download:
        print("All LVB videos already downloaded.")
        return

    print(f"Downloading {len(to_download)} LVB videos...")
    for cfg in to_download:
        yt_id = cfg["youtube_id"]
        out_path = Path(cfg["video"])
        print(f"  Downloading {yt_id} ({cfg['category']}, ~{cfg['duration_min']:.0f}min)...")
        try:
            subprocess.run(
                [
                    "uv", "run", "--with", "yt-dlp", "yt-dlp",
                    f"https://www.youtube.com/watch?v={yt_id}",
                    "-f", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
                    "--merge-output-format", "mp4",
                    "-o", str(out_path),
                    "--no-playlist",
                    "--socket-timeout", "30",
                ],
                check=True,
                timeout=600,
            )
            print(f"  ✓ {yt_id}")
        except Exception as e:
            print(f"  ✗ {yt_id}: {e}")


# ── Subtitle helper ──────────────────────────────────────────

def _check_subtitle_exists(video_path: str) -> bool:
    src = Path(video_path)
    stem = src.stem
    candidates = [
        src.with_name(f"{stem}_en.json"),
        src.parent / "meta" / f"{stem}_en.json",
        src.parent.parent / "meta" / f"{stem}_en.json",
    ]
    return any(c.exists() for c in candidates)


# ── Worker function (runs in subprocess) ─────────────────────

def _run_single_prompt(
    prompt: dict,
    max_iterations: int,
    max_budget_usd: float,
) -> dict:
    """Run a single prompt through the Agent. Executed in a subprocess."""
    import json
    import sys
    import time
    import traceback
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

    from sanjaya import Agent
    from sanjaya.tools.video import VideoToolkit

    prompt_id = prompt["id"]
    prompt_name = prompt["name"]
    video_cfg = VIDEOS[prompt["video_key"]]
    is_mcq = prompt.get("is_mcq", False)

    # Build the actual question text
    if is_mcq:
        mcq_data = LVB_QUESTIONS[prompt_id]
        question = _format_mcq_question(mcq_data["question"], mcq_data["candidates"])
        gt_answer = mcq_data["gt_answer"]
    else:
        question = prompt["question"]
        gt_answer = None

    # Resolve subtitle
    subtitle_path = video_cfg.get("subtitle")
    if subtitle_path and not Path(subtitle_path).exists():
        subtitle_path = None

    has_existing_sub = subtitle_path is not None or _check_subtitle_exists(video_cfg["video"])
    mcq_tag = " [MCQ]" if is_mcq else ""

    print(
        f"\n{'=' * 60}\n"
        f"PROMPT {prompt_id}: {prompt_name}{mcq_tag}\n"
        f"Video: {prompt['video_key']}\n"
        f"Question: {question[:100]}...\n"
        f"{'=' * 60}",
        flush=True,
    )

    start = time.time()
    try:
        agent = Agent(
            model=ROOT_MODEL,
            sub_model=SUB_MODEL,
            vision_model=VISION_MODEL,
            caption_model=CAPTION_MODEL,
            critic_model=CRITIC_MODEL,
            max_iterations=max_iterations,
            max_depth=2,
            max_budget_usd=max_budget_usd,
            tracing=True,
        )

        vt = VideoToolkit(max_frames_per_clip=8)
        agent.use(vt)

        answer = agent.ask(
            question,
            video=video_cfg["video"],
            subtitle=subtitle_path,
        )
        elapsed = time.time() - start

        subtitle_generated = not has_existing_sub and _check_subtitle_exists(video_cfg["video"])
        subtitle_info = {
            "had_existing_subtitle": has_existing_sub,
            "subtitle_generated": subtitle_generated,
            "subtitle_source": "existing" if has_existing_sub else ("whisper_local" if subtitle_generated else "none"),
        }

        trace_events = []
        if hasattr(agent, "_tracer") and agent._tracer is not None:
            trace_events = agent._tracer.dump_events()

        # MCQ correctness check
        mcq_correct = None
        mcq_predicted = None
        if is_mcq and gt_answer:
            raw_text = answer.text or ""
            answer_data = answer.data if isinstance(answer.data, dict) else {}
            mcq_predicted = _extract_mcq_answer(raw_text, answer_data, mcq_data["candidates"])
            mcq_correct = mcq_predicted is not None and mcq_predicted.strip() == gt_answer.strip()

        result = {
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "video_key": prompt["video_key"],
            "question": question[:500],
            "is_mcq": is_mcq,
            "config": {
                "root_model": ROOT_MODEL,
                "sub_model": SUB_MODEL,
                "vision_model": VISION_MODEL,
                "caption_model": CAPTION_MODEL,
                "max_depth": 2,
                "max_budget_usd": max_budget_usd,
                "max_iterations": max_iterations,
            },
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
        if is_mcq:
            result["mcq_gt_answer"] = gt_answer
            result["mcq_predicted"] = mcq_predicted
            result["mcq_correct"] = mcq_correct

        if is_mcq:
            status = " ✅" if mcq_correct else " ❌"
            print(f"\n--- Prompt {prompt_id} Result{status} ---")
            print(f"Predicted: {mcq_predicted!r} | GT: {gt_answer!r}")
        else:
            print(f"\n--- Prompt {prompt_id} Result ---")
            print(f"Answer: {(answer.text or '')[:200]}...")

        print(
            f"Iterations: {answer.iterations} | Cost: ${answer.cost_usd:.6f}\n"
            f"Tokens: {answer.input_tokens} in / {answer.output_tokens} out\n"
            f"Wall time: {elapsed:.1f}s | Evidence: {len(answer.evidence)}",
            flush=True,
        )

    except Exception as e:
        elapsed = time.time() - start
        print(f"\n💥 Prompt {prompt_id} ({prompt_name}) ERROR: {e}", flush=True)
        traceback.print_exc()
        result = {
            "prompt_id": prompt_id,
            "prompt_name": prompt_name,
            "video_key": prompt["video_key"],
            "question": question[:500],
            "is_mcq": is_mcq,
            "error": str(e),
            "cost_usd": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "wall_time_s": round(elapsed, 2),
        }
        if is_mcq:
            result["mcq_correct"] = False

    return result


def _extract_mcq_answer(raw_answer: str, answer_data: dict, candidates: list[str]) -> str | None:
    """Extract the predicted answer from the agent's MCQ output."""
    import re

    # 1. Structured data
    if answer_data:
        for key in ("answer", "prediction", "choice", "selected"):
            val = answer_data.get(key)
            if isinstance(val, str) and val.strip():
                # Letter mapping
                if len(val.strip()) == 1 and val.strip().upper() in "ABCDEFGHIJ":
                    idx = ord(val.strip().upper()) - ord("A")
                    if 0 <= idx < len(candidates):
                        return candidates[idx]
                # Exact match
                for c in candidates:
                    if val.strip().lower() == c.strip().lower():
                        return c
                # Substring
                for c in candidates:
                    if c.strip().lower() in val.strip().lower():
                        return c
                return val.strip()

    # 2. Quoted candidate
    for c in candidates:
        if f'"{c}"' in raw_answer or f"'{c}'" in raw_answer:
            return c

    # 3. "answer" field in text
    answer_patterns = [
        r'"answer"\s*:\s*"([^"]*)"',
        r"answer[:\s]+(.+?)(?:\n|$)",
    ]
    for pat in answer_patterns:
        m = re.search(pat, raw_answer, re.IGNORECASE)
        if m:
            matched = m.group(1).strip()
            for c in candidates:
                if c.strip().lower() == matched.lower() or c.strip().lower() in matched.lower():
                    return c

    # 4. Letter pattern
    letter_match = re.search(r"(?:answer|choice|option)\s*(?:is|:)\s*([A-E])\b", raw_answer, re.IGNORECASE)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(candidates):
            return candidates[idx]

    # 5. Any candidate mentioned
    for c in candidates:
        if c.strip().lower() in raw_answer.lower():
            return c

    return None


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run video benchmarks (12 demo + 9 LVB MCQ) with Codex 5.3 + GPT-4.1-mini"
    )
    parser.add_argument("--prompt", type=int, nargs="*", help="Prompt ID(s) to run (default: all 20)")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers (default: 6)")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max iterations per prompt (default: 20)")
    parser.add_argument("--max-budget-usd", type=float, default=1.0, help="Budget cap per prompt (default: $1.00)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: 10 iterations, $0.50 budget")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--download-lvb", action="store_true", help="Download LVB videos before running")
    args = parser.parse_args()

    if args.download_lvb:
        download_lvb_videos()

    if args.fast:
        max_iter = args.max_iterations if args.max_iterations != 20 else 10
        max_budget = args.max_budget_usd if args.max_budget_usd != 1.0 else 0.50
    else:
        max_iter = args.max_iterations
        max_budget = args.max_budget_usd

    prompts_to_run = PROMPTS
    if args.prompt:
        prompts_to_run = [p for p in PROMPTS if p["id"] in args.prompt]
        if not prompts_to_run:
            print(f"No prompts found with IDs: {args.prompt}")
            sys.exit(1)

    # Check video availability
    missing = []
    for p in prompts_to_run:
        vid = VIDEOS[p["video_key"]]
        if not Path(vid["video"]).exists():
            missing.append((p["id"], p["video_key"], vid["video"]))
    if missing:
        print("⚠️  Missing videos:")
        for pid, vkey, vpath in missing:
            print(f"  Prompt {pid} ({vkey}): {vpath}")
        lvb_missing = [m for m in missing if "lvb" in m[1]]
        if lvb_missing:
            print("\nRun with --download-lvb to fetch LongVideoBench videos.")
        prompts_to_run = [p for p in prompts_to_run if Path(VIDEOS[p["video_key"]]["video"]).exists()]
        if not prompts_to_run:
            print("ERROR: No prompts with available videos.")
            sys.exit(1)
        print(f"Continuing with {len(prompts_to_run)} prompts that have videos.\n")

    run_name = args.run_name or f"codex53_{time.strftime('%Y%m%d_%H%M%S')}"
    results_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    run_dir = results_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    n_demo = sum(1 for p in prompts_to_run if not p.get("is_mcq"))
    n_mcq = sum(1 for p in prompts_to_run if p.get("is_mcq"))

    run_config = {
        "run_name": run_name,
        "models": {"root": ROOT_MODEL, "sub": SUB_MODEL, "vision": VISION_MODEL, "caption": CAPTION_MODEL, "critic": CRITIC_MODEL},
        "max_iterations": max_iter,
        "max_budget_usd_per_prompt": max_budget,
        "max_depth": 2,
        "workers": args.workers,
        "prompts": [p["id"] for p in prompts_to_run],
        "n_demo_prompts": n_demo,
        "n_mcq_prompts": n_mcq,
        "fast_mode": args.fast,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}")
    print(f"Video Benchmark — Codex 5.3 + GPT-4.1-mini")
    print(f"{'=' * 70}")
    print(f"  Run:          {run_name}")
    print(f"  Prompts:      {len(prompts_to_run)} ({n_demo} demo + {n_mcq} MCQ)")
    print(f"  Workers:      {args.workers}")
    print(f"  Root model:   {ROOT_MODEL}")
    print(f"  Sub model:    {SUB_MODEL}")
    print(f"  Vision:       {VISION_MODEL}")
    print(f"  Caption:      {CAPTION_MODEL}")
    print(f"  Max iter:     {max_iter}")
    print(f"  Budget/prompt: ${max_budget:.2f}")
    print(f"  Results:      {run_dir}")
    print(f"{'=' * 70}\n")

    # ── Run with multiprocessing ─────────────────────────────

    results: list[dict] = []
    checkpoint_path = run_dir / "checkpoint.jsonl"
    overall_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_to_prompt = {
            pool.submit(_run_single_prompt, prompt, max_iter, max_budget): prompt
            for prompt in prompts_to_run
        }

        completed = 0
        for future in as_completed(future_to_prompt):
            completed += 1
            prompt = future_to_prompt[future]
            try:
                result = future.result(timeout=900)
            except Exception as e:
                result = {
                    "prompt_id": prompt["id"],
                    "prompt_name": prompt["name"],
                    "video_key": prompt["video_key"],
                    "is_mcq": prompt.get("is_mcq", False),
                    "error": f"Process error: {e}",
                    "cost_usd": 0,
                    "wall_time_s": 0,
                }
                if prompt.get("is_mcq"):
                    result["mcq_correct"] = False
                print(f"  💥 Prompt {prompt['id']} PROCESS ERROR: {e}", flush=True)

            results.append(result)

            out_path = run_dir / f"prompt_{prompt['id']:02d}_{prompt['name']}.json"
            out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

            with open(checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str) + "\n")

            elapsed = time.time() - overall_start
            print(f"\n  >>> Progress: {completed}/{len(prompts_to_run)} | Elapsed: {elapsed:.0f}s\n", flush=True)

    overall_elapsed = time.time() - overall_start

    # ── Summary ──────────────────────────────────────────────

    total_cost = sum(r.get("cost_usd", 0) or 0 for r in results)
    total_time = sum(r.get("wall_time_s", 0) or 0 for r in results)
    total_input = sum(r.get("input_tokens", 0) or 0 for r in results)
    total_output = sum(r.get("output_tokens", 0) or 0 for r in results)
    errors = sum(1 for r in results if "error" in r)
    avg_iter = sum(r.get("iterations", 0) or 0 for r in results) / max(len(results), 1)

    mcq_results = [r for r in results if r.get("is_mcq")]
    mcq_correct = sum(1 for r in mcq_results if r.get("mcq_correct"))
    mcq_accuracy = mcq_correct / len(mcq_results) if mcq_results else None

    summary = {
        "run_name": run_name,
        "models": run_config["models"],
        "total_prompts": len(results),
        "n_demo": n_demo,
        "n_mcq": n_mcq,
        "mcq_accuracy": round(mcq_accuracy, 4) if mcq_accuracy is not None else None,
        "mcq_correct": mcq_correct,
        "mcq_total": len(mcq_results),
        "errors": errors,
        "total_cost_usd": round(total_cost, 4),
        "total_wall_time_s": round(overall_elapsed, 1),
        "sum_wall_time_s": round(total_time, 1),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "avg_cost_per_prompt": round(total_cost / max(len(results), 1), 4),
        "avg_iterations": round(avg_iter, 1),
        "per_prompt": [
            {
                "id": r.get("prompt_id"),
                "name": r.get("prompt_name"),
                "video": r.get("video_key"),
                "is_mcq": r.get("is_mcq", False),
                "mcq_correct": r.get("mcq_correct"),
                "cost_usd": r.get("cost_usd", 0),
                "iterations": r.get("iterations", 0),
                "wall_time_s": r.get("wall_time_s", 0),
                "evidence_count": r.get("evidence_count", 0),
                "error": r.get("error"),
            }
            for r in sorted(results, key=lambda x: x.get("prompt_id", 0))
        ],
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    # ── Print summary ────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print(f"Results — {run_name}")
    print(f"{'=' * 70}")
    print(f"  Total Cost:        ${total_cost:.4f}")
    print(f"  Wall Time (real):  {overall_elapsed:.0f}s ({overall_elapsed / 60:.1f}min)")
    print(f"  Wall Time (sum):   {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"  Tokens:            {total_input} in / {total_output} out")
    print(f"  Avg Iterations:    {avg_iter:.1f}")
    if mcq_results:
        print(f"  MCQ Accuracy:      {mcq_accuracy:.1%} ({mcq_correct}/{len(mcq_results)})")
    if errors:
        print(f"  Errors:            {errors}")
    print()

    print(f"  {'ID':<4} {'Name':<28} {'Video':<14} {'Cost':>8} {'Iter':>5} {'Time':>7} {'MCQ':>5}")
    print(f"  {'-' * 75}")
    for r in sorted(results, key=lambda x: x.get("prompt_id", 0)):
        pid = r.get("prompt_id", "?")
        name = (r.get("prompt_name", "") or "")[:27]
        video = (r.get("video_key", "") or "")[:13]
        cost = r.get("cost_usd", 0) or 0
        iters = r.get("iterations", 0) or 0
        wtime = r.get("wall_time_s", 0) or 0
        err = " ERR" if r.get("error") else ""
        mcq_str = ""
        if r.get("is_mcq"):
            mcq_str = " ✅" if r.get("mcq_correct") else " ❌"
        print(f"  {pid:<4} {name:<28} {video:<14} ${cost:>6.4f} {iters:>5} {wtime:>6.1f}s{mcq_str}{err}")

    print(f"\n  Results: {run_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
