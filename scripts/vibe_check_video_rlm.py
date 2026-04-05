"""Deterministic-ish smoke check for VideoRLM local pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from sanjaya.video_rlm_repl import VideoRLM_REPL


def ensure_demo_subtitle(video_path: str) -> str:
    video = Path(video_path)
    subtitle = Path("data/longvideobench/meta") / f"{video.stem}_en.json"
    subtitle.parent.mkdir(parents=True, exist_ok=True)

    if not subtitle.exists():
        subtitle.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "start": 8.0,
                            "end": 15.0,
                            "text": "A man in white shirt is speaking in a room with a map on wall.",
                        },
                        {
                            "start": 52.0,
                            "end": 61.0,
                            "text": "He continues speaking and gesturing.",
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return str(subtitle)


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoRLM smoke harness")
    parser.add_argument("--video", default="data/longvideobench/videos/7F9IrtSHmc0.mp4")
    parser.add_argument("--question", default="What is the man in white shirt doing?")
    parser.add_argument("--max-iterations", type=int, default=4)
    args = parser.parse_args()

    subtitle = ensure_demo_subtitle(args.video)

    repl = VideoRLM_REPL(max_iterations=args.max_iterations)
    answer = repl.completion(video_path=args.video, question=args.question, subtitle_path=subtitle)
    payload = answer.model_dump()

    print(json.dumps(payload, indent=2))
    print("\nSmoke summary:")
    print(f"- answer non-empty: {bool(payload.get('answer'))}")
    print(f"- evidence count: {len(payload.get('evidence', []))}")
    print(f"- retrieval trace count: {len(payload.get('retrieval_trace', []))}")


if __name__ == "__main__":
    main()
