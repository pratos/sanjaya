# /// script
# requires-python = ">=3.12"
# dependencies = ["sanjaya"]
# ///
"""Quick smoke test for caption_frames on a short video."""
from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from sanjaya.agent import Agent

VIDEO = "/Users/pratos/Downloads/NJusA8_Az55YvNz8.mp4"
QUESTION = "What is shown in this video?"


def main() -> None:
    agent = Agent(
        max_iterations=4,
        caption_model="moondream:moondream3-preview",
    )

    print(f"[test] video={VIDEO} ({QUESTION})")
    t0 = time.time()
    answer = agent.ask(
        QUESTION,
        video=VIDEO,
        subtitle=None,
    )
    elapsed = time.time() - t0
    print(f"\n[test] done in {elapsed:.1f}s")
    print(f"[test] answer:\n{answer.text}")


if __name__ == "__main__":
    main()
