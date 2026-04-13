"""Moondream cloud pricing.

Rates from https://moondream.ai/pricing (Moondream 3 real-time, base model).
Each image consumes a fixed 729 input tokens regardless of resolution.
"""

from __future__ import annotations

# Moondream 3 (Preview) — real-time, base model
MOONDREAM_INPUT_PRICE_PER_M = 0.30   # $/1M input tokens
MOONDREAM_OUTPUT_PRICE_PER_M = 2.50  # $/1M output tokens
MOONDREAM_TOKENS_PER_IMAGE = 729


def moondream_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute Moondream cost based on cloud pricing."""
    return (
        input_tokens * MOONDREAM_INPUT_PRICE_PER_M
        + output_tokens * MOONDREAM_OUTPUT_PRICE_PER_M
    ) / 1_000_000
