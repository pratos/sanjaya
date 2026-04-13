"""Example: optimize sanjaya prompts with GEPA.

Requires: pip install gepa sanjaya[video]

This script shows how to use GEPA's optimize_anything() to evolve
sanjaya's strategy and critic prompts against a benchmark dataset.
The PromptConfig.to_dict() / from_dict() methods provide the
dict[str, str] interface that GEPA expects for candidates.
"""

from __future__ import annotations

import gepa.optimize_anything as oa
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

from sanjaya import Agent
from sanjaya.prompts import PromptConfig

# ---------------------------------------------------------------------------
# 1. Define your benchmark dataset
# ---------------------------------------------------------------------------
# Each example needs a question, input path, and expected answer for scoring.
# Replace these with your actual benchmark data.

train_examples: list[dict] = [
    # {"question": "...", "video": "/path/to/video.mp4", "expected": "..."},
]

val_examples: list[dict] = [
    # {"question": "...", "video": "/path/to/video.mp4", "expected": "..."},
]


# ---------------------------------------------------------------------------
# 2. Define your scoring function
# ---------------------------------------------------------------------------
def score_answer(answer_text: str, expected: str) -> float:
    """Score an answer against ground truth. Returns 0.0-1.0."""
    # Replace with your actual metric (e.g., F1, ROUGE, exact match)
    return 1.0 if expected.lower() in answer_text.lower() else 0.0


# ---------------------------------------------------------------------------
# 3. Define the evaluator (GEPA calls this per example)
# ---------------------------------------------------------------------------
def evaluate(candidate: dict[str, str], example: dict) -> tuple[float, dict]:
    """Run sanjaya with the candidate prompts on one benchmark example."""
    config = PromptConfig.from_dict(candidate)
    agent = Agent(prompts=config, max_iterations=6, max_budget_usd=0.50)

    answer = agent.ask(example["question"], video=example["video"])

    # Log the agent trace for GEPA's reflection LM
    oa.log(f"Question: {example['question']}")
    oa.log(f"Answer: {answer.text}")
    oa.log(f"Iterations: {answer.iterations}, Cost: ${answer.cost_usd:.4f}")

    score = score_answer(answer.text, example["expected"])
    return score, {
        "Question": example["question"],
        "Answer": answer.text,
        "Expected": example["expected"],
        "Iterations": answer.iterations,
    }


# ---------------------------------------------------------------------------
# 4. Run optimization
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Seed candidate: start from current defaults or your custom prompts
    seed = PromptConfig(
        video_strategy="Focus on visual details and transcript evidence...",
        critic="Score strictly on factual accuracy...",
    ).to_dict()

    result = optimize_anything(
        seed_candidate=seed,
        evaluator=evaluate,
        dataset=train_examples,
        valset=val_examples,
        objective="Maximize answer accuracy while minimizing iterations and cost.",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=100, parallel=True, max_workers=4),
            reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
        ),
    )

    # Save optimized prompts
    optimized = PromptConfig.from_dict(result.best_candidate)
    print(f"Best score: {result.val_aggregate_scores}")
    print(f"Optimized config: {optimized}")
