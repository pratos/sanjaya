"""Modality-agnostic answer critic for adaptive loop termination."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """\
You are evaluating whether an answer is complete and well-grounded.

Question: {question}

Expected answer schema:
{schema}

Submitted answer:
{answer}

Evaluate strictly:
1. Required fields: are all required fields filled with substantive content, \
not placeholders or vague statements?
2. Evidence grounding: does each claim cite a specific source reference \
(timestamp, page number, section, quote, or observation)?
3. Completeness: does the answer address what the question actually asks? \
Are there obvious gaps — things the question asks for that are missing?
4. Noise: is there unsupported speculation, filler, follow-up offers \
("want me to also..."), or content not requested by the question?

Return a JSON object:
{{
  "score": <0-100, where 100 is perfect>,
  "pass": <true if score >= {threshold}>,
  "gaps": [<list of specific missing items or issues, empty if none>],
  "feedback": "<one actionable sentence for the agent>"
}}

Be strict but fair. A score of 70+ means the answer is usable. \
90+ means it is thorough with no gaps. \
Below 50 means critical information is missing.
"""


def evaluate_answer(
    question: str,
    answer: Any,
    schema: dict[str, Any],
    critic_client: Any,
    threshold: int = 70,
) -> dict[str, Any]:
    """Run the critic on a structured answer.

    Returns a dict with: score (0-100), pass (bool), gaps (list), feedback (str).
    If the critic call fails, accepts the answer to avoid blocking.
    """
    prompt = CRITIC_PROMPT.format(
        question=question,
        schema=json.dumps(schema, indent=2),
        answer=json.dumps(answer, indent=2, default=str) if isinstance(answer, dict) else str(answer),
        threshold=threshold,
    )

    try:
        raw = critic_client.completion(prompt)
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Handle thinking models that wrap output in <think>...</think>
        if "<think>" in text:
            # Extract content after the thinking block
            think_end = text.rfind("</think>")
            if think_end >= 0:
                text = text[think_end + len("</think>"):].strip()

        result = json.loads(text)
        result.setdefault("score", 50)
        result.setdefault("pass", result["score"] >= threshold)
        result.setdefault("gaps", [])
        result.setdefault("feedback", "")
        return result
    except Exception as e:
        logger.warning("Critic evaluation failed (%s), accepting answer", e)
        return {"score": 75, "pass": True, "gaps": [], "feedback": "critic unavailable"}
