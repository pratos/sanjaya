"""Modality-agnostic answer critic for adaptive loop termination."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

CRITIC_PROMPT = """\
You are evaluating whether an answer is complete, well-grounded, and substantive.

Question: {question}

Expected answer schema:
{schema}

Submitted answer:
{answer}

Evaluate strictly on ALL six dimensions:
1. Required fields: are all required fields filled with substantive content, \
not placeholders or vague statements?
2. Evidence grounding: does each claim cite a specific source reference \
(timestamp, page number, section, quote, or observation)?
3. Completeness: does the answer address what the question actually asks? \
Are there obvious gaps — things the question asks for that are missing?
4. Noise: is there unsupported speculation, filler, follow-up offers \
("want me to also..."), or content not requested by the question?
5. Content quality: are individual claims specific and substantive? \
Reject vague non-answers like "the presenter continues", "further analysis \
needed", or "immediate continuation after the slip." Each list item must \
contain a concrete detail (a name, number, timestamp, or specific description), \
not just a category label or generic narration.
6. Quote validity (if quotes or verbatim text are present): each quote must \
be a complete, coherent utterance — not a mid-sentence fragment. A valid quote \
starts naturally and ends at a sentence or clause boundary. Fragments like \
"of the model are not that great among context because they never seen" are \
NOT acceptable as standalone quotes. If suggested interpretations or overlays \
are provided, they must accurately reflect what was said.

Return a JSON object:
{{
  "score": <0-100, where 100 is perfect>,
  "pass": <true if score >= {threshold}>,
  "gaps": [<list of specific missing items or issues, empty if none>],
  "feedback": "<one actionable sentence for the agent>"
}}

Scoring guidance:
- 70+ means usable with specific, verifiable claims.
- 90+ means thorough with no gaps and all claims grounded.
- Below 50 means critical information is missing.
- Score 0-30 if the answer contains mid-sentence fragments posing as quotes, \
vague hand-waving as descriptions, or "corrections" / "details" that don't \
describe what actually changed or happened.
- Score 0 if the answer admits it could not extract information or has empty \
lists for required fields. An answer that says "I could not find X" when X \
was explicitly asked for is a failure, not an honest answer.
"""


def evaluate_answer(
    question: str,
    answer: Any,
    schema: dict[str, Any],
    critic_client: Any,
    threshold: int = 70,
    budget: Any | None = None,
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

        # Track critic cost in the budget
        if budget is not None:
            usage = getattr(critic_client, "last_usage", None)
            if usage:
                cost = getattr(critic_client, "last_cost_usd", None) or 0.0
                model = getattr(critic_client, "model", None)
                budget.record(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    model=str(model) if model else "critic",
                )
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
