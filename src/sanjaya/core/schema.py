"""Generate answer schemas tailored to the user's question."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_GENERATION_PROMPT = """\
Given this question, generate a JSON schema describing what fields a \
complete, well-structured answer should have.

Question: {question}

Return a JSON object with:
- "fields": dict mapping field name to {{"type": str, "description": str, "required": bool}}
- "answer_type": a short label like "summary", "factual", "comparison", "timeline", "extraction"

Rules:
- Always include a "summary" or "answer" field (the main text response)
- Always include an "evidence" field (list of supporting observations with source references)
- Add domain-specific fields based on the question (e.g., "speakers", "topics", \
"timestamps", "entities", "key_quotes", "chapters")
- Keep it to 4-8 fields max
- Every field should be fillable from the available context (transcript, vision, \
document text, audio, etc.) — do not assume any specific modality
- Do NOT include meta-fields like "confidence" or "caveats"

Return ONLY the JSON object, no explanation.
"""


def generate_answer_schema(
    question: str,
    llm_client: Any,
    schema_model: str | None = None,
) -> dict[str, Any]:
    """Generate a structured answer schema for the given question.

    Uses a cheap LLM call to infer what fields would make a complete answer.
    Falls back to a default schema if the call fails.
    """
    prompt = SCHEMA_GENERATION_PROMPT.format(question=question)

    try:
        if schema_model:
            from ..llm.client import LLMClient

            schema_llm = LLMClient(model=schema_model, name="schema")
            raw = schema_llm.completion(prompt)
        else:
            raw = llm_client.completion(prompt)

        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        schema = json.loads(text)

        # Validate minimum structure
        if "fields" not in schema or not isinstance(schema["fields"], dict):
            logger.warning("Schema generation returned invalid structure, using default")
            return default_schema()

        return schema
    except Exception as e:
        logger.warning("Schema generation failed (%s), using default", e)
        return default_schema()


def default_schema() -> dict[str, Any]:
    """Fallback schema for when generation fails."""
    return {
        "answer_type": "general",
        "fields": {
            "summary": {
                "type": "str",
                "description": "The main answer text",
                "required": True,
            },
            "key_points": {
                "type": "list[str]",
                "description": "Key points or findings",
                "required": True,
            },
            "evidence": {
                "type": "list[dict]",
                "description": "Supporting evidence with source references",
                "required": True,
            },
        },
    }


_MODALITY_PROMPT = """\
Given this question about a video, classify the primary evidence modality needed:
- "transcript_primary": answer is mostly in what people say (quotes, summaries, \
speaker attribution, dialogue analysis)
- "vision_primary": answer requires looking at what's on screen (charts, products, \
UI elements, diagrams, text overlays, physical objects, on-screen code)
- "balanced": both transcript and visual evidence are equally important

Question: {question}

Return ONLY one of: transcript_primary, vision_primary, balanced
"""

_VALID_MODALITIES = {"transcript_primary", "vision_primary", "balanced"}


def classify_question_modality(question: str, llm_client: Any) -> str:
    """Classify whether a question needs transcript-first or vision-first analysis.

    Returns one of: transcript_primary, vision_primary, balanced.
    """
    prompt = _MODALITY_PROMPT.format(question=question)
    try:
        raw = llm_client.completion(prompt).strip().lower()
        # Extract the classification even if the LLM adds extra text
        for modality in _VALID_MODALITIES:
            if modality in raw:
                return modality
        return "balanced"
    except Exception:
        return "balanced"


def schema_to_prompt_section(schema: dict[str, Any]) -> str:
    """Convert the schema to a prompt section telling the orchestrator what to produce."""
    fields = schema.get("fields", {})
    answer_type = schema.get("answer_type", "general")

    lines = [
        "## Structured Answer Format",
        "",
        f"This is a **{answer_type}** question. When you call `done(value)`, "
        "pass a **dict** with these fields:",
        "",
    ]

    for name, spec in fields.items():
        req = " (required)" if spec.get("required") else " (optional)"
        desc = spec.get("description", "")
        type_hint = spec.get("type", "str")
        lines.append(f"- `{name}` ({type_hint}){req}: {desc}")

    example_fields = ", ".join(f'"{k}": ...' for k in list(fields)[:3])
    lines.extend([
        "",
        f"Example: `done({{{example_fields}}})`",
        "",
        "Do NOT include follow-up suggestions, offers to do more analysis, or filler. "
        "Fill the fields above with grounded evidence and stop.",
    ])

    return "\n".join(lines)
