"""Structured answer and evidence models for Agent output."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Evidence(BaseModel):
    """A piece of evidence supporting the answer."""

    source: str
    rationale: str
    artifacts: dict[str, Any] = {}


class Answer(BaseModel):
    """Structured output from Agent.ask()."""

    question: str
    text: str
    evidence: list[Evidence] = []
    iterations: int
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    wall_time_s: float | None = None
    trace_id: str | None = None
