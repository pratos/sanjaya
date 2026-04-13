"""API request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RunRequest(BaseModel):
    """Payload to start a new VideoRLM run."""

    video_path: str
    question: str
    subtitle_path: str | None = None
    subtitle_mode: str = "none"
    subtitle_api_model: str = "gpt-4o-transcribe-diarize"
    max_iterations: int = 20


class DocumentRunRequest(BaseModel):
    """Payload to start a document analysis run."""

    document_paths: list[str]
    question: str
    max_iterations: int = 12


class RunResponse(BaseModel):
    """Response after starting a run."""

    run_id: str


class SSEEvent(BaseModel):
    """Shape of a single SSE event delivered to the frontend."""

    kind: str
    timestamp: float
    payload: dict[str, Any]
