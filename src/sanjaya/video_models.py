"""Data models for VideoRLM query, retrieval, and evidence outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoQuery(BaseModel):
    """Input contract for a single video question answering request."""

    video_path: str = Field(description="Local path to the source video file")
    question: str = Field(description="Open-ended user question")
    subtitle_path: str | None = Field(
        default=None,
        description="Optional subtitle sidecar path (e.g., *_en.json)",
    )


class CandidateWindow(BaseModel):
    """A candidate temporal window proposed by retrieval."""

    window_id: str = Field(description="Stable window identifier")
    strategy: str = Field(description="Retrieval strategy name")
    start_s: float = Field(ge=0, description="Start timestamp in seconds")
    end_s: float = Field(gt=0, description="End timestamp in seconds")
    score: float = Field(default=0.0, description="Combined ranking score")
    reason: str | None = Field(default=None, description="Why this window was proposed")


class ClipArtifact(BaseModel):
    """Generated media artifact for a candidate or selected window."""

    clip_id: str = Field(description="Stable clip identifier")
    clip_path: str = Field(description="Local filesystem path to extracted clip")
    start_s: float = Field(ge=0, description="Clip start timestamp in seconds")
    end_s: float = Field(gt=0, description="Clip end timestamp in seconds")
    frame_paths: list[str] = Field(default_factory=list, description="Sampled frame paths")


class EvidenceItem(BaseModel):
    """Evidence unit supporting the final answer."""

    window_id: str | None = Field(default=None, description="Source retrieval window id")
    start_s: float = Field(ge=0, description="Evidence start timestamp")
    end_s: float = Field(gt=0, description="Evidence end timestamp")
    rationale: str = Field(description="Why this evidence supports the answer")
    clip_path: str | None = Field(default=None, description="Supporting clip path")
    frame_paths: list[str] = Field(default_factory=list, description="Supporting frame paths")


class VideoAnswer(BaseModel):
    """Structured answer payload for VideoRLM output."""

    question: str
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    retrieval_trace: list[CandidateWindow] = Field(default_factory=list)
