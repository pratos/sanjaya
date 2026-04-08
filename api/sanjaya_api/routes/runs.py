"""Run management endpoints: start runs and stream SSE events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from sanjaya_api.models import RunRequest, RunResponse
from sanjaya_api.services.orchestrator import OrchestratorService
from sanjaya_api.sse import format_heartbeat, format_sse_event

router = APIRouter()

# Singleton orchestrator service (lives for the lifetime of the process)
_orchestrator = OrchestratorService()

# Map internal tracer event kinds to frontend-expected SSE event names.
# The tracer emits span pairs like "sanjaya.iteration_start" / "_end".
# The frontend listens for simplified names like "root_response", "code_execution", etc.
_KIND_MAP: dict[str, str] = {
    "sanjaya.completion_start": "run_start",
    "sanjaya.completion_end": "run_end",
    "sanjaya.iteration_start": "iteration_start",
    "sanjaya.iteration_end": "iteration_end",
    "sanjaya.root_llm_call_start": "root_response_start",
    "sanjaya.root_llm_call_end": "root_response",
    "sanjaya.code_execution_start": "code_instruction",
    "sanjaya.code_execution_end": "code_execution",
    "sanjaya.tool_call_start": "tool_call_start",
    "sanjaya.tool_call_end": "tool_call",
    "sanjaya.sub_llm_call.regular_start": "sub_llm_start",
    "sanjaya.sub_llm_call.regular_end": "sub_llm",
    "sanjaya.sub_llm_call.vision_start": "vision_start",
    "sanjaya.sub_llm_call.vision_end": "vision",
    "sanjaya.sub_llm_call.caption_frames_start": "vision_start",
    "sanjaya.sub_llm_call.caption_frames_end": "vision",
    "sanjaya.schema_generation_start": "schema_generation_start",
    "sanjaya.schema_generation_end": "schema_generation",
    "sanjaya.critic_evaluation": "critic_evaluation",
}


def _normalize_event(raw: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    """Extract kind, timestamp, and payload from a flat EventBuffer dict.

    EventBuffer stores ``{kind, timestamp, **payload_kwargs}`` — the payload
    keys are spread into the dict.  This function separates them out and maps
    the internal kind name to a frontend-friendly name.
    """
    internal_kind: str = raw.get("kind", "unknown") or "unknown"
    timestamp: float = raw.get("timestamp", 0.0) or 0.0
    payload = {k: v for k, v in raw.items() if k not in ("kind", "timestamp")}
    frontend_kind = _KIND_MAP.get(internal_kind, internal_kind)
    return frontend_kind, timestamp, payload


@router.post("/runs", response_model=RunResponse)
async def start_run(request: RunRequest) -> RunResponse:
    """Start a new VideoRLM orchestration run."""
    run_id = _orchestrator.start_run(
        video_path=request.video_path,
        question=request.question,
        subtitle_path=request.subtitle_path,
        subtitle_mode=request.subtitle_mode,
        subtitle_api_model=request.subtitle_api_model,
        max_iterations=request.max_iterations,
    )
    return RunResponse(run_id=run_id)


@router.get("/runs/{run_id}/events")
async def stream_events(run_id: str) -> EventSourceResponse:
    """Stream trace events for a run via Server-Sent Events."""
    record = _orchestrator.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        cursor = 0
        heartbeat_interval = 2.0  # seconds
        poll_interval = 0.25  # seconds
        time_since_heartbeat = 0.0

        while True:
            tracer = record.tracer
            if tracer is not None:
                events = tracer.events
                # Yield any new events since our cursor
                while cursor < len(events):
                    raw = events[cursor]
                    cursor += 1
                    kind, timestamp, payload = _normalize_event(raw)
                    yield format_sse_event(
                        kind=kind,
                        timestamp=timestamp,
                        payload=payload,
                    )

            # Check if run is finished
            if record.status in ("complete", "error"):
                # Drain any remaining events
                if tracer is not None:
                    events = tracer.events
                    while cursor < len(events):
                        raw = events[cursor]
                        cursor += 1
                        kind, timestamp, payload = _normalize_event(raw)
                        yield format_sse_event(
                            kind=kind,
                            timestamp=timestamp,
                            payload=payload,
                        )
                # Send terminal status event
                if record.status == "error":
                    yield format_sse_event(
                        kind="stream_error",
                        timestamp=0,
                        payload={"error": record.error or "Unknown error"},
                    )
                yield format_sse_event(
                    kind="stream_end",
                    timestamp=0,
                    payload={"status": record.status},
                )
                return

            # Heartbeat
            time_since_heartbeat += poll_interval
            if time_since_heartbeat >= heartbeat_interval:
                yield format_heartbeat()
                time_since_heartbeat = 0.0

            await asyncio.sleep(poll_interval)

    return EventSourceResponse(event_generator())
