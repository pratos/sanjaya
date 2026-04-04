"""Tracing helpers for VideoRLM orchestration, code execution, and tool chains."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from sanjaya.observability import get_logfire


@dataclass
class VideoTraceEvent:
    """Single trace event emitted by VideoRLM."""

    kind: str
    timestamp: float
    payload: dict[str, Any] = field(default_factory=dict)


class VideoTracer:
    """Lightweight tracer with in-memory events and optional Logfire logging."""

    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or uuid4().hex[:12]
        self.events: list[VideoTraceEvent] = []
        self._event_index = 0
        self._context: dict[str, Any] = {}

    def set_context(self, **context: Any) -> None:
        """Set correlation context added to every emitted event."""
        self._context.update(context)

    def clear_context(self, *keys: str) -> None:
        """Clear all context or selected keys."""
        if not keys:
            self._context = {}
            return
        for key in keys:
            self._context.pop(key, None)

    def _emit(self, kind: str, **payload: Any) -> None:
        self._event_index += 1
        merged_payload = {
            "run_id": self.run_id,
            "event_index": self._event_index,
            **self._context,
            **payload,
        }

        event = VideoTraceEvent(kind=kind, timestamp=time.time(), payload=merged_payload)
        self.events.append(event)

        logfire_module = get_logfire(enabled=True)
        if logfire_module is not None:
            logfire_module.info(
                "video_rlm.{kind}",
                kind=kind,
                **merged_payload,
            )

    def record_run_start(
        self,
        *,
        video_path: str,
        question: str,
        orchestrator_model: str,
        recursive_model: str,
        max_iterations: int,
    ) -> None:
        self._emit(
            "run_start",
            video_path=video_path,
            question_preview=question[:300],
            orchestrator_model=orchestrator_model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
        )

    def record_run_end(self, *, status: str, answer_preview: str | None = None) -> None:
        self._emit(
            "run_end",
            status=status,
            answer_preview=(answer_preview or "")[:300],
            total_events=self._event_index,
        )

    def record_transcription(
        self,
        *,
        source: str | None,
        subtitle_path: str | None,
        generated: bool,
        error: str | None = None,
    ) -> None:
        self._emit(
            "transcription",
            source=source,
            subtitle_path=subtitle_path,
            generated=generated,
            error=error,
        )

    def record_root_response(
        self,
        *,
        iteration: int,
        response: str,
        code_blocks_count: int,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        model_used: str | None = None,
        provider: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._emit(
            "root_response",
            iteration=iteration,
            response_preview=response[:300],
            response_chars=len(response),
            code_blocks_count=code_blocks_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model_used=model_used,
            provider=provider,
            duration_seconds=duration_seconds,
        )

    def record_sub_llm_call(
        self,
        *,
        prompt: str,
        response: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        model_used: str | None = None,
        provider: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._emit(
            "sub_llm",
            prompt_preview=prompt[:240],
            response_preview=response[:240],
            prompt_chars=len(prompt),
            response_chars=len(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model_used=model_used,
            provider=provider,
            duration_seconds=duration_seconds,
        )

    def record_code_instruction(
        self,
        *,
        iteration: int,
        code_block_index: int,
        code_block_total: int,
        code: str,
    ) -> None:
        self._emit(
            "code_instruction",
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code_content=code,
            code_preview=code[:400],
            code_chars=len(code),
        )

    def record_code_execution(
        self,
        *,
        iteration: int | None,
        code_block_index: int | None,
        code_block_total: int | None,
        code: str,
        execution_time: float,
        stderr: str,
        has_final_answer: bool,
    ) -> None:
        self._emit(
            "code_execution",
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code_content=code,
            code_preview=code[:400],
            code_chars=len(code),
            execution_time=round(execution_time, 3),
            stderr_preview=stderr[:200],
            has_final_answer=has_final_answer,
        )

    def record_retrieval(self, *, subtitle_count: int, sliding_count: int, selected_count: int) -> None:
        self._emit(
            "retrieval",
            subtitle_count=subtitle_count,
            sliding_count=sliding_count,
            selected_count=selected_count,
        )

    def record_clip(self, *, clip_id: str, start_s: float, end_s: float, clip_path: str) -> None:
        self._emit(
            "clip",
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
            clip_path=clip_path,
        )

    def record_frames(self, *, clip_id: str, frame_count: int) -> None:
        self._emit(
            "frames",
            clip_id=clip_id,
            frame_count=frame_count,
        )

    def record_vision(
        self,
        *,
        prompt: str,
        frame_count: int,
        clip_count: int,
        response_preview: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_usd: float | None = None,
        model_used: str | None = None,
        provider: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        self._emit(
            "vision",
            prompt=prompt[:200],
            frame_count=frame_count,
            clip_count=clip_count,
            response_preview=response_preview[:200],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            model_used=model_used,
            provider=provider,
            duration_seconds=duration_seconds,
        )

    def dump(self) -> list[dict[str, Any]]:
        return [
            {
                "kind": event.kind,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
            for event in self.events
        ]
