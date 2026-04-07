"""Unified Tracer — generic span methods replacing 28 specialized ones."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from .events import EventBuffer
from .observability import configure_logfire, get_logfire

_logfire_configured = False


@dataclass
class TraceContext:
    """Mutable context for recording trace data within a span."""

    _data: dict[str, Any] = field(default_factory=dict)
    _span: Any = None

    def record(self, **kwargs: Any) -> None:
        """Record additional data."""
        self._data.update(kwargs)
        if self._span is not None:
            try:
                self._span.set_attributes(kwargs)
            except Exception:
                pass

    def record_content(self, *, prompt: str | None = None, response: str | None = None) -> None:
        payload: dict[str, Any] = {}
        if prompt is not None:
            payload["prompt_content"] = prompt
            payload["prompt_chars"] = len(prompt)
        if response is not None:
            payload["response_content"] = response
            payload["response_chars"] = len(response)
        if payload:
            self.record(**payload)

    def record_usage(self, *, input_tokens: int | None = None, output_tokens: int | None = None) -> None:
        payload: dict[str, Any] = {}
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if payload:
            self.record(**payload)

    def record_error(self, error: str | Exception | BaseException) -> None:
        error_msg = str(error)
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        self.record(error=error_msg, error_type=error_type, has_error=True)
        if isinstance(error, BaseException) and self._span is not None:
            try:
                self._span.record_exception(error)
            except Exception:
                pass

    def record_response(self, response: str, preview_len: int = 200) -> None:
        preview = response[:preview_len] if len(response) > preview_len else response
        self.record_content(response=response)
        self.record(response_preview=preview)

    def record_duration(self, duration_seconds: float | None) -> None:
        if duration_seconds is not None:
            self.record(duration_seconds=duration_seconds)

    def record_final_answer(self, answer: str, *, forced: bool = False) -> None:
        self.record(final_answer=answer, forced_answer=forced)

    def record_llm_cost(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Compute and set operation.cost using genai_prices."""
        if self._span is None or (not input_tokens and not output_tokens) or not model_name:
            return
        try:
            from genai_prices import calc_price
            from genai_prices.types import Usage

            usage = Usage(input_tokens=input_tokens or 0, output_tokens=output_tokens or 0)
            for model_ref, provider_id in _cost_lookup_candidates(model_name):
                try:
                    result = calc_price(usage, model_ref=model_ref, provider_id=provider_id)
                    self._span.set_attribute("operation.cost", float(result.total_price))
                    self.record(cost_usd=float(result.total_price))
                    return
                except LookupError:
                    continue
        except Exception:
            pass


def _cost_lookup_candidates(model_name: str) -> list[tuple[str, str]]:
    """Generate (model_ref, provider_id) pairs to try with genai_prices."""
    candidates: list[tuple[str, str]] = []
    if "/" in model_name:
        provider_prefix, base_name = model_name.split("/", 1)
        candidates.append((model_name, provider_prefix))
        candidates.append((base_name, provider_prefix))
        candidates.append((model_name, "openrouter"))
        if provider_prefix != "openai":
            candidates.append((base_name, "openai"))
    else:
        candidates.append((model_name, "openai"))
        candidates.append((model_name, "openrouter"))
    return candidates


class Tracer:
    """Unified tracer for logfire spans + in-memory SSE events."""

    def __init__(self, enabled: bool = True, track_events: bool = False):
        self._enabled_requested = enabled
        self._track_events = track_events
        self._event_buffer = EventBuffer()

        global _logfire_configured
        if enabled and not _logfire_configured:
            configure_logfire()
            _logfire_configured = True

    def _logfire(self) -> Any | None:
        return get_logfire(enabled=self._enabled_requested)

    @contextmanager
    def _span(self, name: str, **attrs: Any) -> Generator[TraceContext, None, None]:
        """Generic span creation."""
        ctx = TraceContext()
        ctx.record(**attrs)

        logfire = self._logfire()
        if logfire is not None:
            try:
                with logfire.span(name, **attrs) as span:
                    ctx._span = span
                    if self._track_events:
                        self._event_buffer.emit(f"{name}_start", **attrs)
                    yield ctx
                    if self._track_events:
                        self._event_buffer.emit(f"{name}_end", **ctx._data)
                    return
            except Exception:
                pass

        # Fallback: no logfire
        if self._track_events:
            self._event_buffer.emit(f"{name}_start", **attrs)
        yield ctx
        if self._track_events:
            self._event_buffer.emit(f"{name}_end", **ctx._data)

    # ── Generic span methods ─��──────────────────────────────

    @contextmanager
    def completion(self, *, question: str, model: str, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """Top-level agent.ask() span."""
        with self._span("sanjaya.completion", question=question, model=model, **kwargs) as ctx:
            yield ctx

    @contextmanager
    def iteration(self, *, iteration: int, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """One orchestrator loop iteration."""
        with self._span("sanjaya.iteration", iteration=iteration, **kwargs) as ctx:
            yield ctx

    @contextmanager
    def orchestrator_call(self, *, model: str, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """Root LLM call."""
        with self._span("sanjaya.root_llm_call", model=model, **kwargs) as ctx:
            yield ctx

    @contextmanager
    def code_execution(self, *, code: str, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """REPL code block execution."""
        with self._span("sanjaya.code_execution", code_preview=code[:200], **kwargs) as ctx:
            yield ctx

    @contextmanager
    def tool_call(self, *, tool_name: str, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """Any tool invocation."""
        with self._span("sanjaya.tool_call", tool_name=tool_name, **kwargs) as ctx:
            yield ctx

    @contextmanager
    def llm_call(self, *, model: str, prompt: str, **kwargs: Any) -> Generator[TraceContext, None, None]:
        """Sub-LLM text call (llm_query)."""
        with self._span("sanjaya.sub_llm_call.regular", model=model, prompt_chars=len(prompt), **kwargs) as ctx:
            yield ctx

    # ── Events ──────────────────────────────────────────────

    @property
    def events(self) -> list[dict[str, Any]]:
        """All emitted events, for SSE polling."""
        return self._event_buffer.events

    def emit(self, kind: str, **payload: Any) -> None:
        """Emit a named event."""
        self._event_buffer.emit(kind, **payload)

    def dump_events(self) -> list[dict[str, Any]]:
        """Dump all events (alias for events property)."""
        return self._event_buffer.events

