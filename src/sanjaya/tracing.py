"""Typed instrumentation for Sanjaya RLM operations."""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from .observability import get_logfire


@dataclass
class TraceContext:
    """Mutable context for recording trace data within a span."""

    _data: dict[str, Any] = field(default_factory=dict)
    _span: Any = None  # LogfireSpan if available

    def _set_span_attributes(self, attributes: dict[str, Any]) -> None:
        """Set span attributes if a span is attached."""
        if self._span is None:
            return

        try:
            self._span.set_attributes(attributes)
        except Exception:
            pass

    def record(self, **kwargs: Any) -> None:
        """Record additional data to be logged at span end."""
        self._data.update(kwargs)
        self._set_span_attributes(kwargs)

    def record_content(
        self,
        *,
        prompt: str | None = None,
        response: str | None = None,
    ) -> None:
        """Record prompt/response content and character lengths."""
        payload: dict[str, Any] = {}

        if prompt is not None:
            payload["prompt_content"] = prompt
            payload["prompt_chars"] = len(prompt)

        if response is not None:
            payload["response_content"] = response
            payload["response_chars"] = len(response)

        if payload:
            self.record(**payload)

    def record_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        """Record usage metadata."""
        payload: dict[str, Any] = {}
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if payload:
            self.record(**payload)

    def record_duration(self, duration_seconds: float | None) -> None:
        """Record duration metadata."""
        if duration_seconds is None:
            return
        self.record(duration_seconds=duration_seconds)

    def record_final_answer(self, answer: str, *, forced: bool = False) -> None:
        """Record final answer state and content."""
        self.record(final_answer=answer, forced_answer=forced)

    def record_exception(self, error: BaseException) -> None:
        """Record exception details on the active span."""
        if self._span is None:
            return

        try:
            self._span.record_exception(error)
        except Exception:
            pass

    def record_error(self, error: str | Exception | BaseException) -> None:
        """Record structured error metadata."""
        error_msg = str(error)
        error_type = type(error).__name__ if isinstance(error, BaseException) else "Error"
        payload = {
            "error": error_msg,
            "error_type": error_type,
            "has_error": True,
        }

        self._data.update(payload)
        self._set_span_attributes(payload)

        if isinstance(error, BaseException):
            self.record_exception(error)

    def record_response(self, response: str, preview_len: int = 200) -> None:
        """Record a response with automatic preview."""
        preview = response[:preview_len] if len(response) > preview_len else response
        self.record_content(response=response)
        self.record(response_preview=preview)

    def record_llm_cost(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        model_name: str | None = None,
    ) -> None:
        """Compute and set operation.cost on the span using genai_prices.

        OpenRouter returns model names like 'openai/gpt-4.1-mini-2025-04-14'.
        genai_prices can't find these under provider_id='openrouter', so we
        strip the provider prefix and look up under provider_id='openai'.
        """
        if self._span is None:
            return
        if not input_tokens and not output_tokens:
            return
        if not model_name:
            return

        try:
            from genai_prices import calc_price
            from genai_prices.types import Usage

            usage = Usage(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
            )

            # Try with the model name as-is first (works for direct provider calls)
            for model_ref, provider_id in _cost_lookup_candidates(model_name):
                try:
                    result = calc_price(usage, model_ref=model_ref, provider_id=provider_id)
                    self._span.set_attribute("operation.cost", float(result.total_price))
                    self.record(cost_usd=float(result.total_price))
                    return
                except LookupError:
                    continue
        except Exception:
            pass  # Best effort — don't fail tracing over cost calculation


def _cost_lookup_candidates(model_name: str) -> list[tuple[str, str]]:
    """Generate (model_ref, provider_id) pairs to try with genai_prices.

    OpenRouter returns model names like 'openai/gpt-4.1-mini-2025-04-14'.
    We try multiple strategies to find a match in the pricing database.
    """
    candidates: list[tuple[str, str]] = []

    if "/" in model_name:
        # e.g. 'openai/gpt-4.1-mini-2025-04-14'
        provider_prefix, base_name = model_name.split("/", 1)
        # Strategy 1: full name under original provider prefix (e.g. openai)
        candidates.append((model_name, provider_prefix))
        # Strategy 2: base name under the provider prefix
        candidates.append((base_name, provider_prefix))
        # Strategy 3: full name under openrouter
        candidates.append((model_name, "openrouter"))
        # Strategy 4: base name under openai (most reliable for OpenRouter models)
        if provider_prefix != "openai":
            candidates.append((base_name, "openai"))
    else:
        # Direct model name like 'gpt-4.1-mini'
        candidates.append((model_name, "openai"))
        candidates.append((model_name, "openrouter"))

    return candidates


class Tracer:
    """Typed instrumentation for Sanjaya operations."""

    def __init__(self, enabled: bool = True, track_events: bool = False):
        self._enabled_requested = enabled
        self._sub_llm_count = 0
        # In-memory event tracking (opt-in, for workspace manifest persistence)
        self._track_events = track_events
        self._events: list[dict[str, Any]] = []
        self._event_index = 0
        self._context: dict[str, Any] = {}
        self._run_id: str | None = None

    def _logfire(self):
        """Get configured Logfire module or None."""
        return get_logfire(enabled=self._enabled_requested)

    @property
    def enabled(self) -> bool:
        """Whether tracing is currently enabled."""
        return self._logfire() is not None

    def reset_counters(self) -> None:
        """Reset all counters (call at start of new completion)."""
        self._sub_llm_count = 0

    # ── In-memory event tracking ──────────────────────────────────────

    def _emit_event(self, kind: str, **payload: Any) -> None:
        """Record an in-memory trace event (for workspace persistence)."""
        if not self._track_events:
            return
        self._event_index += 1
        event = {
            "kind": kind,
            "timestamp": time.time(),
            "payload": {
                "run_id": self._run_id,
                "event_index": self._event_index,
                **self._context,
                **payload,
            },
        }
        self._events.append(event)

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

    @property
    def events(self) -> list[dict[str, Any]]:
        """Direct access to in-memory events list (for SSE cursor-based polling)."""
        return self._events

    def dump_events(self) -> list[dict[str, Any]]:
        """Return a copy of in-memory trace events (same schema as old VideoTracer.dump())."""
        return list(self._events)

    @contextmanager
    def rlm_completion(
        self,
        query: str,
        context_size: int,
        max_iterations: int,
        orchestrator_model: str,
        sub_llm_model: str,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace the entire RLM completion."""
        ctx = TraceContext()
        self.reset_counters()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "sanjaya.rlm_completion",
            run_id=run_id,
            query=query[:100],
            context_size=context_size,
            max_iterations=max_iterations,
            orchestrator_model=orchestrator_model,
            sub_llm_model=sub_llm_model,
        ) as span:
            ctx._span = span
            logfire_module.info(
                "Starting RLM completion {query=} context={context_size} chars",
                query=query[:100],
                context_size=context_size,
                run_id=run_id,
            )
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                if ctx._data.get("has_error"):
                    logfire_module.error(
                        "RLM completion failed: {error}",
                        error=ctx._data.get("error", "unknown"),
                        run_id=run_id,
                    )
                else:
                    logfire_module.info(
                        "RLM completion finished at iteration {iteration} with answer: {answer}",
                        iteration=ctx._data.get("final_iteration", "?"),
                        answer=ctx._data.get("final_answer", "none"),
                        run_id=run_id,
                    )

    @contextmanager
    def iteration(
        self,
        iteration: int,
        max_iterations: int,
        message_count: int = 0,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace a single RLM iteration."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "sanjaya.rlm_iteration",
            run_id=run_id,
            iteration=iteration,
            max_iterations=max_iterations,
            message_count=message_count,
        ) as span:
            ctx._span = span
            logfire_module.info(
                "Starting iteration {iteration}/{max_iterations} messages={message_count}",
                iteration=iteration,
                max_iterations=max_iterations,
                message_count=message_count,
                run_id=run_id,
            )
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                logfire_module.info(
                    "Iteration {iteration} completed: {code_blocks} code blocks, {sub_llm_calls} sub-LLM calls",
                    iteration=iteration,
                    code_blocks=ctx._data.get("code_blocks_count", 0),
                    sub_llm_calls=ctx._data.get("sub_llm_calls", 0),
                    run_id=run_id,
                )

    @contextmanager
    def orchestrator_call(
        self,
        model: str,
        prompt_chars: int,
        run_id: str | None = None,
        iteration: int | None = None,
        prompt_content: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace an orchestrator LLM call."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "sanjaya.orchestrator_call",
            run_id=run_id,
            iteration=iteration,
            model=model,
            prompt_chars=prompt_chars,
            prompt_content=prompt_content,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                if ctx._data.get("has_error"):
                    logfire_module.error(
                        "Orchestrator call failed: {error}",
                        error=ctx._data.get("error", "unknown"),
                        run_id=run_id,
                        iteration=iteration,
                    )
                else:
                    logfire_module.info(
                        "Orchestrator response {response_chars=}",
                        run_id=run_id,
                        iteration=iteration,
                        **ctx._data,
                    )

    @contextmanager
    def sub_llm_call(
        self,
        model: str,
        context: str,
        preview_len: int = 200,
        run_id: str | None = None,
        iteration: int | None = None,
        code_block_index: int | None = None,
        code_block_total: int | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace a sub-LLM call with automatic numbering."""
        self._sub_llm_count += 1
        call_num = self._sub_llm_count
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        context_preview = context[:preview_len] + "..." if len(context) > preview_len else context
        context_chars = len(context)

        with logfire_module.span(
            "sanjaya.sub_llm_call",
            run_id=run_id,
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            call_num=call_num,
            model=model,
            context_chars=context_chars,
            context_preview=context_preview,
            prompt_content=context,
        ) as span:
            ctx._span = span
            logfire_module.info(
                "Sub-LLM call #{call_num} {model=} context={context_chars} chars\n{context_preview}",
                run_id=run_id,
                iteration=iteration,
                code_block_index=code_block_index,
                code_block_total=code_block_total,
                call_num=call_num,
                model=model,
                context_chars=context_chars,
                context_preview=context_preview,
            )
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                if ctx._data.get("has_error"):
                    logfire_module.error(
                        "Sub-LLM #{call_num} ERROR: {error}",
                        run_id=run_id,
                        iteration=iteration,
                        code_block_index=code_block_index,
                        code_block_total=code_block_total,
                        call_num=call_num,
                        error=ctx._data.get("error", "unknown"),
                    )
                else:
                    logfire_module.info(
                        "Sub-LLM #{call_num} response: {response_chars} chars\n{response_preview}",
                        run_id=run_id,
                        iteration=iteration,
                        code_block_index=code_block_index,
                        code_block_total=code_block_total,
                        call_num=call_num,
                        response_chars=ctx._data.get("response_chars", 0),
                        response_preview=ctx._data.get("response_preview", ""),
                    )

    @contextmanager
    def code_execution(
        self,
        code: str,
        preview_len: int = 500,
        run_id: str | None = None,
        iteration: int | None = None,
        code_block_index: int | None = None,
        code_block_total: int | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace code execution in the REPL."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        code_preview = code[:preview_len] + "..." if len(code) > preview_len else code
        code_chars = len(code)

        with logfire_module.span(
            "sanjaya.code_execution",
            run_id=run_id,
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code_chars=code_chars,
            code_preview=code_preview,
            code_content=code,
        ) as span:
            ctx._span = span
            logfire_module.info(
                "Executing code ({code_chars} chars):\n{code_preview}",
                run_id=run_id,
                iteration=iteration,
                code_block_index=code_block_index,
                code_block_total=code_block_total,
                code_chars=code_chars,
                code_preview=code_preview,
            )
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                if ctx._data.get("has_error"):
                    logfire_module.error(
                        "Code execution FAILED: {error}",
                        run_id=run_id,
                        iteration=iteration,
                        code_block_index=code_block_index,
                        code_block_total=code_block_total,
                        error=ctx._data.get("error", "unknown"),
                    )
                else:
                    logfire_module.info(
                        "Code executed in {execution_time}s, {llm_calls} sub-LLM calls, has_final={has_final}\nOutput: {stdout}",
                        run_id=run_id,
                        iteration=iteration,
                        code_block_index=code_block_index,
                        code_block_total=code_block_total,
                        execution_time=ctx._data.get("execution_time", 0),
                        llm_calls=ctx._data.get("llm_queries_count", 0),
                        has_final=ctx._data.get("has_final_answer", False),
                        stdout=ctx._data.get("stdout_preview", ""),
                    )

    def log_final_answer(self, answer: str, iteration: int, run_id: str | None = None) -> None:
        """Log when final answer is found."""
        logfire_module = self._logfire()
        if logfire_module is None:
            return

        logfire_module.info(
            "✅ FINAL ANSWER at iteration {iteration}: {answer}",
            run_id=run_id,
            iteration=iteration,
            answer=str(answer)[:500],
        )

    def log_max_iterations_reached(self, max_iterations: int, run_id: str | None = None) -> None:
        """Log when max iterations is reached without answer."""
        logfire_module = self._logfire()
        if logfire_module is None:
            return

        logfire_module.error(
            "❌ Max iterations ({max_iterations}) reached without final answer",
            run_id=run_id,
            max_iterations=max_iterations,
        )

    # ── Video-specific span methods ───────────────────────────────────

    @contextmanager
    def video_completion(
        self,
        *,
        video_path: str,
        question: str,
        orchestrator_model: str,
        recursive_model: str,
        max_iterations: int,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace the entire VideoRLM completion run."""
        ctx = TraceContext()
        self.reset_counters()
        self._run_id = run_id

        self._emit_event(
            "run_start",
            video_path=video_path,
            question_preview=question[:300],
            orchestrator_model=orchestrator_model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
        )

        logfire_module = self._logfire()
        if logfire_module is None:
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event("run_end", status=ctx._data.get("completion_status", "unknown"))
            return

        with logfire_module.span(
            "video_rlm.completion",
            run_id=run_id,
            video_path=video_path,
            question=question[:100],
            orchestrator_model=orchestrator_model,
            recursive_model=recursive_model,
            max_iterations=max_iterations,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                self._emit_event("run_end", status="error", error=str(exc))
                raise
            finally:
                status = ctx._data.get("completion_status", "unknown")
                self._emit_event(
                    "run_end",
                    status=status,
                    answer_preview=ctx._data.get("final_answer", "")[:300],
                    total_events=self._event_index,
                )

    @contextmanager
    def video_iteration(
        self,
        *,
        iteration: int,
        max_iterations: int,
        message_count: int = 0,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace a single VideoRLM iteration."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.iteration",
            run_id=run_id,
            iteration=iteration,
            max_iterations=max_iterations,
            message_count=message_count,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise

    @contextmanager
    def video_orchestrator_call(
        self,
        *,
        model: str,
        run_id: str | None = None,
        iteration: int | None = None,
        prompt_chars: int = 0,
    ) -> Generator[TraceContext, None, None]:
        """Trace a VideoRLM orchestrator LLM call."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.orchestrator_call",
            run_id=run_id,
            iteration=iteration,
            model=model,
            prompt_chars=prompt_chars,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "root_response",
                    iteration=iteration,
                    response_chars=ctx._data.get("response_chars", 0),
                    code_blocks_count=ctx._data.get("code_blocks_count", 0),
                    input_tokens=ctx._data.get("input_tokens"),
                    output_tokens=ctx._data.get("output_tokens"),
                    model_used=ctx._data.get("model_used"),
                    duration_seconds=ctx._data.get("duration_seconds"),
                )

    @contextmanager
    def video_code_execution(
        self,
        *,
        code: str,
        run_id: str | None = None,
        iteration: int | None = None,
        code_block_index: int | None = None,
        code_block_total: int | None = None,
        preview_len: int = 400,
    ) -> Generator[TraceContext, None, None]:
        """Trace code execution in the VideoRLM REPL."""
        ctx = TraceContext()
        code_preview = code[:preview_len] + "..." if len(code) > preview_len else code
        code_chars = len(code)

        self._emit_event(
            "code_instruction",
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code_content=code,
            code_preview=code_preview,
            code_chars=code_chars,
        )

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.code_execution",
            run_id=run_id,
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code_chars=code_chars,
            code_preview=code_preview,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "code_execution",
                    iteration=iteration,
                    code_block_index=code_block_index,
                    code_block_total=code_block_total,
                    code_content=code,
                    code_preview=code_preview,
                    code_chars=code_chars,
                    execution_time=ctx._data.get("execution_time", 0),
                    stderr_preview=ctx._data.get("stderr_preview", ""),
                    has_final_answer=ctx._data.get("has_final_answer", False),
                )

    def log_transcription(
        self,
        *,
        source: str | None,
        subtitle_path: str | None,
        generated: bool,
        error: str | None = None,
    ) -> None:
        """Log transcription info event (inside a video_completion span)."""
        self._emit_event(
            "transcription",
            source=source,
            subtitle_path=subtitle_path,
            generated=generated,
            error=error,
        )

        logfire_module = self._logfire()
        if logfire_module is None:
            return

        logfire_module.info(
            "video_rlm.transcription",
            run_id=self._run_id,
            source=source,
            subtitle_path=subtitle_path,
            generated=generated,
            error=error,
        )

    @contextmanager
    def video_retrieval(
        self,
        *,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace candidate window retrieval."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.retrieval",
            run_id=run_id,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "retrieval",
                    subtitle_count=ctx._data.get("subtitle_count", 0),
                    sliding_count=ctx._data.get("sliding_count", 0),
                    selected_count=ctx._data.get("selected_count", 0),
                )

    @contextmanager
    def video_clip_extraction(
        self,
        *,
        clip_id: str,
        start_s: float,
        end_s: float,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace clip extraction from video."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.clip",
            run_id=run_id,
            clip_id=clip_id,
            start_s=start_s,
            end_s=end_s,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "clip",
                    clip_id=clip_id,
                    start_s=start_s,
                    end_s=end_s,
                    clip_path=ctx._data.get("clip_path", ""),
                )

    @contextmanager
    def video_frame_sampling(
        self,
        *,
        clip_id: str,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace frame sampling from a clip."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.frames",
            run_id=run_id,
            clip_id=clip_id,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "frames",
                    clip_id=clip_id,
                    frame_count=ctx._data.get("frame_count", 0),
                )

    @contextmanager
    def video_vision_query(
        self,
        *,
        prompt: str,
        frame_count: int = 0,
        clip_count: int = 0,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace a vision model query (significant latency — needs span for duration)."""
        ctx = TraceContext()

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.vision",
            run_id=run_id,
            prompt=prompt[:200],
            frame_count=frame_count,
            clip_count=clip_count,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "vision",
                    prompt=prompt[:200],
                    frame_count=frame_count,
                    clip_count=clip_count,
                    response_preview=ctx._data.get("response_preview", "")[:200],
                    input_tokens=ctx._data.get("input_tokens"),
                    output_tokens=ctx._data.get("output_tokens"),
                    model_used=ctx._data.get("model_used"),
                    duration_seconds=ctx._data.get("duration_seconds"),
                )

    @contextmanager
    def video_sub_llm_call(
        self,
        *,
        model: str,
        context: str,
        preview_len: int = 240,
        run_id: str | None = None,
    ) -> Generator[TraceContext, None, None]:
        """Trace a sub-LLM call in VideoRLM (with in-memory event)."""
        self._sub_llm_count += 1
        call_num = self._sub_llm_count
        ctx = TraceContext()

        context_preview = context[:preview_len] + "..." if len(context) > preview_len else context
        context_chars = len(context)

        logfire_module = self._logfire()
        if logfire_module is None:
            yield ctx
            return

        with logfire_module.span(
            "video_rlm.sub_llm",
            run_id=run_id,
            call_num=call_num,
            model=model,
            context_chars=context_chars,
            context_preview=context_preview,
        ) as span:
            ctx._span = span
            try:
                yield ctx
            except Exception as exc:
                ctx.record_error(exc)
                raise
            finally:
                self._emit_event(
                    "sub_llm",
                    call_num=call_num,
                    prompt_preview=context_preview,
                    prompt_chars=context_chars,
                    response_preview=ctx._data.get("response_preview", "")[:240],
                    response_chars=ctx._data.get("response_chars", 0),
                    input_tokens=ctx._data.get("input_tokens"),
                    output_tokens=ctx._data.get("output_tokens"),
                    model_used=ctx._data.get("model_used"),
                    duration_seconds=ctx._data.get("duration_seconds"),
                )


# Global tracer instance
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer
