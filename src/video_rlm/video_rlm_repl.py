"""VideoRLM orchestrator loop (separate from text-first sanjaya.RLM_REPL)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from rich.console import Console

from sanjaya.tracing import Tracer, get_tracer

from .llm import VideoLLMClient
from .video_models import VideoAnswer, VideoQuery
from .video_prompts import (
    VIDEO_DEFAULT_QUERY,
    VideoMeta,
    build_video_system_prompt,
    format_transcript_block,
    next_video_action_prompt,
)
from .video_repl import VideoExecutionResult, VideoMontyREPL
from .video_tools.media import video_duration_seconds
from .video_tools.retrieval import load_subtitle_segments
from .video_tools.transcription import ensure_subtitle_sidecar
from .video_utils import (
    build_video_answer,
    compute_video_run_id,
    extract_final_answer,
    find_code_blocks,
    format_execution_feedback,
    infer_subtitle_path,
)

_console = Console()


class VideoRLM_REPL:
    """Recursive orchestrator for long-video question answering."""

    def __init__(
        self,
        model: str = "openrouter:openai/gpt-5.3-codex",
        recursive_model: str = "openrouter:openai/gpt-4.1-mini",
        recursive_fallback_model: str = "openrouter:vikhyatk/moondream2",
        max_iterations: int = 20,
        force_all_iterations: bool = False,
    ):
        self.model = model
        self.recursive_model = recursive_model
        self.recursive_fallback_model = recursive_fallback_model
        self.max_iterations = max_iterations
        self.force_all_iterations = force_all_iterations
        self.llm = VideoLLMClient(model=model)

        # Shared tracer with in-memory event tracking for workspace persistence
        self.tracer: Tracer = get_tracer()
        self.tracer._track_events = True

        self.messages: list[dict[str, str]] = []
        self.query: VideoQuery | None = None
        self.repl: VideoMontyREPL | None = None
        self.last_execution_result: VideoExecutionResult | None = None
        self.run_id: str | None = None

    def setup_query(
        self,
        video_path: str,
        question: str,
        subtitle_path: str | None = None,
        *,
        subtitle_mode: str = "none",
        subtitle_local_model: str = "base",
        subtitle_api_model: str = "gpt-4o-transcribe-diarize",
    ) -> None:
        """Initialize messages and REPL context for a video question."""
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

        resolved_question = question.strip() if question else VIDEO_DEFAULT_QUERY
        inferred_subtitle = infer_subtitle_path(video_path, subtitle_path)

        _console.print(f"[cyan]🧩 Phase: setup[/] Resolving subtitles (mode={subtitle_mode})")
        subtitle_result = ensure_subtitle_sidecar(
            video_path=str(video),
            explicit_subtitle_path=inferred_subtitle or subtitle_path,
            mode=subtitle_mode,
            local_model=subtitle_local_model,
            api_model=subtitle_api_model,
        )

        if subtitle_result.error:
            _console.print(
                "[yellow]Subtitle resolution failed; continuing without subtitles:[/] "
                f"{subtitle_result.error}"
            )

        subtitle_source = subtitle_result.source or "none"
        subtitle_path_display = subtitle_result.subtitle_path or "(none)"
        _console.print(
            f"[cyan]🧩 Phase: setup[/] Subtitle source={subtitle_source}, path={subtitle_path_display}"
        )

        self.query = VideoQuery(
            video_path=str(video),
            question=resolved_question,
            subtitle_path=subtitle_result.subtitle_path,
        )

        # Build video metadata + transcript for the system prompt
        video_meta: VideoMeta | None = None
        try:
            duration_s = video_duration_seconds(str(video))
            segments: list[dict] = []
            if subtitle_result.subtitle_path:
                raw_segs = load_subtitle_segments(subtitle_result.subtitle_path)
                segments = [
                    {"start": s.start_s, "end": s.end_s, "text": s.text}
                    for s in raw_segs
                ]
            if segments:
                transcript_text = format_transcript_block(segments)
                video_meta = VideoMeta(
                    duration_s=duration_s,
                    segment_count=len(segments),
                    transcript_text=transcript_text,
                )
                _console.print(
                    f"[cyan]🧩 Phase: setup[/] Transcript loaded: "
                    f"{len(segments)} segments, {len(transcript_text):,} chars"
                )
            else:
                _console.print("[yellow]🧩 Phase: setup[/] No transcript segments available for prompt")
        except Exception as exc:
            _console.print(f"[yellow]🧩 Phase: setup[/] Could not load video meta: {exc}")

        self.messages = build_video_system_prompt(video_meta=video_meta)
        self.run_id = compute_video_run_id(str(video))
        _console.print(f"[cyan]🧩 Phase: setup[/] run_id={self.run_id}")
        self.repl = VideoMontyREPL(
            recursive_model=self.recursive_model,
            recursive_fallback_model=self.recursive_fallback_model,
            context=self.query.model_dump(),
            run_id=self.run_id,
            tracer=self.tracer,
        )
        self.last_execution_result = None
        self._subtitle_result = subtitle_result

    @staticmethod
    def _normalize_answer_text(answer_payload: object) -> str:
        """Coerce done(value) payloads into a clean final answer string."""

        def mapping_summary(value: dict) -> str | None:
            for key in ("answer", "summary", "final_answer"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
            return None

        def sequence_summary(items: list) -> str | None:
            summaries: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    summary = mapping_summary(item)
                    if summary:
                        summaries.append(summary)
            if summaries:
                return " ".join(summaries)
            return None

        if answer_payload is None:
            return ""

        if isinstance(answer_payload, dict):
            summary = mapping_summary(answer_payload)
            if summary:
                return summary
            return json.dumps(answer_payload, ensure_ascii=False)

        if isinstance(answer_payload, list):
            summary = sequence_summary(answer_payload)
            if summary:
                return summary
            return json.dumps(answer_payload, ensure_ascii=False)

        if isinstance(answer_payload, str):
            text = answer_payload.strip()
            if not text:
                return ""

            parsed: object | None = None
            try:
                parsed = json.loads(text)
            except Exception:
                try:
                    parsed = ast.literal_eval(text)
                except Exception:
                    parsed = None

            if isinstance(parsed, dict):
                summary = mapping_summary(parsed)
                if summary:
                    return summary
            if isinstance(parsed, list):
                summary = sequence_summary(parsed)
                if summary:
                    return summary

            return text

        return str(answer_payload).strip()

    @staticmethod
    def _is_weak_answer(answer_text: str) -> bool:
        """Detect low-quality placeholder answers that should be replaced."""
        text = answer_text.strip()
        if not text:
            return True

        lowered = text.lower()
        weak_markers = (
            "no artifacts captured",
            "no visual evidence",
            "without artifacts",
            "content and context of the video remain unclear",
        )
        return any(marker in lowered for marker in weak_markers)

    def _ensure_minimum_evidence(self, max_clips: int = 3) -> str | None:
        """Run deterministic fallback retrieval over top windows if evidence is missing."""
        assert self.repl is not None
        assert self.query is not None

        manifest = self.repl.get_clip_manifest()
        clips = manifest.get("clips", {})
        if isinstance(clips, dict) and clips:
            return None

        _console.print("[yellow]🧩 Phase: fallback[/] No clips found, extracting minimum evidence...")

        windows = self.repl.list_candidate_windows(top_k=max(5, max_clips))
        if not windows:
            return None

        summaries: list[str] = []
        for window in windows[:max_clips]:
            clip = self.repl.extract_clip(window_id=window["window_id"])
            self.repl.sample_frames(clip_id=clip["clip_id"], max_frames=4)
            vision_summary = self.repl.vision_query(
                clip_id=clip["clip_id"],
                prompt=(
                    f"Question: {self.query.question}\n"
                    f"Focus window: {window['start_s']}s - {window['end_s']}s"
                ),
            )
            summaries.append(f"[{window['start_s']}s-{window['end_s']}s] {vision_summary}")

        return "\n\n".join(summaries) if summaries else None

    def _build_answer(self, answer_payload: object) -> VideoAnswer:
        assert self.repl is not None
        assert self.query is not None

        answer_text = self._normalize_answer_text(answer_payload)

        manifest = self.repl.get_clip_manifest()
        if not manifest.get("clips"):
            fallback = self._ensure_minimum_evidence()
            if fallback and self._is_weak_answer(answer_text):
                answer_text = fallback
            manifest = self.repl.get_clip_manifest()

        return build_video_answer(
            question=self.query.question,
            answer=answer_text,
            manifest=manifest,
        )

    def completion(
        self,
        video_path: str,
        question: str,
        subtitle_path: str | None = None,
        *,
        subtitle_mode: str = "none",
        subtitle_local_model: str = "base",
        subtitle_api_model: str = "gpt-4o-transcribe-diarize",
    ) -> VideoAnswer:
        """Run recursive orchestration and return a structured answer payload."""
        self.setup_query(
            video_path=video_path,
            question=question,
            subtitle_path=subtitle_path,
            subtitle_mode=subtitle_mode,
            subtitle_local_model=subtitle_local_model,
            subtitle_api_model=subtitle_api_model,
        )
        assert self.query is not None
        assert self.repl is not None

        _console.print("[bold]🎬 Starting VideoRLM completion[/]")
        _console.print(
            f"[cyan]🧩 Phase: orchestration[/] model={self.model}, recursive_model={self.recursive_model}"
        )
        tracer = self.tracer

        with tracer.video_completion(
            video_path=self.query.video_path,
            question=self.query.question,
            orchestrator_model=self.model,
            recursive_model=self.recursive_model,
            max_iterations=self.max_iterations,
            run_id=self.run_id,
        ) as completion_ctx:
            try:
                # Log transcription inside the completion span
                tracer.log_transcription(
                    source=self._subtitle_result.source,
                    subtitle_path=self._subtitle_result.subtitle_path,
                    generated=self._subtitle_result.generated,
                    error=self._subtitle_result.error,
                )

                for iteration in range(self.max_iterations):
                    _console.print(
                        f"[cyan]🧩 Phase: iteration[/] {iteration + 1}/{self.max_iterations}"
                    )
                    with tracer.video_iteration(
                        iteration=iteration + 1,
                        max_iterations=self.max_iterations,
                        message_count=len(self.messages),
                        run_id=self.run_id,
                    ) as iter_ctx:
                        orchestrator_messages = self.messages + [
                            next_video_action_prompt(self.query.question, iteration),
                        ]
                        prompt_chars = sum(len(m.get("content", "")) for m in orchestrator_messages)

                        # Orchestrator LLM call inside its own span
                        with tracer.video_orchestrator_call(
                            model=self.model,
                            run_id=self.run_id,
                            iteration=iteration + 1,
                            prompt_chars=prompt_chars,
                        ) as orch_ctx:
                            response = self.llm.completion(orchestrator_messages)
                            self.messages.append({"role": "assistant", "content": response})

                            code_blocks = find_code_blocks(response)
                            _console.print(
                                f"[cyan]🧩 Phase: planning[/] orchestrator returned {len(code_blocks)} code block(s)"
                            )
                            usage = self.llm.last_usage
                            metadata = self.llm.last_call_metadata or {}
                            orch_ctx.record(
                                response_chars=len(response),
                                code_blocks_count=len(code_blocks),
                                input_tokens=getattr(usage, "input_tokens", None),
                                output_tokens=getattr(usage, "output_tokens", None),
                                total_tokens=getattr(usage, "total_tokens", None),
                                cost_usd=metadata.get("cost_usd"),
                                model_used=metadata.get("model_used"),
                                provider=metadata.get("provider"),
                                duration_seconds=metadata.get("duration_seconds"),
                            )
                            orch_ctx.record_llm_cost(
                                input_tokens=getattr(usage, "input_tokens", None),
                                output_tokens=getattr(usage, "output_tokens", None),
                                model_name=metadata.get("model_used"),
                            )

                        # Code execution — each block gets its own span
                        if code_blocks:
                            total = len(code_blocks)
                            for idx, code in enumerate(code_blocks, start=1):
                                _console.print(
                                    f"[cyan]🧩 Phase: execution[/] running code block {idx}/{total}"
                                )
                                self.last_execution_result = self.repl.code_execution(
                                    code,
                                    iteration=iteration + 1,
                                    code_block_index=idx,
                                    code_block_total=total,
                                )
                                feedback = format_execution_feedback(self.last_execution_result, idx, total)
                                self.messages.append({"role": "user", "content": feedback})
                        else:
                            _console.print("[yellow]🧩 Phase: execution[/] No executable code blocks returned")
                            self.messages.append(
                                {"role": "user", "content": f"No executable code found. Response: {response}"}
                            )
                            self.last_execution_result = None

                        final_answer = extract_final_answer(response, self.last_execution_result)
                        if final_answer:
                            _console.print("[green]✅ Phase: finalize[/] Final answer produced")
                            iter_ctx.record(iteration_status="final_answer")
                            answer = self._build_answer(final_answer)
                            completion_ctx.record(completion_status="final_answer")
                            completion_ctx.record_final_answer(str(answer.answer), forced=False)
                            self.repl.persist_trace_events()
                            return answer
                        else:
                            iter_ctx.record(iteration_status="continue")

                # Max iterations reached — force final answer
                _console.print("[yellow]🧩 Phase: finalize[/] Max iterations reached, forcing best-effort answer")
                forced_messages = self.messages + [
                    next_video_action_prompt(self.query.question, self.max_iterations - 1, final_answer=True)
                ]
                forced_answer = self.llm.completion(forced_messages)
                answer = self._build_answer(forced_answer)
                completion_ctx.record(completion_status="forced_final_answer")
                completion_ctx.record_final_answer(str(answer.answer), forced=True)
                self.repl.persist_trace_events()
                return answer
            except KeyboardInterrupt as exc:
                completion_ctx.record(completion_status="interrupted")
                completion_ctx.record_error(exc)
                self.repl.persist_trace_events()
                _console.print("[yellow]Interrupted by user (KeyboardInterrupt). Trace persisted.[/]")
                raise
            except Exception as exc:
                completion_ctx.record(completion_status="error")
                completion_ctx.record_error(exc)
                self.repl.persist_trace_events()
                raise

    def completion_from_query(self, query: VideoQuery) -> VideoAnswer:
        """Convenience wrapper for callers that already have a VideoQuery object."""
        return self.completion(
            video_path=query.video_path,
            question=query.question,
            subtitle_path=query.subtitle_path,
        )
