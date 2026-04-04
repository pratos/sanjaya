"""VideoRLM orchestrator loop (separate from text-first sanjaya.RLM_REPL)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .llm import VideoLLMClient
from .video_models import VideoAnswer, VideoQuery
from .video_prompts import VIDEO_DEFAULT_QUERY, build_video_system_prompt, next_video_action_prompt
from .video_repl import VideoExecutionResult, VideoMontyREPL
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
    ):
        self.model = model
        self.recursive_model = recursive_model
        self.recursive_fallback_model = recursive_fallback_model
        self.max_iterations = max_iterations
        self.llm = VideoLLMClient(model=model)

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
        subtitle_mode: str = "auto",
        subtitle_local_model: str = "base",
        subtitle_api_model: str = "gpt-4o-mini-transcribe",
    ) -> None:
        """Initialize messages and REPL context for a video question."""
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video file not found: {video}")

        resolved_question = question.strip() if question else VIDEO_DEFAULT_QUERY
        inferred_subtitle = infer_subtitle_path(video_path, subtitle_path)

        subtitle_result = ensure_subtitle_sidecar(
            video_path=str(video),
            explicit_subtitle_path=inferred_subtitle or subtitle_path,
            mode=subtitle_mode,
            local_model=subtitle_local_model,
            api_model=subtitle_api_model,
        )

        self.query = VideoQuery(
            video_path=str(video),
            question=resolved_question,
            subtitle_path=subtitle_result.subtitle_path,
        )
        self.messages = build_video_system_prompt()
        self.run_id = compute_video_run_id(str(video))
        self.repl = VideoMontyREPL(
            recursive_model=self.recursive_model,
            recursive_fallback_model=self.recursive_fallback_model,
            context=self.query.model_dump(),
            run_id=self.run_id,
        )
        self.last_execution_result = None

        self.repl.tracer.record_run_start(
            video_path=self.query.video_path,
            question=self.query.question,
            orchestrator_model=self.model,
            recursive_model=self.recursive_model,
            max_iterations=self.max_iterations,
        )
        self.repl.tracer.record_transcription(
            source=subtitle_result.source,
            subtitle_path=subtitle_result.subtitle_path,
            generated=subtitle_result.generated,
            error=subtitle_result.error,
        )

    def _ensure_minimum_evidence(self, max_clips: int = 3) -> str | None:
        """Run deterministic fallback retrieval over top windows if evidence is missing."""
        assert self.repl is not None
        assert self.query is not None

        manifest = self.repl.get_clip_manifest()
        clips = manifest.get("clips", {})
        if isinstance(clips, dict) and clips:
            return None

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

    def _build_answer(self, answer_text: str) -> VideoAnswer:
        assert self.repl is not None
        assert self.query is not None

        manifest = self.repl.get_clip_manifest()
        if not manifest.get("clips"):
            fallback = self._ensure_minimum_evidence()
            if fallback and (not answer_text or "I could not" in answer_text):
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
        subtitle_mode: str = "auto",
        subtitle_local_model: str = "base",
        subtitle_api_model: str = "gpt-4o-mini-transcribe",
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

        try:
            for iteration in range(self.max_iterations):
                orchestrator_messages = self.messages + [
                    next_video_action_prompt(self.query.question, iteration),
                ]
                self.repl.tracer.set_context(phase="orchestrator", iteration=iteration + 1)
                response = self.llm.completion(orchestrator_messages)
                self.messages.append({"role": "assistant", "content": response})

                code_blocks = find_code_blocks(response)
                usage = self.llm.last_usage
                metadata = self.llm.last_call_metadata or {}
                self.repl.tracer.record_root_response(
                    iteration=iteration + 1,
                    response=response,
                    code_blocks_count=len(code_blocks),
                    input_tokens=getattr(usage, "input_tokens", None),
                    output_tokens=getattr(usage, "output_tokens", None),
                    total_tokens=getattr(usage, "total_tokens", None),
                    cost_usd=metadata.get("cost_usd"),
                    model_used=metadata.get("model_used"),
                    provider=metadata.get("provider"),
                    duration_seconds=metadata.get("duration_seconds"),
                )

                if code_blocks:
                    total = len(code_blocks)
                    for idx, code in enumerate(code_blocks, start=1):
                        self.repl.tracer.record_code_instruction(
                            iteration=iteration + 1,
                            code_block_index=idx,
                            code_block_total=total,
                            code=code,
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
                    self.messages.append({"role": "user", "content": f"No executable code found. Response: {response}"})
                    self.last_execution_result = None

                final_answer = extract_final_answer(response, self.last_execution_result)
                self.repl.tracer.clear_context("phase", "iteration", "code_block_index", "code_block_total")
                if final_answer:
                    answer = self._build_answer(final_answer)
                    self.repl.tracer.record_run_end(status="final_answer", answer_preview=answer.answer)
                    self.repl.persist_trace_events()
                    return answer

            forced_messages = self.messages + [
                next_video_action_prompt(self.query.question, self.max_iterations - 1, final_answer=True)
            ]
            forced_answer = self.llm.completion(forced_messages)
            answer = self._build_answer(forced_answer)
            self.repl.tracer.record_run_end(status="forced_final_answer", answer_preview=answer.answer)
            self.repl.persist_trace_events()
            return answer
        except Exception as exc:
            self.repl.tracer.record_run_end(status="error", answer_preview=str(exc))
            self.repl.persist_trace_events()
            raise

    def completion_from_query(self, query: VideoQuery) -> VideoAnswer:
        """Convenience wrapper for callers that already have a VideoQuery object."""
        return self.completion(
            video_path=query.video_path,
            question=query.question,
            subtitle_path=query.subtitle_path,
        )
