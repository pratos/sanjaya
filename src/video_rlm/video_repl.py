"""Video-aware Monty REPL wrapper with retrieval and media tooling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_monty import MontyRepl
from rich.console import Console

from .llm import VideoLLMClient
from .tracing import VideoTracer
from .video_models import CandidateWindow, ClipArtifact, VideoQuery
from .video_tools.media import MediaToolError, extract_clip, sample_frames, video_duration_seconds
from .video_tools.monty_mount import WorkspaceMount
from .video_tools.retrieval import hybrid_merge, sliding_windows, subtitle_anchored_windows
from .video_tools.workspace import ArtifactWorkspace

_console = Console()


@dataclass
class VideoExecutionResult:
    """Result of one Monty code execution."""

    stdout: str
    stderr: str
    result: Any
    execution_time: float
    final_answer: Any | None = None
    llm_queries: list[tuple[str, str]] = field(default_factory=list)


class VideoMontyREPL:
    """Monty REPL environment that supports video-specific external tools."""

    def __init__(
        self,
        recursive_model: str = "openrouter:openai/gpt-4.1-mini",
        recursive_fallback_model: str = "openrouter:vikhyatk/moondream2",
        context: Any = None,
        workspace_base_dir: str = "data/longvideobench/artifacts",
        run_id: str | None = None,
    ):
        self.repl = MontyRepl()
        self.sub_llm = VideoLLMClient(
            model=recursive_model,
            vision_model=recursive_model,
            fallback_model=recursive_fallback_model,
        )
        self.context = context

        self.workspace = ArtifactWorkspace(base_dir=workspace_base_dir)
        self.mount = WorkspaceMount(str(self.workspace.run_dir))
        self.tracer = VideoTracer(run_id=run_id)

        self._external_functions: dict[str, Callable[..., Any]] = {}
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._final_value: Any | None = None
        self._is_done = False
        self._llm_queries: list[tuple[str, str]] = []
        self._os_access: Any | None = None

        self._candidate_windows: list[CandidateWindow] = []
        self._candidate_by_id: dict[str, CandidateWindow] = {}
        self._clips: dict[str, ClipArtifact] = {}

        self.register_external_function("list_candidate_windows", self.list_candidate_windows)
        self.register_external_function("extract_clip", self.extract_clip)
        self.register_external_function("sample_frames", self.sample_frames)
        self.register_external_function("get_clip_manifest", self.get_clip_manifest)
        self.register_external_function("vision_query", self.vision_query)
        self.register_external_function("get_trace_log", self.get_trace_log)

        self._refresh_mount()

    def _active_query(self) -> VideoQuery:
        if isinstance(self.context, VideoQuery):
            return self.context
        if isinstance(self.context, dict):
            return VideoQuery.model_validate(self.context)
        raise ValueError("VideoMontyREPL context must be VideoQuery or dict")

    def register_external_function(self, name: str, func: Callable[..., Any]) -> None:
        """Register a custom external function callable from Monty code."""
        self._external_functions[name] = func

    def set_os_access(self, os_access: Any | None) -> None:
        """Attach optional Monty OS access adapter used during execution."""
        self._os_access = os_access

    def _refresh_mount(self) -> None:
        self.set_os_access(self.mount.build_os_access())

    def _reset_state(self) -> None:
        self._stdout_lines = []
        self._stderr_lines = []
        self._final_value = None
        self._is_done = False
        self._llm_queries = []

    def _print_callback(self, stream: str, text: str) -> None:
        if stream == "stdout":
            self._stdout_lines.append(text)
        else:
            self._stderr_lines.append(text)

    def _get_context(self) -> Any:
        return self.context

    def _llm_query(self, prompt: str) -> str:
        _console.print(f"[magenta]🤖 llm_query called ({len(prompt)} chars)[/]")
        response = self.sub_llm.completion(prompt, timeout=300)
        self._llm_queries.append((prompt, response))

        usage = self.sub_llm.last_usage
        metadata = self.sub_llm.last_call_metadata or {}
        self.tracer.record_sub_llm_call(
            prompt=prompt,
            response=response,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            cost_usd=metadata.get("cost_usd"),
            model_used=metadata.get("model_used"),
            provider=metadata.get("provider"),
            duration_seconds=metadata.get("duration_seconds"),
        )
        return response

    def _done(self, value: Any) -> Any:
        self._is_done = True
        self._final_value = value
        return value

    def list_candidate_windows(
        self,
        *,
        question: str | None = None,
        top_k: int = 8,
        window_size_s: float = 45.0,
        stride_s: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Generate hybrid candidate windows from subtitle + sliding strategies."""
        query = self._active_query()
        effective_question = question or query.question

        duration_s = video_duration_seconds(query.video_path)

        subtitle_windows: list[CandidateWindow] = []
        if query.subtitle_path:
            subtitle_windows = subtitle_anchored_windows(
                question=effective_question,
                subtitle_path=query.subtitle_path,
                window_size_s=window_size_s,
                top_k=max(top_k, 12),
            )

        sliding = sliding_windows(
            duration_s=duration_s,
            window_size_s=window_size_s,
            stride_s=stride_s,
        )

        merged = hybrid_merge(
            subtitle_windows=subtitle_windows,
            sliding=sliding,
            top_k=top_k,
        )

        self._candidate_windows = merged
        self._candidate_by_id = {window.window_id: window for window in merged}
        self.workspace.record_windows(merged)
        self.tracer.record_retrieval(
            subtitle_count=len(subtitle_windows),
            sliding_count=len(sliding),
            selected_count=len(merged),
        )
        self._refresh_mount()

        return [window.model_dump() for window in merged]

    def extract_clip(
        self,
        *,
        window_id: str | None = None,
        start_s: float | None = None,
        end_s: float | None = None,
        clip_id: str | None = None,
    ) -> dict[str, Any]:
        """Extract a clip from a selected candidate window or explicit timestamps."""
        query = self._active_query()

        selected_window = self._candidate_by_id.get(window_id or "") if window_id else None
        clip_start = float(start_s) if start_s is not None else (selected_window.start_s if selected_window else None)
        clip_end = float(end_s) if end_s is not None else (selected_window.end_s if selected_window else None)

        if clip_start is None or clip_end is None:
            raise ValueError("extract_clip requires window_id or explicit start_s/end_s")

        resolved_clip_id = clip_id or window_id or f"clip-{len(self._clips) + 1}"
        output_path = self.workspace.clip_path(resolved_clip_id)

        clip_path = extract_clip(
            video_path=query.video_path,
            start_s=clip_start,
            end_s=clip_end,
            output_path=str(output_path),
        )

        artifact = ClipArtifact(
            clip_id=resolved_clip_id,
            clip_path=clip_path,
            start_s=clip_start,
            end_s=clip_end,
            frame_paths=[],
        )

        self._clips[resolved_clip_id] = artifact
        self.workspace.record_clip(artifact, window_id=window_id)
        self.tracer.record_clip(
            clip_id=resolved_clip_id,
            start_s=clip_start,
            end_s=clip_end,
            clip_path=clip_path,
        )
        self._refresh_mount()
        return artifact.model_dump()

    def sample_frames(
        self,
        *,
        clip_id: str | None = None,
        clip_path: str | None = None,
        max_frames: int = 8,
    ) -> list[str]:
        """Sample frames from an extracted clip."""
        resolved_clip_id = clip_id
        resolved_path = clip_path

        if resolved_clip_id:
            artifact = self._clips.get(resolved_clip_id)
            if artifact is None:
                raise ValueError(f"Unknown clip_id: {resolved_clip_id}")
            resolved_path = artifact.clip_path
        elif resolved_path:
            resolved_clip_id = f"clip-{len(self._clips) + 1}"
        else:
            raise ValueError("sample_frames requires clip_id or clip_path")

        assert resolved_path is not None
        duration = video_duration_seconds(resolved_path)
        frame_dir = self.workspace.frame_dir(resolved_clip_id)

        frames = sample_frames(
            video_path=resolved_path,
            start_s=0.0,
            end_s=duration,
            output_dir=str(frame_dir),
            max_frames=max_frames,
        )

        if resolved_clip_id in self._clips:
            self._clips[resolved_clip_id].frame_paths = frames
        self.workspace.update_frames(resolved_clip_id, frames)
        self.tracer.record_frames(
            clip_id=resolved_clip_id,
            frame_count=len(frames),
        )
        self._refresh_mount()
        return frames

    def vision_query(
        self,
        *,
        prompt: str | None = None,
        clip_id: str | None = None,
        frame_paths: list[str] | None = None,
        clip_paths: list[str] | None = None,
    ) -> str:
        """Query a multimodal-safe model using selected clips/frames."""
        query = self._active_query()

        collected_frames = list(frame_paths or [])
        collected_clips = list(clip_paths or [])

        if clip_id:
            artifact = self._clips.get(clip_id)
            if artifact is None:
                raise ValueError(f"Unknown clip_id: {clip_id}")
            collected_clips.append(artifact.clip_path)
            collected_frames.extend(artifact.frame_paths)

        if not collected_frames and not collected_clips and self._clips:
            latest = next(reversed(self._clips.values()))
            collected_clips.append(latest.clip_path)
            collected_frames.extend(latest.frame_paths)

        response = self.sub_llm.vision_completion(
            question=prompt or query.question,
            frame_paths=collected_frames,
            clip_paths=collected_clips,
            extra_context=f"video={query.video_path}",
            timeout=300,
        )

        vision_usage = self.sub_llm.last_vision_usage
        vision_metadata = self.sub_llm.last_vision_call_metadata or {}

        self.tracer.record_vision(
            prompt=prompt or query.question,
            frame_count=len(collected_frames),
            clip_count=len(collected_clips),
            response_preview=response,
            input_tokens=getattr(vision_usage, "input_tokens", None),
            output_tokens=getattr(vision_usage, "output_tokens", None),
            total_tokens=getattr(vision_usage, "total_tokens", None),
            cost_usd=vision_metadata.get("cost_usd"),
            model_used=vision_metadata.get("model_used"),
            provider=vision_metadata.get("provider"),
            duration_seconds=vision_metadata.get("duration_seconds"),
        )
        return response

    def get_trace_log(self) -> list[dict[str, Any]]:
        """Return in-memory trace events emitted by retrieval/media/vision tools."""
        return self.tracer.dump()

    def persist_trace_events(self) -> None:
        """Persist current trace timeline into workspace manifest."""
        self.workspace.record_trace_events(self.tracer.dump())

    def get_clip_manifest(self) -> dict[str, Any]:
        """Return workspace manifest with windows and artifact records."""
        self.persist_trace_events()
        manifest = self.workspace.load_manifest()
        self._refresh_mount()
        return manifest

    def code_execution(
        self,
        code: str,
        *,
        iteration: int | None = None,
        code_block_index: int | None = None,
        code_block_total: int | None = None,
    ) -> VideoExecutionResult:
        """Run one code block inside Monty with core + registered external tools."""
        self._reset_state()
        self._refresh_mount()
        self.tracer.set_context(
            phase="code_execution",
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
        )

        start = time.time()
        result: Any = None
        stderr = ""

        external_functions = {
            "get_context": self._get_context,
            "llm_query": self._llm_query,
            "done": self._done,
            **self._external_functions,
        }

        try:
            result = self.repl.feed_run(
                code,
                external_functions=external_functions,
                print_callback=self._print_callback,
                os=self._os_access,
            )
        except (MediaToolError, FileNotFoundError, ValueError) as exc:
            stderr = str(exc)
            self._stderr_lines.append(stderr)
        except Exception as exc:
            stderr = str(exc)
            self._stderr_lines.append(stderr)

        execution_time = time.time() - start
        merged_stderr = stderr or "".join(self._stderr_lines)

        self.tracer.record_code_execution(
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
            code=code,
            execution_time=execution_time,
            stderr=merged_stderr,
            has_final_answer=self._is_done,
        )
        self.tracer.clear_context("phase", "iteration", "code_block_index", "code_block_total")

        return VideoExecutionResult(
            stdout="".join(self._stdout_lines),
            stderr=merged_stderr,
            result=result,
            execution_time=execution_time,
            final_answer=self._final_value if self._is_done else None,
            llm_queries=self._llm_queries.copy(),
        )
