"""Video-aware Monty REPL wrapper with retrieval and media tooling."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic_monty import MontyRepl
from rich.console import Console

from .tracing import Tracer, get_tracer

from .video_llm import VideoLLMClient
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
        tracer: Tracer | None = None,
    ):
        self.repl = MontyRepl()
        self.sub_llm = VideoLLMClient(
            model=recursive_model,
            vision_model=recursive_model,
            fallback_model=recursive_fallback_model,
        )
        self.context = context
        self.run_id = run_id

        self.workspace = ArtifactWorkspace(base_dir=workspace_base_dir)
        self.mount = WorkspaceMount(str(self.workspace.run_dir))
        self.tracer: Tracer = tracer or get_tracer()

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
        self._visited_window_ids: set[str] = set()
        self._visited_ranges: list[tuple[float, float]] = []

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

        with self.tracer.video_sub_llm_call(
            model=self.sub_llm.text_client.model,
            context=prompt,
            run_id=self.run_id,
        ) as t:
            response = self.sub_llm.completion(prompt, timeout=300)
            self._llm_queries.append((prompt, response))

            usage = self.sub_llm.last_usage
            metadata = self.sub_llm.last_call_metadata or {}
            t.record_response(response)
            t.record_usage(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
            )
            t.record(
                total_tokens=getattr(usage, "total_tokens", None),
                cost_usd=metadata.get("cost_usd"),
                model_used=metadata.get("model_used"),
                provider=metadata.get("provider"),
                duration_seconds=metadata.get("duration_seconds"),
            )
            t.record_llm_cost(
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                model_name=metadata.get("model_used"),
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
        top_k: int | None = None,
        window_size_s: float = 45.0,
        stride_s: float = 30.0,
    ) -> list[dict[str, Any]]:
        """Generate hybrid candidate windows from subtitle + sliding strategies.

        Args:
            top_k: Maximum windows to return.  When *None* (default) the
                   value is auto-scaled with video duration so that longer
                   videos get broader coverage:
                   ``max(8, int(duration_s / 60))``.
            question: Override the question used for subtitle scoring.
            window_size_s: Size of each candidate window in seconds.
            stride_s: Stride for the sliding-window generator.

        Previously-visited windows (those that had clips extracted) are
        automatically excluded so that successive calls surface fresh
        content (progressive scanning).
        """
        query = self._active_query()
        effective_question = question or query.question

        with self.tracer.video_retrieval(run_id=self.run_id) as t:
            duration_s = video_duration_seconds(query.video_path)

            # --- auto-scale top_k with video length ---
            if top_k is None:
                top_k = max(8, int(duration_s / 60))

            _console.print(
                f"[blue]🔎 Phase: retrieval[/] Generating candidate windows "
                f"(top_k={top_k}, window={window_size_s}s, visited={len(self._visited_window_ids)})"
            )

            subtitle_windows: list[CandidateWindow] = []
            if query.subtitle_path:
                # Request extra subtitle windows to compensate for exclusions
                subtitle_top = max(top_k * 2, 24) + len(self._visited_window_ids)
                subtitle_windows = subtitle_anchored_windows(
                    question=effective_question,
                    subtitle_path=query.subtitle_path,
                    window_size_s=window_size_s,
                    top_k=subtitle_top,
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
                exclude_ids=self._visited_window_ids,
                exclude_ranges=self._visited_ranges,
            )

            self._candidate_windows = merged
            self._candidate_by_id = {window.window_id: window for window in merged}
            _console.print(f"[blue]🔎 Phase: retrieval[/] Selected {len(merged)} window(s)")
            self.workspace.record_windows(merged)
            t.record(
                subtitle_count=len(subtitle_windows),
                sliding_count=len(sliding),
                selected_count=len(merged),
                visited_excluded=len(self._visited_window_ids),
                auto_top_k=top_k,
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

        # --- progressive scanning: mark this region as visited ---
        if window_id:
            self._visited_window_ids.add(window_id)
        self._visited_ranges.append((clip_start, clip_end))
        # Also mark any candidate windows that substantially overlap this
        # extraction so they are excluded from future retrieval calls,
        # even when the orchestrator used ad-hoc start_s/end_s.
        for cand in self._candidate_windows:
            if cand.window_id in self._visited_window_ids:
                continue
            overlap_start = max(cand.start_s, clip_start)
            overlap_end = min(cand.end_s, clip_end)
            overlap = max(0.0, overlap_end - overlap_start)
            cand_len = cand.end_s - cand.start_s
            if cand_len > 0 and overlap / cand_len >= 0.5:
                self._visited_window_ids.add(cand.window_id)

        resolved_clip_id = clip_id or window_id or f"clip-{len(self._clips) + 1}"
        _console.print(
            f"[blue]🎞️ Phase: clip[/] Extracting {resolved_clip_id} ({clip_start:.1f}s-{clip_end:.1f}s)"
        )

        with self.tracer.video_clip_extraction(
            clip_id=resolved_clip_id,
            start_s=clip_start,
            end_s=clip_end,
            run_id=self.run_id,
        ) as t:
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
            t.record(clip_path=clip_path)
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
        assert resolved_clip_id is not None
        _console.print(f"[blue]🖼️ Phase: frames[/] Sampling frames for {resolved_clip_id} (max={max_frames})")

        with self.tracer.video_frame_sampling(
            clip_id=resolved_clip_id,
            run_id=self.run_id,
        ) as t:
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
            _console.print(f"[blue]🖼️ Phase: frames[/] Collected {len(frames)} frame(s)")
            t.record(frame_count=len(frames))
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

        effective_prompt = prompt or query.question
        _console.print(
            f"[blue]👁️ Phase: vision[/] Querying vision model (clips={len(collected_clips)}, frames={len(collected_frames)})"
        )

        with self.tracer.video_vision_query(
            prompt=effective_prompt,
            frame_count=len(collected_frames),
            clip_count=len(collected_clips),
            run_id=self.run_id,
        ) as t:
            response = self.sub_llm.vision_completion(
                question=effective_prompt,
                frame_paths=collected_frames,
                clip_paths=collected_clips,
                extra_context=f"video={query.video_path}",
                timeout=300,
            )

            vision_usage = self.sub_llm.last_vision_usage
            vision_metadata = self.sub_llm.last_vision_call_metadata or {}

            t.record(
                response_preview=response[:200],
                input_tokens=getattr(vision_usage, "input_tokens", None),
                output_tokens=getattr(vision_usage, "output_tokens", None),
                total_tokens=getattr(vision_usage, "total_tokens", None),
                cost_usd=vision_metadata.get("cost_usd"),
                model_used=vision_metadata.get("model_used"),
                provider=vision_metadata.get("provider"),
                duration_seconds=vision_metadata.get("duration_seconds"),
            )
            t.record_llm_cost(
                input_tokens=getattr(vision_usage, "input_tokens", None),
                output_tokens=getattr(vision_usage, "output_tokens", None),
                model_name=vision_metadata.get("model_used"),
            )
            _console.print("[blue]👁️ Phase: vision[/] Response received")

        return response

    def get_trace_log(self) -> list[dict[str, Any]]:
        """Return in-memory trace events emitted by retrieval/media/vision tools."""
        return self.tracer.dump_events()

    def persist_trace_events(self) -> None:
        """Persist current trace timeline into workspace manifest."""
        self.workspace.record_trace_events(self.tracer.dump_events())

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
        if code_block_index is not None and code_block_total is not None:
            _console.print(
                f"[blue]🧪 Phase: monty[/] Executing block {code_block_index}/{code_block_total}"
            )
        else:
            _console.print("[blue]🧪 Phase: monty[/] Executing code block")

        with self.tracer.video_code_execution(
            code=code,
            run_id=self.run_id,
            iteration=iteration,
            code_block_index=code_block_index,
            code_block_total=code_block_total,
        ) as t:
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
                t.record_error(exc)
            except Exception as exc:
                stderr = str(exc)
                self._stderr_lines.append(stderr)
                t.record_error(exc)

            execution_time = time.time() - start
            merged_stderr = stderr or "".join(self._stderr_lines)

            t.record(
                execution_time=round(execution_time, 3),
                stderr_preview=merged_stderr[:200],
                has_final_answer=self._is_done,
                llm_queries_count=len(self._llm_queries),
                stdout_preview="".join(self._stdout_lines)[:200],
            )

        _console.print(
            f"[blue]🧪 Phase: monty[/] Done in {execution_time:.2f}s"
            + (" (final answer signaled)" if self._is_done else "")
        )

        return VideoExecutionResult(
            stdout="".join(self._stdout_lines),
            stderr=merged_stderr,
            result=result,
            execution_time=execution_time,
            final_answer=self._final_value if self._is_done else None,
            llm_queries=self._llm_queries.copy(),
        )
