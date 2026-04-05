"""VideoToolkit — complete video analysis toolkit for the agent."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from ...answer import Evidence
from ..base import Tool, Toolkit, ToolParam
from .media import extract_clip as _extract_clip_impl
from .media import get_video_info, video_duration_seconds
from .media import sample_frames as _sample_frames_impl
from .mount import WorkspaceMount
from .retrieval import hybrid_merge, sliding_windows, subtitle_anchored_windows
from .transcription import ensure_subtitle_sidecar
from .vision import make_vision_query_batched_fn, make_vision_query_fn
from .workspace import ArtifactWorkspace

_console = Console()

_VIDEO_STRATEGY_PROMPT = """\
## Video Analysis Strategy

You are analyzing a video. Here's the recommended workflow:

1. **Understand the video**: Call `get_video_info()` to get duration, resolution, etc.
2. **Find relevant segments**: Call `list_windows(question=...)` to get ranked temporal windows.
3. **Extract and examine clips**: Use `extract_clip(window_id=...)` then `sample_frames(clip_id=...)`.
4. **Analyze visually**: Use `vision_query(clip_id=..., prompt=...)` for visual analysis.
5. **Batch analysis**: Use `vision_query_batched([...])` for multiple clips at once (much faster).
6. **Iterate**: If initial clips aren't sufficient, call `list_windows()` again — previously-visited
   windows are auto-excluded, so you'll get fresh content (progressive scanning).
7. **Answer**: When confident, call `done(answer)`.

The transcript (if available) is included below. You can search it with Python string operations
or use llm_query() to analyze chunks — you don't need a special tool for transcript search.
"""


class VideoToolkit(Toolkit):
    """Complete video analysis toolkit with retrieval, media, and vision tools."""

    def __init__(
        self,
        vision_model: str | None = None,
        subtitle_mode: str = "auto",
        workspace_dir: str = "./sanjaya_artifacts",
        max_frames_per_clip: int = 8,
        window_size_s: float = 45.0,
        stride_s: float = 30.0,
    ):
        self.vision_model = vision_model
        self.subtitle_mode = subtitle_mode
        self.workspace_dir = workspace_dir
        self.max_frames_per_clip = max_frames_per_clip
        self.window_size_s = window_size_s
        self.stride_s = stride_s

        # State initialized during setup()
        self._video_path: str | None = None
        self._question: str | None = None
        self._subtitle_path: str | None = None
        self._transcript_text: str | None = None
        self._workspace: ArtifactWorkspace | None = None
        self._mount: WorkspaceMount | None = None
        self._llm_client: Any = None  # Set by Agent before setup

        # Progressive scanning state
        self._candidate_windows: list[dict[str, Any]] = []
        self._candidate_by_id: dict[str, dict[str, Any]] = {}
        self._clips: dict[str, dict[str, Any]] = {}
        self._visited_window_ids: set[str] = set()
        self._visited_ranges: list[tuple[float, float]] = []

    def setup(self, context: dict[str, Any]) -> None:
        """Initialize workspace, resolve subtitles, prepare state."""
        self._video_path = context.get("video")
        self._question = context.get("question")
        self._subtitle_path = context.get("subtitle")

        if not self._video_path:
            return

        # Create workspace
        self._workspace = ArtifactWorkspace(base_dir=self.workspace_dir)
        self._mount = WorkspaceMount(str(self._workspace.run_dir))

        # Resolve subtitles
        if self.subtitle_mode != "none":
            result = ensure_subtitle_sidecar(
                video_path=self._video_path,
                explicit_subtitle_path=self._subtitle_path,
                mode=self.subtitle_mode,
            )
            if result.subtitle_path:
                self._subtitle_path = result.subtitle_path
                self._transcript_text = self._load_transcript_text(result.subtitle_path)

    def _load_transcript_text(self, subtitle_path: str) -> str | None:
        """Load transcript as readable text for prompt injection."""
        import json
        from pathlib import Path

        path = Path(subtitle_path)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        segments = []
        if isinstance(data, dict):
            segments = data.get("segments", [])
        elif isinstance(data, list):
            segments = data

        if not segments:
            return None

        lines: list[str] = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = seg.get("start", "?")
            end = seg.get("end", "?")
            text = seg.get("text", "").strip()
            if text:
                lines.append(f"[{start}s-{end}s] {text}")

        return "\n".join(lines) if lines else None

    def teardown(self) -> None:
        """No persistent cleanup needed."""

    def tools(self) -> list[Tool]:
        """Returns all 7 video tools."""
        return [
            self._make_get_video_info_tool(),
            self._make_list_windows_tool(),
            self._make_extract_clip_tool(),
            self._make_sample_frames_tool(),
            self._make_vision_query_tool(),
            self._make_vision_query_batched_tool(),
            self._make_get_state_tool(),
        ]

    def get_state(self) -> dict[str, Any]:
        return {
            "clips_extracted": len(self._clips),
            "windows_visited": len(self._visited_window_ids),
            "visited_ranges": self._visited_ranges,
            "total_coverage_s": sum(e - s for s, e in self._visited_ranges),
            "clips": self._clips,
            "candidate_windows": self._candidate_windows,
            "run_id": self._workspace.run_id if self._workspace else None,
        }

    def build_evidence(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        for clip_id, clip in self._clips.items():
            evidence.append(Evidence(
                source=f"video:{clip['start_s']:.1f}s-{clip['end_s']:.1f}s",
                rationale=f"Clip {clip_id} extracted for analysis",
                artifacts={
                    "clip_path": clip.get("clip_path"),
                    "frame_paths": clip.get("frame_paths", []),
                },
            ))
        return evidence

    def prompt_section(self) -> str | None:
        parts = [_VIDEO_STRATEGY_PROMPT]
        if self._transcript_text:
            parts.append(f"\n## Transcript\n```\n{self._transcript_text}\n```")
        return "\n".join(parts)

    def get_os_access(self) -> Any | None:
        """Return OSAccess for Monty mount, if workspace is set up."""
        if self._mount:
            return self._mount.build_os_access()
        return None

    # ── Tool factories ──────────────────────────────────────

    def _make_get_video_info_tool(self) -> Tool:
        video_path = self._video_path

        def _get_video_info() -> dict:
            """Get video metadata: duration, resolution, codec, file size."""
            if not video_path:
                return {"error": "No video loaded"}
            return get_video_info(video_path)

        return Tool(
            name="get_video_info",
            description="Get video metadata: duration, resolution, codec, file size. Call this first to understand what you're working with.",
            fn=_get_video_info,
            parameters={},
            return_type="dict",
        )

    def _make_list_windows_tool(self) -> Tool:
        toolkit = self

        def _list_windows(
            question: str | None = None,
            top_k: int | None = None,
            window_size_s: float = 45.0,
        ) -> list[dict]:
            """Generate ranked candidate temporal windows for analysis."""
            if not toolkit._video_path:
                return []

            effective_question = question or toolkit._question or ""
            duration_s = video_duration_seconds(toolkit._video_path)

            if top_k is None:
                top_k = max(8, int(duration_s / 60))

            subtitle_wins: list[dict[str, Any]] = []
            if toolkit._subtitle_path:
                subtitle_top = max(top_k * 2, 24) + len(toolkit._visited_window_ids)
                subtitle_wins = subtitle_anchored_windows(
                    question=effective_question,
                    subtitle_path=toolkit._subtitle_path,
                    window_size_s=window_size_s,
                    top_k=subtitle_top,
                )

            sliding = sliding_windows(
                duration_s=duration_s,
                window_size_s=window_size_s,
                stride_s=toolkit.stride_s,
            )

            merged = hybrid_merge(
                subtitle_windows=subtitle_wins,
                sliding=sliding,
                top_k=top_k,
                exclude_ids=toolkit._visited_window_ids,
                exclude_ranges=toolkit._visited_ranges,
            )

            toolkit._candidate_windows = merged
            toolkit._candidate_by_id = {w["window_id"]: w for w in merged}

            if toolkit._workspace:
                toolkit._workspace.record_windows(merged)
                toolkit._mount = WorkspaceMount(str(toolkit._workspace.run_dir))

            return merged

        return Tool(
            name="list_windows",
            description="Generate ranked candidate temporal windows for analysis. Previously-visited windows are auto-excluded (progressive scanning).",
            fn=_list_windows,
            parameters={
                "question": ToolParam(name="question", type_hint="str | None", default=None, description="Override question for scoring."),
                "top_k": ToolParam(name="top_k", type_hint="int | None", default=None, description="Max windows. None = auto-scale with duration."),
                "window_size_s": ToolParam(name="window_size_s", type_hint="float", default=45.0, description="Window duration in seconds."),
            },
            return_type="list[dict]",
        )

    def _make_extract_clip_tool(self) -> Tool:
        toolkit = self

        def _extract_clip(
            *,
            window_id: str | None = None,
            start_s: float | None = None,
            end_s: float | None = None,
        ) -> dict:
            """Extract a video clip from the source video."""
            if not toolkit._video_path:
                raise ValueError("No video loaded")

            selected = toolkit._candidate_by_id.get(window_id or "") if window_id else None
            clip_start = float(start_s) if start_s is not None else (selected["start_s"] if selected else None)
            clip_end = float(end_s) if end_s is not None else (selected["end_s"] if selected else None)

            if clip_start is None or clip_end is None:
                raise ValueError("extract_clip requires window_id or explicit start_s/end_s")

            # Progressive scanning
            if window_id:
                toolkit._visited_window_ids.add(window_id)
            toolkit._visited_ranges.append((clip_start, clip_end))

            for cand in toolkit._candidate_windows:
                if cand["window_id"] in toolkit._visited_window_ids:
                    continue
                overlap_start = max(cand["start_s"], clip_start)
                overlap_end = min(cand["end_s"], clip_end)
                overlap = max(0.0, overlap_end - overlap_start)
                cand_len = cand["end_s"] - cand["start_s"]
                if cand_len > 0 and overlap / cand_len >= 0.5:
                    toolkit._visited_window_ids.add(cand["window_id"])

            resolved_clip_id = window_id or f"clip-{len(toolkit._clips) + 1}"

            if not toolkit._workspace:
                raise ValueError("Workspace not initialized")

            output_path = toolkit._workspace.clip_path(resolved_clip_id)
            clip_path = _extract_clip_impl(
                video_path=toolkit._video_path,
                start_s=clip_start,
                end_s=clip_end,
                output_path=str(output_path),
            )

            artifact = {
                "clip_id": resolved_clip_id,
                "clip_path": clip_path,
                "start_s": clip_start,
                "end_s": clip_end,
                "frame_paths": [],
            }

            toolkit._clips[resolved_clip_id] = artifact
            toolkit._workspace.record_clip(artifact, window_id=window_id)

            if toolkit._mount:
                toolkit._mount = WorkspaceMount(str(toolkit._workspace.run_dir))

            return artifact

        return Tool(
            name="extract_clip",
            description="Extract a video clip from the source video. Provide either a window_id (from list_windows) or explicit timestamps.",
            fn=_extract_clip,
            parameters={
                "window_id": ToolParam(name="window_id", type_hint="str | None", default=None, description="Window ID from list_windows()."),
                "start_s": ToolParam(name="start_s", type_hint="float | None", default=None, description="Start timestamp in seconds."),
                "end_s": ToolParam(name="end_s", type_hint="float | None", default=None, description="End timestamp in seconds."),
            },
            return_type="dict",
        )

    def _make_sample_frames_tool(self) -> Tool:
        toolkit = self

        def _sample_frames(
            *,
            clip_id: str | None = None,
            clip_path: str | None = None,
            max_frames: int = 8,
        ) -> list[str]:
            """Sample uniformly-spaced frames from an extracted clip."""
            resolved_clip_id = clip_id
            resolved_path = clip_path

            if resolved_clip_id:
                artifact = toolkit._clips.get(resolved_clip_id)
                if artifact is None:
                    raise ValueError(f"Unknown clip_id: {resolved_clip_id}")
                resolved_path = artifact["clip_path"]
            elif resolved_path:
                resolved_clip_id = f"clip-{len(toolkit._clips) + 1}"
            else:
                raise ValueError("sample_frames requires clip_id or clip_path")

            assert resolved_path is not None
            assert resolved_clip_id is not None

            if not toolkit._workspace:
                raise ValueError("Workspace not initialized")

            duration = video_duration_seconds(resolved_path)
            frame_dir = toolkit._workspace.frame_dir(resolved_clip_id)

            frames = _sample_frames_impl(
                video_path=resolved_path,
                start_s=0.0,
                end_s=duration,
                output_dir=str(frame_dir),
                max_frames=max_frames,
            )

            if resolved_clip_id in toolkit._clips:
                toolkit._clips[resolved_clip_id]["frame_paths"] = frames
            toolkit._workspace.update_frames(resolved_clip_id, frames)

            if toolkit._mount:
                toolkit._mount = WorkspaceMount(str(toolkit._workspace.run_dir))

            return frames

        return Tool(
            name="sample_frames",
            description="Sample uniformly-spaced frames from an extracted clip.",
            fn=_sample_frames,
            parameters={
                "clip_id": ToolParam(name="clip_id", type_hint="str | None", default=None, description="ID from extract_clip()."),
                "clip_path": ToolParam(name="clip_path", type_hint="str | None", default=None, description="Direct path (alternative to clip_id)."),
                "max_frames": ToolParam(name="max_frames", type_hint="int", default=8, description="Number of frames to extract."),
            },
            return_type="list[str]",
        )

    def _make_vision_query_tool(self) -> Tool:
        toolkit = self

        if self._llm_client:
            fn = make_vision_query_fn(
                llm_client=self._llm_client,
                get_clips=lambda: toolkit._clips,
                get_question=lambda: toolkit._question or "",
            )
        else:
            def fn(**kwargs: Any) -> str:
                raise RuntimeError("Vision model not configured. Set vision_model on the Agent.")

        return Tool(
            name="vision_query",
            description="Query a vision model about video frames or a clip.",
            fn=fn,
            parameters={
                "prompt": ToolParam(name="prompt", type_hint="str | None", default=None, description="What to ask about the visual content."),
                "clip_id": ToolParam(name="clip_id", type_hint="str | None", default=None, description="ID from extract_clip()."),
                "frame_paths": ToolParam(name="frame_paths", type_hint="list[str] | None", default=None, description="Direct frame paths."),
            },
            return_type="str",
        )

    def _make_vision_query_batched_tool(self) -> Tool:
        toolkit = self

        if self._llm_client:
            fn = make_vision_query_batched_fn(
                llm_client=self._llm_client,
                get_clips=lambda: toolkit._clips,
                get_question=lambda: toolkit._question or "",
            )
        else:
            def fn(queries: list[dict]) -> list[str]:
                raise RuntimeError("Vision model not configured. Set vision_model on the Agent.")

        return Tool(
            name="vision_query_batched",
            description="Run multiple vision queries concurrently. Much faster than sequential vision_query() calls.",
            fn=fn,
            parameters={
                "queries": ToolParam(name="queries", type_hint="list[dict]", description="List of dicts with keys matching vision_query params."),
            },
            return_type="list[str]",
        )

    def _make_get_state_tool(self) -> Tool:
        toolkit = self

        def _get_state() -> dict:
            """Inspect accumulated analysis state and workspace manifest."""
            return toolkit.get_state()

        return Tool(
            name="get_state",
            description="Inspect accumulated analysis state and workspace manifest.",
            fn=_get_state,
            parameters={},
            return_type="dict",
        )
