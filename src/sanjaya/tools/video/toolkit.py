"""VideoToolkit — complete video analysis toolkit for the agent."""

from __future__ import annotations

from typing import Any

from rich.console import Console

from ...answer import Evidence
from ...retrieval.sqlite_fts import SQLiteFTSBackend
from ..base import Tool, Toolkit, ToolParam
from .media import extract_clip as _extract_clip_impl
from .media import get_video_info, video_duration_seconds
from .media import sample_frames as _sample_frames_impl
from .mount import WorkspaceMount
from .retrieval import hybrid_merge, load_subtitle_segments, sliding_windows, subtitle_anchored_windows
from .transcription import ensure_subtitle_sidecar
from .vision import make_caption_frames_fn, make_vision_query_batched_fn, make_vision_query_fn
from .workspace import ArtifactWorkspace

_console = Console()

_VIDEO_STRATEGY_PROMPT = """\
## Available Tools

You are analyzing a video. You decide the strategy.

### Vision tools (two modes — choose based on need):

- **caption_frames(clip_id)** — cheap per-frame captioning via a focused vision model.
  Returns timestamped text like `["[12.5s] Presenter showing bar chart..."]`.
  Best for broad visual coverage. Feed the output to llm_query() for reasoning.

- **vision_query(prompt, clip_id)** — sends all frames in a clip to a vision model
  with your prompt. More expensive but lets you ask targeted questions about specific
  visual details (e.g. "read the exact text on this chart", "what color is the button").

### Text tools:

- **search_transcript(query)** — BM25 keyword search over subtitles. Fast.
  Returns matching segments with timestamps and scores.

- **llm_query(prompt)** — text-only reasoning. Feed it captions, transcript
  excerpts, or any accumulated evidence. Cheap, sees everything at once.
  Good for synthesis and cross-referencing.

### Media tools:

- **get_video_info()** — video metadata (duration, resolution, codec).
- **extract_clip(window_id= or start_s=/end_s=)** — cut a clip from the video.
- **sample_frames(clip_id=, max_frames=)** — extract uniformly-spaced frames from a clip.
- **list_windows()** — ranked candidate temporal windows (progressive: visited ones auto-excluded).
- **get_state()** — inspect accumulated clips, windows, coverage.

### When to use what:

- Broad visual understanding → caption_frames() + llm_query() over the captions
- Specific visual detail (read text, check UI element) → vision_query() with a targeted prompt
- Factual / dialogue questions → search_transcript() + llm_query()
- Always: explore before answering, cite timestamps, print results so you can read them.

### One code block per response. Print everything.
"""


_VISION_FIRST_STRATEGY_PROMPT = """\
## Available Tools (Vision-Primary Question)

You are analyzing a video where the answer is primarily VISUAL — on-screen content,
products, diagrams, charts, UI elements, code, or physical objects.

### Vision tools (two modes — choose based on need):

- **caption_frames(clip_id)** — cheap per-frame captioning. Returns timestamped
  descriptions. Best for scanning large portions of the video quickly.
  Feed results to llm_query() for reasoning across many frames.

- **vision_query(prompt, clip_id)** — targeted visual question. Sends frames
  directly with your prompt. Use when you need to read specific text, verify
  a detail, or ask about something the captions missed.

### Text tools:

- **search_transcript(query)** — keyword search over subtitles. Use as
  secondary confirmation for visual observations.

- **llm_query(prompt)** — text-only reasoning over captions and transcript.

### Media tools:

- **get_video_info()** — video metadata.
- **extract_clip(window_id= or start_s=/end_s=)** — cut a clip.
- **sample_frames(clip_id=, max_frames=)** — extract frames from a clip.
- **list_windows()** — ranked candidate windows (progressive scanning).
- **get_state()** — inspect accumulated state.

### When to use what:

- Dense visual coverage → extract clips across the video, caption_frames() each, llm_query() to synthesize
- Specific visual verification → vision_query() with a precise prompt about what to look for
- Transcript is secondary for this question — use it to confirm, not to lead
- Sample densely: more clips and frames gives better visual coverage
- Always: explore before answering, cite timestamps, print results so you can read them.

### One code block per response. Print everything.
"""


def _ranges_overlap(a: dict[str, Any], b: dict[str, Any], threshold: float = 0.5) -> bool:
    """Check if two clip ranges overlap by at least `threshold` of the shorter one."""
    overlap = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    shorter = min(a["end_s"] - a["start_s"], b["end_s"] - b["start_s"])
    return shorter > 0 and overlap / shorter >= threshold


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
        self._captioner: Any = None  # Set by Agent before setup (Moondream or similar)
        self._tracer: Any = None  # Set by Agent before setup
        self._budget: Any = None  # Set by Agent before setup

        # Question modality (set during setup from context)
        self._modality: str = "balanced"

        # Transcript search
        self._fts: SQLiteFTSBackend | None = None

        # Progressive scanning state
        self._candidate_windows: list[dict[str, Any]] = []
        self._candidate_by_id: dict[str, dict[str, Any]] = {}
        self._clips: dict[str, dict[str, Any]] = {}
        self._visited_window_ids: set[str] = set()
        self._visited_ranges: list[tuple[float, float]] = []

    def setup(self, context: dict[str, Any]) -> None:
        """Initialize workspace, resolve subtitles, prepare state."""
        from pathlib import Path

        video = context.get("video")
        self._video_path = str(Path(video).resolve()) if video else None
        self._question = context.get("question")
        self._modality = context.get("modality", "balanced")
        subtitle = context.get("subtitle")
        self._subtitle_path = str(Path(subtitle).resolve()) if subtitle else None

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

                # Index transcript segments into FTS for search_transcript()
                segments = load_subtitle_segments(result.subtitle_path)
                if segments:
                    self._fts = SQLiteFTSBackend(path=":memory:")
                    self._fts.index(
                        documents=[seg.text for seg in segments],
                        metadata=[{"start_s": seg.start_s, "end_s": seg.end_s} for seg in segments],
                        collection="transcript",
                    )

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
        """Returns all video tools."""
        return [
            self._make_get_video_info_tool(),
            self._make_search_transcript_tool(),
            self._make_list_windows_tool(),
            self._make_extract_clip_tool(),
            self._make_sample_frames_tool(),
            self._make_caption_frames_tool(),
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
        # Sort clips by start time and merge overlapping ranges
        sorted_clips = sorted(self._clips.values(), key=lambda c: c["start_s"])
        merged: list[dict[str, Any]] = []
        for clip in sorted_clips:
            if merged and _ranges_overlap(merged[-1], clip):
                prev = merged[-1]
                prev["start_s"] = min(prev["start_s"], clip["start_s"])
                prev["end_s"] = max(prev["end_s"], clip["end_s"])
                # Keep the clip with more frames
                if len(clip.get("frame_paths", [])) > len(prev.get("frame_paths", [])):
                    prev["clip_path"] = clip["clip_path"]
                    prev["frame_paths"] = clip["frame_paths"]
                    prev["clip_id"] = clip["clip_id"]
            else:
                merged.append(dict(clip))

        return [
            Evidence(
                source=f"video:{c['start_s']:.1f}s-{c['end_s']:.1f}s",
                rationale=f"Clip {c['clip_id']} extracted for analysis",
                artifacts={
                    "clip_path": c.get("clip_path"),
                    "frame_paths": c.get("frame_paths", []),
                },
            )
            for c in merged
        ]

    def prompt_section(self) -> str | None:
        if self._modality == "vision_primary":
            parts = [_VISION_FIRST_STRATEGY_PROMPT]
        else:
            parts = [_VIDEO_STRATEGY_PROMPT]
        if self._transcript_text:
            parts.append(
                "\n## Transcript\n"
                "A transcript is available. Use `search_transcript(query)` to find "
                "relevant segments by keyword. Do NOT expect the full transcript in context."
            )
        return "\n".join(parts)

    def get_os_access(self) -> Any | None:
        """Return OSAccess for Monty mount, if workspace is set up."""
        if self._mount:
            return self._mount.build_os_access()
        return None

    # ── Tool factories ──────────────────────────────────────

    def _make_get_video_info_tool(self) -> Tool:
        toolkit = self

        def _get_video_info() -> dict:
            """Get video metadata: duration, resolution, codec, file size."""
            if not toolkit._video_path:
                return {"error": "No video loaded"}
            return get_video_info(toolkit._video_path)

        return Tool(
            name="get_video_info",
            description="Get video metadata: duration, resolution, codec, file size. Call this first to understand what you're working with.",
            fn=_get_video_info,
            parameters={},
            return_type="dict",
        )

    def _make_search_transcript_tool(self) -> Tool:
        toolkit = self

        def _search_transcript(query: str, top_k: int = 5) -> list[dict]:
            """Search the video transcript for segments matching a query.

            Returns ranked results with text, timestamps, and relevance score.
            Use this to find what was said at specific times or about specific topics.
            """
            if toolkit._fts is None:
                return [{"error": "No transcript indexed"}]
            results = toolkit._fts.search(query, top_k=top_k, collection="transcript")
            return [
                {
                    "text": r["text"],
                    "start_s": r["metadata"]["start_s"],
                    "end_s": r["metadata"]["end_s"],
                    "score": round(r["score"], 4),
                }
                for r in results
            ]

        return Tool(
            name="search_transcript",
            description="Search the video transcript by keyword/phrase. Returns matching segments with timestamps and scores.",
            fn=_search_transcript,
            parameters={
                "query": ToolParam(name="query", type_hint="str", description="Search query."),
                "top_k": ToolParam(name="top_k", type_hint="int", default=5, description="Max results."),
            },
            return_type="list[dict]",
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
                    llm_client=toolkit._llm_client,
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

        default_max = toolkit.max_frames_per_clip

        def _sample_frames(
            *,
            clip_id: str | None = None,
            clip_path: str | None = None,
            max_frames: int = default_max,
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
                "max_frames": ToolParam(name="max_frames", type_hint="int", default=default_max, description="Number of frames to extract."),
            },
            return_type="list[str]",
        )

    def _make_caption_frames_tool(self) -> Tool:
        toolkit = self

        if self._llm_client:
            fn = make_caption_frames_fn(
                llm_client=self._llm_client,
                captioner=self._captioner,
                get_clips=lambda: toolkit._clips,
                get_tracer=lambda: toolkit._tracer,
                get_budget=lambda: toolkit._budget,
            )
        else:
            def fn(**kwargs: Any) -> list[str]:  # type: ignore[misc]
                raise RuntimeError("Vision model not configured. Set vision_model on the Agent.")

        return Tool(
            name="caption_frames",
            description=(
                "Caption each frame in a clip individually using a focused vision model. "
                "Returns timestamped descriptions like '[12.5s] Presenter shows bar chart'. "
                "Cheap and concurrent — use for broad visual coverage, then feed results "
                "to llm_query() for reasoning. "
                "IMPORTANT: call sample_frames() on the clip first."
            ),
            fn=fn,
            parameters={
                "clip_id": ToolParam(
                    name="clip_id", type_hint="str",
                    description="ID from extract_clip().",
                ),
                "prompt": ToolParam(
                    name="prompt", type_hint="str | None", default=None,
                    description="Custom captioning prompt (default: focused scene description).",
                ),
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
                get_tracer=lambda: toolkit._tracer,
                get_budget=lambda: toolkit._budget,
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
                get_tracer=lambda: toolkit._tracer,
                get_budget=lambda: toolkit._budget,
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
