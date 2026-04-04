"""Prompt templates for VideoRLM orchestration."""

from __future__ import annotations

from dataclasses import dataclass

VIDEO_DEFAULT_QUERY = "Answer the user question grounded in video evidence."


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


@dataclass
class VideoMeta:
    """Lightweight metadata passed into the prompt builder."""

    duration_s: float
    segment_count: int
    transcript_text: str  # pre-formatted timestamped transcript


def format_transcript_block(
    segments: list[dict],
    *,
    max_chars: int = 60_000,
) -> str:
    """Turn subtitle segments into a compact timestamped transcript string.

    Format:  [M:SS-M:SS] text
    Segments are concatenated until *max_chars* is reached; a truncation
    notice is appended when the transcript is clipped.
    """
    lines: list[str] = []
    chars = 0
    truncated = False
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        speaker = seg.get("speaker")
        if speaker:
            line = f"[{_fmt_ts(start)}-{_fmt_ts(end)}] ({speaker}) {text}"
        else:
            line = f"[{_fmt_ts(start)}-{_fmt_ts(end)}] {text}"
        if chars + len(line) > max_chars:
            truncated = True
            break
        lines.append(line)
        chars += len(line) + 1  # +1 for newline

    body = "\n".join(lines)
    if truncated:
        body += f"\n... (transcript truncated at {max_chars:,} chars)"
    return body


def build_video_system_prompt(
    *,
    video_meta: VideoMeta | None = None,
) -> list[dict[str, str]]:
    """System prompt for the video-capable orchestrator.

    When *video_meta* is provided the prompt includes the full timestamped
    transcript and video metadata so the orchestrator can **plan** which
    parts of the video to explore before doing any visual inspection.
    """

    # ── tool contract (always present) ──────────────────────────────
    tool_contract = (
        "You are a recursive video-analysis orchestrator with a Python REPL.\n"
        "Produce at most ONE fenced Python block per iteration.\n\n"
        "## Available tools\n"
        "Core:  get_context(), llm_query(prompt), done(value)\n"
        "Video:\n"
        "  list_candidate_windows(question=None, top_k=None, window_size_s=45.0, stride_s=30.0)\n"
        "  extract_clip(window_id=None, start_s=None, end_s=None, clip_id=None)\n"
        "  sample_frames(clip_id=None, clip_path=None, max_frames=8)\n"
        "  vision_query(prompt=None, clip_id=None, frame_paths=None, clip_paths=None)\n"
        "  get_clip_manifest()\n"
        "  get_trace_log()\n"
        "Do NOT invent parameter names.\n"
    )

    # ── strategy guidance ───────────────────────────────────────────
    strategy = (
        "## Strategy\n"
        "1. **Plan first** — Read the transcript below to understand the video's "
        "structure, topics, and interesting moments BEFORE extracting any clips.\n"
        "2. **Identify candidates** — Based on the transcript and the user's question, "
        "decide which timestamp ranges are most relevant or interesting.\n"
        "3. **Explore progressively** — Each call to list_candidate_windows() "
        "excludes previously-visited regions. Use this to sweep through the video "
        "across iterations rather than re-checking the same windows.\n"
        "4. **Visually verify** — Use extract_clip → sample_frames → vision_query "
        "to confirm what you identified from the transcript.\n"
        "5. **Converge** — Once you have enough evidence, call done(value) with "
        "a structured answer including timestamps.\n\n"
        "Canonical tool chain:\n"
        "  windows = list_candidate_windows(...)\n"
        "  clip = extract_clip(window_id=windows[i]['window_id'])\n"
        "  sample_frames(clip_id=clip['clip_id'], max_frames=4)\n"
        "  vision_query(clip_id=clip['clip_id'], prompt='...')\n"
        "  done({'answer': ..., 'evidence': [...]})\n"
    )

    parts = [tool_contract, strategy]

    # ── video metadata + transcript ─────────────────────────────────
    if video_meta is not None:
        meta_block = (
            "## Video metadata\n"
            f"- Duration: {_fmt_ts(video_meta.duration_s)} "
            f"({video_meta.duration_s:.0f}s)\n"
            f"- Transcript segments: {video_meta.segment_count}\n\n"
            "## Full transcript\n"
            "Use this to plan which parts of the video to explore.\n"
            "Timestamps are in [start-end] format.\n\n"
            f"{video_meta.transcript_text}\n"
        )
        parts.append(meta_block)

    return [{"role": "system", "content": "\n".join(parts)}]


def next_video_action_prompt(
    question: str,
    iteration: int,
    final_answer: bool = False,
) -> dict[str, str]:
    """Iteration prompt for the video orchestrator."""
    if final_answer:
        content = (
            "Max iterations reached. Return your best grounded final answer now.\n"
            "Call done(value) with a structured answer including timestamps and evidence."
        )
    else:
        content = (
            f"Iteration {iteration + 1}. Question: {question}\n\n"
            "Write one Python code block. Use the transcript above to target "
            "the most relevant parts of the video. Use exact tool signatures only.\n"
            "Previously-visited windows are auto-excluded from list_candidate_windows().\n"
            "Call done(value) when you have a well-grounded answer with timestamps."
        )
    return {"role": "user", "content": content}
