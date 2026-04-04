"""Utility helpers for VideoRLM orchestration and evidence formatting."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .video_models import CandidateWindow, EvidenceItem, VideoAnswer
from .video_repl import VideoExecutionResult


def find_code_blocks(response: str) -> list[str]:
    """Extract fenced Python code blocks from an LLM response."""
    pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in pattern.findall(response)]


def format_execution_feedback(result: VideoExecutionResult, index: int, total: int) -> str:
    """Format execution result for next-iteration LLM feedback."""
    parts = [f"Code block {index}/{total} executed."]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.result is not None:
        parts.append(f"RESULT: {result.result!r}")
    if result.final_answer is not None:
        parts.append(f"FINAL_ANSWER: {result.final_answer}")
    return "\n\n".join(parts)


def extract_final_answer(response: str, execution_result: VideoExecutionResult | None) -> str | None:
    """Extract final answer from done(), FINAL(), or plain-text marker."""
    if execution_result is not None and execution_result.final_answer is not None:
        return str(execution_result.final_answer)

    final_match = re.search(r"FINAL\((.*?)\)", response, re.DOTALL)
    if final_match:
        return final_match.group(1).strip().strip("'\"")

    done_match = re.search(r"final answer\s*[:=]\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    if done_match:
        return done_match.group(1).strip()

    return None


def infer_subtitle_path(video_path: str, explicit_subtitle_path: str | None = None) -> str | None:
    """Infer subtitle sidecar path from known LongVideoBench layouts."""
    if explicit_subtitle_path:
        return explicit_subtitle_path

    video = Path(video_path)
    stem = video.stem
    candidates = [
        video.with_name(f"{stem}_en.json"),
        video.parent.parent / "meta" / f"{stem}_en.json",
        Path("data/longvideobench/meta") / f"{stem}_en.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def compute_video_run_id(video_path: str, prefix: str = "vid") -> str:
    """Compute a stable hash-based run id from video file identity/content."""
    path = Path(video_path)
    stat = path.stat()

    sha = hashlib.sha1()
    sha.update(str(path.resolve()).encode("utf-8"))
    sha.update(str(stat.st_size).encode("utf-8"))
    sha.update(str(int(stat.st_mtime)).encode("utf-8"))

    chunk_size = 1024 * 1024
    with path.open("rb") as handle:
        head = handle.read(chunk_size)
        sha.update(head)

        if stat.st_size > chunk_size:
            handle.seek(max(0, stat.st_size - chunk_size))
            tail = handle.read(chunk_size)
            sha.update(tail)

    digest = sha.hexdigest()[:12]
    return f"{prefix}-{digest}"


def build_video_answer(question: str, answer: str, manifest: dict) -> VideoAnswer:
    """Build structured VideoAnswer from raw manifest data."""
    windows: list[CandidateWindow] = []
    for item in manifest.get("candidate_windows", []):
        try:
            windows.append(CandidateWindow.model_validate(item))
        except Exception:
            continue

    evidence: list[EvidenceItem] = []
    clips = manifest.get("clips", {})
    if isinstance(clips, dict):
        for clip in clips.values():
            try:
                evidence.append(
                    EvidenceItem(
                        window_id=clip.get("window_id"),
                        start_s=float(clip.get("start_s", 0.0)),
                        end_s=float(clip.get("end_s", 0.1)),
                        rationale="Evidence clip extracted during recursive retrieval.",
                        clip_path=clip.get("clip_path"),
                        frame_paths=list(clip.get("frame_paths", [])),
                    )
                )
            except Exception:
                continue

    return VideoAnswer(
        question=question,
        answer=answer,
        evidence=evidence,
        retrieval_trace=windows,
    )
