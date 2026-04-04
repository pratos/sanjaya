"""Hybrid retrieval for long-video question answering."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from video_rlm.video_models import CandidateWindow


@dataclass
class SubtitleSegment:
    """Normalized subtitle segment."""

    start_s: float
    end_s: float
    text: str


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(text.lower()) if len(tok) > 1}


def _overlap_score(question: str, content: str) -> float:
    q = _tokenize(question)
    c = _tokenize(content)
    if not q or not c:
        return 0.0

    shared = len(q & c)
    return shared / math.sqrt(len(q) * len(c))


def _read_json(path: Path) -> object:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def load_subtitle_segments(subtitle_path: str) -> list[SubtitleSegment]:
    """Load subtitles from common LongVideoBench-style JSON structures."""
    path = Path(subtitle_path)
    if not path.exists():
        return []

    payload = _read_json(path)
    candidates: list[dict] = []

    if isinstance(payload, list):
        candidates = [x for x in payload if isinstance(x, dict)]
    elif isinstance(payload, dict):
        for key in ("segments", "subtitles", "items", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates = [x for x in value if isinstance(x, dict)]
                if candidates:
                    break

    segments: list[SubtitleSegment] = []
    for item in candidates:
        start = item.get("start") or item.get("start_time") or item.get("from")
        end = item.get("end") or item.get("end_time") or item.get("to")
        text = item.get("text") or item.get("subtitle") or item.get("content")
        if text is None:
            continue

        try:
            start_s = float(start) if start is not None else None
            end_s = float(end) if end is not None else None
        except (TypeError, ValueError):
            continue

        if start_s is None or end_s is None or end_s <= start_s:
            continue

        segments.append(SubtitleSegment(start_s=start_s, end_s=end_s, text=str(text).strip()))

    return segments


def subtitle_anchored_windows(
    *,
    question: str,
    subtitle_path: str,
    window_size_s: float,
    top_k: int,
) -> list[CandidateWindow]:
    """Propose windows centered around subtitle segments relevant to the question."""
    segments = load_subtitle_segments(subtitle_path)
    if not segments:
        return []

    proposals: list[CandidateWindow] = []
    for index, segment in enumerate(segments):
        score = _overlap_score(question, segment.text)
        if score <= 0:
            continue

        center = (segment.start_s + segment.end_s) / 2
        start_s = max(0.0, center - window_size_s / 2)
        end_s = start_s + window_size_s
        proposals.append(
            CandidateWindow(
                window_id=f"sub-{index}",
                strategy="subtitle",
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                score=round(score, 4),
                reason=f"Subtitle overlap: {segment.text[:160]}",
            )
        )

    proposals.sort(key=lambda x: x.score, reverse=True)
    return proposals[:top_k]


def sliding_windows(
    *,
    duration_s: float,
    window_size_s: float,
    stride_s: float,
) -> list[CandidateWindow]:
    """Generate uniform temporal windows covering the full video."""
    if duration_s <= 0:
        return []

    duration_s = float(duration_s)
    window_size_s = max(1.0, float(window_size_s))
    stride_s = max(1.0, float(stride_s))

    windows: list[CandidateWindow] = []
    position = 0.0
    idx = 0

    while position < duration_s:
        start_s = position
        end_s = min(duration_s, position + window_size_s)
        coverage = (end_s - start_s) / window_size_s
        score = max(0.05, coverage)

        windows.append(
            CandidateWindow(
                window_id=f"slide-{idx}",
                strategy="sliding",
                start_s=round(start_s, 3),
                end_s=round(end_s, 3),
                score=round(score, 4),
                reason="Uniform temporal coverage",
            )
        )

        if end_s >= duration_s:
            break

        idx += 1
        position += stride_s

    return windows


def hybrid_merge(
    *,
    subtitle_windows: list[CandidateWindow],
    sliding: list[CandidateWindow],
    top_k: int,
    subtitle_weight: float = 1.0,
    sliding_weight: float = 0.35,
) -> list[CandidateWindow]:
    """Merge/re-rank windows from both strategies with weighted scoring."""
    merged: dict[str, CandidateWindow] = {}

    for window in subtitle_windows:
        weighted = window.model_copy(update={"score": round(window.score * subtitle_weight, 4)})
        merged[weighted.window_id] = weighted

    for window in sliding:
        weighted_score = round(window.score * sliding_weight, 4)
        if window.window_id in merged:
            existing = merged[window.window_id]
            merged[window.window_id] = existing.model_copy(
                update={
                    "score": round(existing.score + weighted_score, 4),
                    "reason": f"{existing.reason}; + sliding prior",
                }
            )
        else:
            merged[window.window_id] = window.model_copy(update={"score": weighted_score})

    ranked = sorted(
        merged.values(),
        key=lambda item: (item.score, item.strategy == "subtitle", -(item.end_s - item.start_s)),
        reverse=True,
    )
    return ranked[:top_k]
