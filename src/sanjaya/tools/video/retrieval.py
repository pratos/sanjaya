"""Hybrid retrieval for generating ranked candidate temporal windows.

Ported from video_tools/retrieval.py — same logic, uses dict instead of CandidateWindow model.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class SubtitleSegment:
    """Normalized subtitle segment."""
    start_s: float
    end_s: float
    text: str


_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(text.lower()) if len(tok) > 1}


def expand_query(question: str, llm_client: Any) -> str:
    """Use a sub-LLM to expand a question into additional search terms.

    Returns the original question plus expanded keywords, concatenated
    so that _tokenize picks up all terms for matching.
    """
    if llm_client is None:
        return question
    try:
        expanded = llm_client.completion(
            f"List 10 keywords and synonyms to search for in a video transcript "
            f"to answer this question. Return ONLY the words, comma-separated, "
            f"no explanation:\n\n{question}"
        )
        return f"{question} {expanded}"
    except Exception:
        return question


def _overlap_score(question_tokens: set[str], content: str) -> float:
    """Bag-of-words cosine between pre-tokenized query and content."""
    c = _tokenize(content)
    if not question_tokens or not c:
        return 0.0
    return len(question_tokens & c) / math.sqrt(len(question_tokens) * len(c))


def load_subtitle_segments(subtitle_path: str) -> list[SubtitleSegment]:
    """Load subtitles from common JSON structures."""
    path = Path(subtitle_path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
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
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        segments.append(SubtitleSegment(start_s=start_s, end_s=end_s, text=str(text).strip()))

    return segments


def subtitle_anchored_windows(
    *,
    question: str,
    subtitle_path: str,
    window_size_s: float,
    top_k: int,
    llm_client: Any = None,
) -> list[dict[str, Any]]:
    """Propose windows centered around subtitle segments relevant to the question."""
    segments = load_subtitle_segments(subtitle_path)
    if not segments:
        return []

    expanded = expand_query(question, llm_client)
    q_tokens = _tokenize(expanded)

    proposals: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        score = _overlap_score(q_tokens, segment.text)
        if score <= 0:
            continue

        center = (segment.start_s + segment.end_s) / 2
        start_s = max(0.0, center - window_size_s / 2)
        end_s = start_s + window_size_s
        proposals.append({
            "window_id": f"sub-{index}",
            "strategy": "subtitle",
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "score": round(score, 4),
            "reason": f"Subtitle overlap: {segment.text[:160]}",
        })

    proposals.sort(key=lambda x: x["score"], reverse=True)
    return proposals[:top_k]


def sliding_windows(
    *,
    duration_s: float,
    window_size_s: float,
    stride_s: float,
) -> list[dict[str, Any]]:
    """Generate uniform temporal windows covering the full video."""
    if duration_s <= 0:
        return []

    duration_s = float(duration_s)
    window_size_s = max(1.0, float(window_size_s))
    stride_s = max(1.0, float(stride_s))

    windows: list[dict[str, Any]] = []
    position = 0.0
    idx = 0

    while position < duration_s:
        start_s = position
        end_s = min(duration_s, position + window_size_s)
        coverage = (end_s - start_s) / window_size_s
        score = max(0.05, coverage)

        windows.append({
            "window_id": f"slide-{idx}",
            "strategy": "sliding",
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
            "score": round(score, 4),
            "reason": "Uniform temporal coverage",
        })

        if end_s >= duration_s:
            break
        idx += 1
        position += stride_s

    return windows


def _overlaps_visited(
    window: dict[str, Any],
    visited_ranges: list[tuple[float, float]],
    threshold: float = 0.5,
) -> bool:
    if not visited_ranges:
        return False
    w_len = window["end_s"] - window["start_s"]
    if w_len <= 0:
        return False
    total_overlap = 0.0
    for v_start, v_end in visited_ranges:
        overlap = max(0.0, min(window["end_s"], v_end) - max(window["start_s"], v_start))
        total_overlap += overlap
    return total_overlap / w_len >= threshold


def hybrid_merge(
    *,
    subtitle_windows: list[dict[str, Any]],
    sliding: list[dict[str, Any]],
    top_k: int,
    subtitle_weight: float = 1.0,
    sliding_weight: float = 0.35,
    exclude_ids: set[str] | None = None,
    exclude_ranges: list[tuple[float, float]] | None = None,
) -> list[dict[str, Any]]:
    """Merge/re-rank windows from both strategies with weighted scoring."""
    _exclude = exclude_ids or set()
    _ranges = exclude_ranges or []
    merged: dict[str, dict[str, Any]] = {}

    for window in subtitle_windows:
        if window["window_id"] in _exclude or _overlaps_visited(window, _ranges):
            continue
        weighted = dict(window)
        weighted["score"] = round(window["score"] * subtitle_weight, 4)
        merged[weighted["window_id"]] = weighted

    for window in sliding:
        if window["window_id"] in _exclude or _overlaps_visited(window, _ranges):
            continue
        weighted_score = round(window["score"] * sliding_weight, 4)
        if window["window_id"] in merged:
            existing = merged[window["window_id"]]
            existing["score"] = round(existing["score"] + weighted_score, 4)
            existing["reason"] = f"{existing['reason']}; + sliding prior"
        else:
            w = dict(window)
            w["score"] = weighted_score
            merged[window["window_id"]] = w

    ranked = sorted(
        merged.values(),
        key=lambda item: (item["score"], item["strategy"] == "subtitle", -(item["end_s"] - item["start_s"])),
        reverse=True,
    )
    return ranked[:top_k]
