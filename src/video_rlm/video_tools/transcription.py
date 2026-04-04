"""Subtitle transcription helpers (local Whisper CLI or OpenAI API)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sanjaya.settings import get_settings


class TranscriptionError(RuntimeError):
    """Raised when subtitle transcription fails."""


@dataclass
class SubtitlePreparationResult:
    """Result of subtitle sidecar lookup/generation."""

    subtitle_path: str | None
    generated: bool
    source: str | None
    error: str | None = None


def _normalize_segments(raw_segments: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for segment in raw_segments:
        start = getattr(segment, "start", None)
        end = getattr(segment, "end", None)
        text = getattr(segment, "text", None)

        if isinstance(segment, dict):
            start = segment.get("start", start)
            end = segment.get("end", end)
            text = segment.get("text", text)

        try:
            start_s = float(start)
            end_s = float(end)
        except (TypeError, ValueError):
            continue

        if end_s <= start_s:
            continue

        normalized.append(
            {
                "start": round(start_s, 3),
                "end": round(end_s, 3),
                "text": str(text or "").strip(),
            }
        )

    return normalized


def _write_sidecar(path: Path, segments: list[dict[str, Any]], metadata: dict[str, Any]) -> str:
    payload = {
        "segments": segments,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def transcribe_with_whisper_local(
    *,
    video_path: str,
    output_path: str,
    model: str = "base",
    language: str = "en",
) -> str:
    """Generate subtitle JSON using local whisper CLI with timestamps."""
    if shutil.which("whisper") is None:
        raise TranscriptionError("Local whisper CLI not found. Install openai-whisper to use local transcription.")

    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    out = Path(output_path)

    with tempfile.TemporaryDirectory(prefix="video-rlm-whisper-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            "whisper",
            str(src),
            "--model",
            model,
            "--task",
            "transcribe",
            "--language",
            language,
            "--output_format",
            "json",
            "--output_dir",
            str(tmp_path),
            "--fp16",
            "False",
        ]

        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise TranscriptionError(result.stderr.strip() or "whisper transcription failed")

        whisper_json = tmp_path / f"{src.stem}.json"
        if not whisper_json.exists():
            raise TranscriptionError(f"whisper did not produce expected output: {whisper_json}")

        payload = json.loads(whisper_json.read_text(encoding="utf-8"))
        segments = _normalize_segments(payload.get("segments", []))
        if not segments:
            raise TranscriptionError("whisper output contained no usable segments")

        return _write_sidecar(
            out,
            segments,
            metadata={
                "source": "whisper-local",
                "model": model,
                "language": language,
                "video_path": str(src),
            },
        )


def transcribe_with_openai_api(
    *,
    video_path: str,
    output_path: str,
    model: str = "gpt-4o-mini-transcribe",
    language: str = "en",
    api_key: str | None = None,
) -> str:
    """Generate subtitle JSON using OpenAI transcription API with segment timestamps."""
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")

    key = api_key or os.getenv("OPENAI_API_KEY") or get_settings().openai_api_key
    if not key:
        raise TranscriptionError("OPENAI_API_KEY is required for API transcription")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise TranscriptionError("openai package not available for API transcription") from exc

    client = OpenAI(api_key=key)

    with src.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    raw_segments = getattr(response, "segments", None)
    if raw_segments is None and isinstance(response, dict):
        raw_segments = response.get("segments")
    segments = _normalize_segments(list(raw_segments or []))

    if not segments:
        raise TranscriptionError("API transcription returned no usable segments")

    out = Path(output_path)
    return _write_sidecar(
        out,
        segments,
        metadata={
            "source": "openai-api",
            "model": model,
            "language": language,
            "video_path": str(src),
        },
    )


def ensure_subtitle_sidecar(
    *,
    video_path: str,
    explicit_subtitle_path: str | None = None,
    mode: str = "auto",
    output_dir: str = "data/longvideobench/meta",
    local_model: str = "base",
    api_model: str = "gpt-4o-mini-transcribe",
    language: str = "en",
) -> SubtitlePreparationResult:
    """Find or generate subtitle sidecar using local or API transcription."""
    mode_normalized = mode.strip().lower()
    if mode_normalized not in {"auto", "local", "api", "none"}:
        raise ValueError("subtitle mode must be one of: auto, local, api, none")

    explicit_path = Path(explicit_subtitle_path) if explicit_subtitle_path else None
    if explicit_path and explicit_path.exists():
        return SubtitlePreparationResult(
            subtitle_path=str(explicit_path),
            generated=False,
            source="existing",
        )

    src = Path(video_path)
    stem = src.stem

    inferred_candidates = [
        src.with_name(f"{stem}_en.json"),
        src.parent.parent / "meta" / f"{stem}_en.json",
        Path(output_dir) / f"{stem}_en.json",
    ]

    for candidate in inferred_candidates:
        if candidate.exists():
            return SubtitlePreparationResult(
                subtitle_path=str(candidate),
                generated=False,
                source="existing",
            )

    if mode_normalized == "none":
        return SubtitlePreparationResult(subtitle_path=None, generated=False, source=None)

    target = explicit_path or (Path(output_dir) / f"{stem}_en.json")

    errors: list[str] = []

    def _try_local() -> str:
        return transcribe_with_whisper_local(
            video_path=video_path,
            output_path=str(target),
            model=local_model,
            language=language,
        )

    def _try_api() -> str:
        return transcribe_with_openai_api(
            video_path=video_path,
            output_path=str(target),
            model=api_model,
            language=language,
        )

    order = []
    if mode_normalized == "local":
        order = [("whisper-local", _try_local)]
    elif mode_normalized == "api":
        order = [("openai-api", _try_api)]
    else:
        order = [("whisper-local", _try_local), ("openai-api", _try_api)]

    for source, producer in order:
        try:
            generated_path = producer()
            return SubtitlePreparationResult(
                subtitle_path=generated_path,
                generated=True,
                source=source,
            )
        except Exception as exc:
            errors.append(f"{source}: {exc}")

    return SubtitlePreparationResult(
        subtitle_path=None,
        generated=False,
        source=None,
        error="; ".join(errors),
    )
