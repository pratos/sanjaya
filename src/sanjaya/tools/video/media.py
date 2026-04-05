"""Media helpers for probing videos, extracting clips, and sampling frames.

Ported from video_tools/media.py — same ffmpeg/ffprobe logic.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MediaToolError(RuntimeError):
    """Raised when ffmpeg/ffprobe operations fail."""


def _require_binary(name: str) -> None:
    if shutil.which(name):
        return
    raise MediaToolError(f"Required binary '{name}' was not found in PATH")


def ffprobe_metadata(video_path: str) -> dict:
    """Return ffprobe metadata as JSON dict."""
    _require_binary("ffprobe")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,filename,size:stream=index,codec_type,width,height,r_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffprobe failed")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaToolError("ffprobe returned invalid JSON") from exc


def video_duration_seconds(video_path: str) -> float:
    """Return duration in seconds."""
    meta = ffprobe_metadata(video_path)
    duration = meta.get("format", {}).get("duration")
    try:
        return float(duration)
    except (TypeError, ValueError):
        raise MediaToolError(f"Could not parse duration: {duration!r}")


def get_video_info(video_path: str) -> dict:
    """Get video metadata: duration, resolution, codec, file size."""
    meta = ffprobe_metadata(video_path)
    fmt = meta.get("format", {})
    streams = meta.get("streams", [])

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})

    return {
        "duration_s": float(fmt.get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_type", "unknown"),
        "file_size_mb": round(int(fmt.get("size", 0)) / (1024 * 1024), 2),
    }


def extract_clip(video_path: str, start_s: float, end_s: float, output_path: str) -> str:
    """Extract a clip using ffmpeg and return output path."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))

    src = Path(video_path)
    dst = Path(output_path)
    if not src.exists():
        raise FileNotFoundError(f"Video not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg clip extraction failed")
    return str(dst)


def sample_frames(
    video_path: str,
    start_s: float,
    end_s: float,
    output_dir: str,
    *,
    max_frames: int = 8,
) -> list[str]:
    """Sample up to max_frames between start/end timestamps."""
    _require_binary("ffmpeg")
    start_s = max(0.0, float(start_s))
    end_s = max(start_s + 0.1, float(end_s))
    duration = max(0.1, end_s - start_s)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = max_frames / duration
    pattern = out_dir / "frame_%04d.jpg"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
        "-i", video_path,
        "-vf", f"fps={fps:.4f}",
        "-q:v", "2",
        str(pattern),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaToolError(result.stderr.strip() or "ffmpeg frame sampling failed")

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if len(frames) > max_frames:
        for extra in frames[max_frames:]:
            extra.unlink(missing_ok=True)
        frames = frames[:max_frames]

    return [str(path) for path in frames]
