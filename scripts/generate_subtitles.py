"""Generate subtitle sidecars for videos using parakeet-mlx.

Usage:
    uv run --with parakeet-mlx python scripts/generate_subtitles.py VIDEO [VIDEO ...]
    uv run --with parakeet-mlx python scripts/generate_subtitles.py data/youtube/*.mp4
    uv run --with parakeet-mlx python scripts/generate_subtitles.py --all-youtube
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "youtube"


CHUNK_DURATION_S = 600  # 10 minutes per chunk — safe for Metal buffer limits


def get_duration(path: Path) -> float:
    """Get audio/video duration in seconds via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def extract_audio(video_path: Path, audio_path: Path, *, ss: float = 0, duration: float | None = None) -> None:
    """Extract mono 16kHz WAV from video via ffmpeg."""
    cmd = ["ffmpeg", "-y", "-ss", str(ss)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(video_path), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
    subprocess.run(cmd, check=True, capture_output=True)


def _extract_segments(result) -> list[dict]:
    """Pull segment dicts from a parakeet result object."""
    if hasattr(result, "sentences") and result.sentences:
        items = result.sentences
    elif hasattr(result, "segments") and result.segments:
        items = result.segments
    else:
        return [{"start": 0.0, "end": 0.0, "text": result.text.strip()}]
    return [
        {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text.strip()}
        for s in items
    ]


def transcribe(video_path: Path, tmp_dir: Path) -> list[dict]:
    """Transcribe a video, chunking if needed to stay within Metal limits."""
    from parakeet_mlx import from_pretrained

    model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    total_dur = get_duration(video_path)

    if total_dur <= CHUNK_DURATION_S:
        audio_path = tmp_dir / "full.wav"
        extract_audio(video_path, audio_path)
        return _extract_segments(model.transcribe(str(audio_path)))

    # Chunked transcription for long videos
    segments: list[dict] = []
    offset = 0.0
    chunk_idx = 0
    while offset < total_dur:
        chunk_dur = min(CHUNK_DURATION_S, total_dur - offset)
        chunk_path = tmp_dir / f"chunk_{chunk_idx}.wav"
        print(f"    chunk {chunk_idx}: {offset:.0f}s – {offset + chunk_dur:.0f}s")
        extract_audio(video_path, chunk_path, ss=offset, duration=chunk_dur)

        result = model.transcribe(str(chunk_path))
        for seg in _extract_segments(result):
            segments.append({
                "start": round(seg["start"] + offset, 3),
                "end": round(seg["end"] + offset, 3),
                "text": seg["text"],
            })

        chunk_path.unlink(missing_ok=True)
        offset += chunk_dur
        chunk_idx += 1

    return segments


def sidecar_path(video: Path) -> Path:
    """Return the canonical sidecar path: meta/{stem}_en.json."""
    meta_dir = video.parent / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir / f"{video.stem}_en.json"


def already_has_sidecar(video: Path) -> bool:
    """Check if a subtitle sidecar already exists."""
    stem = video.stem
    candidates = [
        video.with_name(f"{stem}_en.json"),
        video.parent / "meta" / f"{stem}_en.json",
        video.parent.parent / "meta" / f"{stem}_en.json",
    ]
    return any(c.exists() for c in candidates)


def process_video(video: Path, force: bool = False) -> Path | None:
    """Generate subtitle sidecar for a single video. Returns output path or None if skipped."""
    if not force and already_has_sidecar(video):
        print(f"  skip (sidecar exists): {video.name}")
        return None

    print(f"  transcribing: {video.name}")
    start = time.time()

    with tempfile.TemporaryDirectory() as tmp_dir:
        segments = transcribe(video, Path(tmp_dir))

    elapsed = time.time() - start

    out = sidecar_path(video)
    payload = {
        "segments": segments,
        "metadata": {
            "source": "parakeet-mlx",
            "model": "mlx-community/parakeet-tdt-0.6b-v3",
            "language": "en",
            "video_path": str(video),
            "generation_time_s": round(elapsed, 2),
        },
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  done ({len(segments)} segments, {elapsed:.1f}s) -> {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitle sidecars with parakeet-mlx")
    parser.add_argument("videos", nargs="*", type=Path, help="Video files to transcribe")
    parser.add_argument("--all-youtube", action="store_true", help="Process all mp4s in data/youtube/")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sidecars")
    args = parser.parse_args()

    videos: list[Path] = []
    if args.all_youtube:
        videos = sorted(DATA_DIR.glob("*.mp4"))
    elif args.videos:
        videos = [v.resolve() for v in args.videos]
    else:
        parser.print_help()
        sys.exit(1)

    if not videos:
        print("No video files found.")
        sys.exit(1)

    print(f"Processing {len(videos)} video(s)...\n")
    generated = 0
    for video in videos:
        if not video.exists():
            print(f"  NOT FOUND: {video}")
            continue
        result = process_video(video, force=args.force)
        if result:
            generated += 1

    print(f"\nDone. Generated {generated} sidecar(s).")


if __name__ == "__main__":
    main()
