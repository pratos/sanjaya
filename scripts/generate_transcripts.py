# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.0", "python-dotenv>=1.0"]
# ///
"""Generate transcript sidecar JSONs for all videos in data/ using OpenAI API.

Handles large files by extracting compressed audio and chunking if >24MB.
Skips videos that already have transcripts.

Usage:
    uv run python scripts/generate_transcripts.py
    uv run python scripts/generate_transcripts.py --model whisper-1
    uv run python scripts/generate_transcripts.py --dry-run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── config ──────────────────────────────────────────────────────────────
VIDEO_DIRS = [
    Path("data/longvideobench/videos"),
    Path("data/youtube"),
]
META_MAP = {
    "data/longvideobench/videos": Path("data/longvideobench/meta"),
    "data/youtube": Path("data/youtube/meta"),
}
MAX_CHUNK_MB = 24.0
LANGUAGE = "en"


# ── helpers ─────────────────────────────────────────────────────────────
def get_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def extract_audio(video: Path, out: Path) -> None:
    """Extract mono 16 kHz mp3 (very compact)."""
    subprocess.run(
        ["ffmpeg", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", "-f", "mp3", "-y", str(out)],
        check=True, capture_output=True,
    )


def split_audio(audio: Path) -> list[tuple[Path, float]]:
    """Split into ≤24 MB chunks if needed. Returns [(path, offset_sec), ...]."""
    size_mb = audio.stat().st_size / (1024 * 1024)
    if size_mb <= MAX_CHUNK_MB:
        return [(audio, 0.0)]

    duration = get_duration(audio)
    n = int(size_mb / MAX_CHUNK_MB) + 1
    seg_dur = duration / n
    chunks: list[tuple[Path, float]] = []
    for i in range(n):
        start = i * seg_dur
        chunk = audio.parent / f"chunk_{i:03d}.mp3"
        # Re-encode chunks (not -c copy) to avoid corrupt boundaries
        subprocess.run(
            [
                "ffmpeg", "-i", str(audio),
                "-ss", str(start), "-t", str(seg_dur),
                "-ac", "1", "-ar", "16000", "-b:a", "48k",
                "-y", str(chunk),
            ],
            check=True, capture_output=True,
        )
        chunks.append((chunk, start))
    return chunks


def transcribe_chunk(client: OpenAI, audio: Path, model: str) -> list[dict]:
    """Call OpenAI transcription API for one chunk."""
    kwargs: dict = {"model": model, "language": LANGUAGE}

    if model == "whisper-1":
        kwargs["response_format"] = "verbose_json"
        kwargs["timestamp_granularities"] = ["segment"]
    else:
        # gpt-4o-*-transcribe models: use json (no verbose_json support)
        kwargs["response_format"] = "json"

    with audio.open("rb") as f:
        kwargs["file"] = f
        resp = client.audio.transcriptions.create(**kwargs)

    raw = getattr(resp, "segments", None)
    if raw is None and isinstance(resp, dict):
        raw = resp.get("segments")
    return list(raw or [])


def transcribe_video(client: OpenAI, video: Path, model: str) -> dict:
    """Full pipeline: extract audio → chunk → transcribe → merge."""
    with tempfile.TemporaryDirectory(prefix="transcript-") as tmp:
        tmp_dir = Path(tmp)
        audio = tmp_dir / "audio.mp3"

        print(f"  ↳ extracting audio …")
        extract_audio(video, audio)
        audio_mb = audio.stat().st_size / (1024 * 1024)
        print(f"  ↳ audio: {audio_mb:.1f} MB")

        chunks = split_audio(audio)
        if len(chunks) > 1:
            print(f"  ↳ split into {len(chunks)} chunks (>{MAX_CHUNK_MB:.0f} MB)")

        all_segments: list[dict] = []
        for i, (chunk_path, offset) in enumerate(chunks):
            label = f"chunk {i + 1}/{len(chunks)}" if len(chunks) > 1 else "audio"
            print(f"  ↳ transcribing {label} …")

            for seg in transcribe_chunk(client, chunk_path, model):
                s = seg if isinstance(seg, dict) else seg.__dict__
                start = float(s.get("start", 0))
                end = float(s.get("end", 0))
                text = str(s.get("text", "")).strip()
                if end <= start or not text:
                    continue
                all_segments.append({
                    "start": round(start + offset, 3),
                    "end": round(end + offset, 3),
                    "text": text,
                })

    return {
        "segments": all_segments,
        "metadata": {
            "source": "openai-api",
            "model": model,
            "language": LANGUAGE,
            "video_path": str(video),
        },
    }


def output_path_for(video: Path) -> Path:
    """Determine the sidecar JSON path for a video."""
    for prefix, meta_dir in META_MAP.items():
        if str(video).startswith(prefix):
            return meta_dir / f"{video.stem}_en.json"
    # fallback: next to the video
    return video.with_name(f"{video.stem}_en.json")


def find_videos() -> list[tuple[Path, Path]]:
    """Return [(video_path, output_path)] for videos that need transcripts."""
    pairs: list[tuple[Path, Path]] = []
    for d in VIDEO_DIRS:
        if not d.exists():
            continue
        for v in sorted(d.glob("*.mp4")):
            out = output_path_for(v)
            if not out.exists():
                pairs.append((v, out))
    return pairs


# ── main ────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Generate transcript sidecars for all videos")
    p.add_argument("--model", default="gpt-4o-mini-transcribe", help="OpenAI transcription model")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    p.add_argument("--workers", type=int, default=3, help="Parallel workers (default: 3)")
    args = p.parse_args()

    pairs = find_videos()
    if not pairs:
        print("✓ All videos already have transcripts.")
        return

    print(f"Found {len(pairs)} video(s) needing transcripts (model: {args.model}):\n")
    for v, o in pairs:
        dur = get_duration(v)
        print(f"  {v}  ({dur / 60:.1f} min) → {o}")
    print()

    if args.dry_run:
        print("(dry run — exiting)")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Sort smallest first for fast early results
    pairs.sort(key=lambda p: p[0].stat().st_size)

    def process(video: Path, out: Path) -> str:
        print(f"\n▶ {video.name}")
        result = transcribe_video(client, video, args.model)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        n = len(result["segments"])
        return f"  ✓ {video.name}: {n} segments → {out}"

    # Run with thread pool for I/O-bound API calls
    results: list[str] = []
    if args.workers > 1 and len(pairs) > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, v, o): v.name for v, o in pairs}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    name = futures[fut]
                    results.append(f"  ✗ {name}: {exc}")
                    print(f"  ✗ {name}: {exc}", file=sys.stderr)
    else:
        for v, o in pairs:
            try:
                results.append(process(v, o))
            except Exception as exc:
                results.append(f"  ✗ {v.name}: {exc}")
                print(f"  ✗ {v.name}: {exc}", file=sys.stderr)

    print("\n" + "=" * 60)
    print("Done!\n")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
