"""Run VideoRLM on a local long video with an open-ended question."""

from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from video_rlm.video_rlm_repl import VideoRLM_REPL


def _to_markdown(payload: dict) -> str:
    lines = [
        "# VideoRLM Answer",
        "",
        f"**Question:** {payload.get('question', '')}",
        "",
        f"**Answer:** {payload.get('answer', '')}",
        "",
        "## Evidence",
    ]

    evidence = payload.get("evidence", [])
    if not evidence:
        lines.append("- (none)")
    else:
        for item in evidence:
            start_s = item.get("start_s", 0)
            end_s = item.get("end_s", 0)
            rationale = item.get("rationale", "")
            clip_path = item.get("clip_path")
            lines.append(f"- `{start_s:.2f}s - {end_s:.2f}s`: {rationale}")
            if clip_path:
                lines.append(f"  - clip: `{clip_path}`")
            frame_paths = item.get("frame_paths", [])
            if frame_paths:
                lines.append(f"  - frames: {', '.join(frame_paths)}")

    lines.extend(["", "## Retrieval Trace"])
    for window in payload.get("retrieval_trace", []):
        lines.append(
            f"- `{window.get('window_id')}` {window.get('strategy')} "
            f"[{window.get('start_s')}s, {window.get('end_s')}s] score={window.get('score')}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoRLM local long-video QnA")
    parser.add_argument("--video", required=True, help="Path to local video file")
    parser.add_argument("--question", required=True, help="Open-ended question")
    parser.add_argument("--subtitle", default=None, help="Optional subtitle JSON sidecar path")
    parser.add_argument(
        "--subtitle-mode",
        choices=["auto", "local", "api", "none"],
        default="auto",
        help="How to resolve subtitles when sidecar is missing",
    )
    parser.add_argument("--subtitle-local-model", default="base", help="Local whisper model name")
    parser.add_argument("--subtitle-api-model", default="gpt-4o-mini-transcribe", help="API transcription model")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    args = parser.parse_args()

    repl = VideoRLM_REPL(max_iterations=args.max_iterations)
    answer = repl.completion(
        video_path=args.video,
        question=args.question,
        subtitle_path=args.subtitle,
        subtitle_mode=args.subtitle_mode,
        subtitle_local_model=args.subtitle_local_model,
        subtitle_api_model=args.subtitle_api_model,
    )

    payload = answer.model_dump()
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(_to_markdown(payload))


if __name__ == "__main__":
    main()
