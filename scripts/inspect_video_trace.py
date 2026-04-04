"""Inspect and summarize persisted VideoRLM trace events from artifact manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_manifests(artifacts_dir: Path) -> list[Path]:
    manifests = [p for p in artifacts_dir.glob("*/manifest.json") if p.is_file()]
    manifests.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return manifests


def _resolve_manifest(artifacts_dir: Path, manifest_path: str | None, run_id: str | None) -> tuple[Path, dict[str, Any]]:
    if manifest_path:
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        return path, _load_manifest(path)

    manifests = _find_manifests(artifacts_dir)
    if not manifests:
        raise FileNotFoundError(f"No manifests found under {artifacts_dir}")

    if run_id is None:
        path = manifests[0]
        return path, _load_manifest(path)

    for path in manifests:
        manifest = _load_manifest(path)
        events = manifest.get("trace_events", [])
        for event in events:
            payload = event.get("payload", {}) if isinstance(event, dict) else {}
            if payload.get("run_id") == run_id:
                return path, manifest

    raise FileNotFoundError(f"No manifest with trace run_id={run_id} under {artifacts_dir}")


def _as_num(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _event_info(kind: str, payload: dict[str, Any]) -> str:
    if kind == "code_instruction":
        return (payload.get("code_preview") or "").replace("\n", " ")[:80]
    if kind == "code_execution":
        err = (payload.get("stderr_preview") or "").strip()
        return "ok" if not err else f"stderr: {err[:60]}"
    if kind == "retrieval":
        return (
            f"sub={payload.get('subtitle_count', 0)} "
            f"slide={payload.get('sliding_count', 0)} "
            f"selected={payload.get('selected_count', 0)}"
        )
    if kind == "clip":
        return f"{payload.get('clip_id')} [{payload.get('start_s')}s-{payload.get('end_s')}s]"
    if kind == "vision":
        return (payload.get("response_preview") or "")[:80].replace("\n", " ")
    if kind == "transcription":
        return f"source={payload.get('source')} generated={payload.get('generated')}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect VideoRLM trace events")
    parser.add_argument("--artifacts-dir", default="data/longvideobench/artifacts")
    parser.add_argument("--manifest", default=None, help="Path to a specific manifest.json")
    parser.add_argument("--run-id", default=None, help="Trace run_id to select/filter")
    parser.add_argument("--json", action="store_true", help="Print trace events JSON only")
    args = parser.parse_args()

    manifest_path, manifest = _resolve_manifest(Path(args.artifacts_dir), args.manifest, args.run_id)
    events = manifest.get("trace_events", [])
    if not isinstance(events, list):
        events = []

    if args.run_id:
        events = [
            e
            for e in events
            if isinstance(e, dict) and isinstance(e.get("payload"), dict) and e["payload"].get("run_id") == args.run_id
        ]

    events.sort(
        key=lambda e: (
            (e.get("payload", {}) or {}).get("event_index", 10**9),
            e.get("timestamp", 0),
        )
    )

    if args.json:
        print(json.dumps(events, indent=2))
        return

    run_id = None
    if events:
        run_id = (events[0].get("payload") or {}).get("run_id")

    clips = manifest.get("clips", {})
    clip_count = len(clips) if isinstance(clips, dict) else 0
    window_count = len(manifest.get("candidate_windows", []) or [])

    total_input = sum(_as_num((e.get("payload") or {}).get("input_tokens")) for e in events)
    total_output = sum(_as_num((e.get("payload") or {}).get("output_tokens")) for e in events)
    total_cost = sum(_as_num((e.get("payload") or {}).get("cost_usd")) for e in events)

    print(f"Manifest: {manifest_path}")
    print(f"Run ID:   {run_id or '(unknown)'}")
    print(f"Events:   {len(events)} | Clips: {clip_count} | Windows: {window_count}")
    print(f"Tokens:   in={int(total_input)} out={int(total_output)}")
    print(f"Cost USD: {total_cost:.6f}")
    print()

    header = f"{'idx':>4} {'kind':<16} {'it':>3} {'blk':<7} {'phase':<14} {'cost':>10}  info"
    print(header)
    print("-" * len(header))

    for event in events:
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind", ""))
        payload = event.get("payload", {}) if isinstance(event.get("payload"), dict) else {}
        idx = payload.get("event_index", "")
        it = payload.get("iteration", "")

        block_index = payload.get("code_block_index")
        block_total = payload.get("code_block_total")
        blk = ""
        if block_index is not None and block_total is not None:
            blk = f"{block_index}/{block_total}"

        phase = str(payload.get("phase", ""))
        cost = payload.get("cost_usd")
        cost_str = f"{_as_num(cost):.6f}" if isinstance(cost, (int, float)) else ""
        info = _event_info(kind, payload)

        print(f"{str(idx):>4} {kind:<16} {str(it):>3} {blk:<7} {phase:<14} {cost_str:>10}  {info}")


if __name__ == "__main__":
    main()
