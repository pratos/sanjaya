"""Load persisted traces from sanjaya_artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_traces(
    n: int = 5,
    artifacts_dir: str = "./sanjaya_artifacts",
) -> list[dict[str, Any]]:
    """Load the last N traces, sorted newest-first.

    Scans artifacts_dir for run directories containing trace.json,
    sorts by directory name (which is a timestamp), and returns
    the most recent N as parsed dicts.
    """
    base = Path(artifacts_dir)
    if not base.exists():
        return []

    traces = []
    for run_dir in sorted(base.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        trace_path = run_dir / "trace.json"
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                traces.append(trace)
            except (json.JSONDecodeError, OSError):
                continue
        if len(traces) >= n:
            break

    return traces


def load_trace(
    run_id: str,
    artifacts_dir: str = "./sanjaya_artifacts",
) -> dict[str, Any] | None:
    """Load a specific trace by run_id."""
    trace_path = Path(artifacts_dir) / run_id / "trace.json"
    if not trace_path.exists():
        return None
    return json.loads(trace_path.read_text(encoding="utf-8"))


def print_trace_summary(trace: dict[str, Any]) -> None:
    """Print a concise summary of a trace for terminal inspection."""
    print(f"Run: {trace.get('run_id', '?')}")
    print(f"Question: {trace.get('question', '?')}")
    print(f"Answer: {str(trace.get('answer', '?'))[:200]}")
    print(f"Model: {trace.get('model', '?')} | Vision: {trace.get('vision_model', '?')}")
    print(f"Iterations: {trace.get('iterations', '?')} | Wall time: {trace.get('wall_time_s', '?')}s")

    cost = trace.get("cost", {})
    print(
        f"Cost: ${cost.get('total_cost_usd', 0):.6f} "
        f"({cost.get('total_input_tokens', 0)} in / {cost.get('total_output_tokens', 0)} out)"
    )

    events = trace.get("events", [])
    orch_calls = [e for e in events if e.get("kind") == "sanjaya.orchestrator_call_end"]
    llm_calls = [e for e in events if e.get("kind") == "sanjaya.llm_call_end"]
    print(f"LLM calls: {len(orch_calls)} orchestrator + {len(llm_calls)} sub-LLM")
    print("---")
