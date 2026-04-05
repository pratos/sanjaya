"""SSE event formatting helpers."""

from __future__ import annotations

import json
from typing import Any


def format_sse_event(kind: str, timestamp: float, payload: dict[str, Any]) -> dict[str, str]:
    """Format a trace event as an SSE-compatible dict for sse-starlette."""
    data = json.dumps({"kind": kind, "timestamp": timestamp, "payload": payload})
    return {"event": kind, "data": data}


def format_heartbeat() -> dict[str, str]:
    """Format a heartbeat SSE event."""
    return {"event": "heartbeat", "data": "{}"}
