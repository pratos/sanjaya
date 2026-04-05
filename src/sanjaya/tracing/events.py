"""In-memory event buffer for SSE polling."""

from __future__ import annotations

import time
from typing import Any


class EventBuffer:
    """In-memory event buffer for SSE streaming."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []

    def emit(self, kind: str, **payload: Any) -> None:
        """Emit a named event."""
        self._events.append({
            "kind": kind,
            "timestamp": time.time(),
            **payload,
        })

    @property
    def events(self) -> list[dict[str, Any]]:
        """All emitted events."""
        return list(self._events)

    def events_since(self, index: int) -> list[dict[str, Any]]:
        """Events emitted since a given index."""
        return self._events[index:]

    def clear(self) -> None:
        self._events.clear()
