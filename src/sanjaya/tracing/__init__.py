"""Unified tracing for sanjaya agents."""

from .events import EventBuffer
from .loader import load_trace, load_traces, print_trace_summary
from .tracer import TraceContext, Tracer

# Global tracer instance (backward compat with old sanjaya.tracing module)
_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


__all__ = [
    "EventBuffer",
    "TraceContext",
    "Tracer",
    "get_tracer",
    "load_trace",
    "load_traces",
    "print_trace_summary",
]
