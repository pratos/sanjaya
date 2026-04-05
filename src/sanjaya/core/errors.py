"""Monty error hints and smart feedback formatting."""

from __future__ import annotations

from typing import Any

MONTY_HINTS: dict[str, str] = {
    "ModuleNotFoundError": (
        "This module is not available in the sandboxed REPL. "
        "Use the provided tools for external operations: "
        "get_video_info() for metadata, extract_clip() for media, "
        "save_note() for file output, llm_query() for LLM calls."
    ),
    "FileNotFoundError": (
        "Direct file access is not available. Use the provided tools: "
        "get_video_info(), extract_clip(), sample_frames(), save_note()."
    ),
    "PermissionError": (
        "The REPL is sandboxed. Use save_note() or save_data() to write files."
    ),
    "ImportError": (
        "Import statements are not available in the sandboxed REPL. "
        "Use the provided tools instead. Available modules: math, re, json, "
        "collections, itertools, functools."
    ),
    "OSError": (
        "OS operations are not available in the sandbox. Use the provided tools."
    ),
}


def format_error_with_hints(
    exc: Exception,
    registry: Any | None = None,
) -> str:
    """Format an error with Monty-specific recovery hints.

    Includes: the error message, a hint based on error type,
    and a list of available tools that might help.
    """
    error_type = type(exc).__name__
    error_msg = str(exc)

    parts = [f"Error: {error_type}: {error_msg}"]

    hint = MONTY_HINTS.get(error_type)
    if hint:
        parts.append(f"Hint: {hint}")

    if registry is not None:
        tool_names = [t.name for t in registry.all_tools()]
        if tool_names:
            parts.append(f"Available tools: {', '.join(tool_names)}")

    return "\n".join(parts)
