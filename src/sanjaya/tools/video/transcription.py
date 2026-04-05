"""Subtitle transcription — re-exports from the original module.

The actual transcription logic lives in video_tools/transcription.py.
This module provides the same interface for the new toolkit structure.
"""

from __future__ import annotations

# Re-export from original module (will be inlined in a future cleanup)
from ...video_tools.transcription import (
    SubtitlePreparationResult,
    TranscriptionError,
    ensure_subtitle_sidecar,
)

__all__ = [
    "SubtitlePreparationResult",
    "TranscriptionError",
    "ensure_subtitle_sidecar",
]
