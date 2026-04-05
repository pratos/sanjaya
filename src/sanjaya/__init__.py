"""Sanjaya — extensible RLM agent framework with video understanding."""

# Patch event loop early so pydantic-ai works inside Jupyter notebooks.
from .llm.client import _patch_event_loop as _patch
_patch()
del _patch

from .agent import Agent
from .answer import Answer, Evidence
from .llm.client import ModelSpec
from .tools.base import Toolkit, tool

__all__ = [
    "Agent",
    "Answer",
    "Evidence",
    "ModelSpec",
    "Toolkit",
    "tool",
]
