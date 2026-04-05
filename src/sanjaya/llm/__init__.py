"""LLM layer — unified client with text, vision, and batched support."""

from .client import LLMClient, ModelSpec
from .types import CallMetadata, UsageSnapshot

__all__ = [
    "CallMetadata",
    "LLMClient",
    "ModelSpec",
    "UsageSnapshot",
]
