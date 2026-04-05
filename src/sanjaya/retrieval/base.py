"""RetrievalBackend ABC — pluggable retrieval for semantic/keyword search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrievalBackend(ABC):
    """Pluggable retrieval backend for semantic/keyword search."""

    @abstractmethod
    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        collection: str = "default",
    ) -> None:
        """Index documents. Metadata is stored alongside for filtering."""

    @abstractmethod
    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        collection: str = "default",
        filter_condition: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search indexed documents.

        Returns: [{"text": str, "score": float, "metadata": dict, "doc_id": int}]
        """

    @abstractmethod
    def delete(
        self,
        *,
        collection: str = "default",
        condition: str | None = None,
    ) -> int:
        """Delete documents. Returns count deleted."""
