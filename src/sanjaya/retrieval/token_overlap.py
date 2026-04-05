"""Token overlap retrieval backend — legacy, in-memory only."""

from __future__ import annotations

import math
import re
from typing import Any

from .base import RetrievalBackend

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> set[str]:
    return {tok for tok in _WORD_RE.findall(text.lower()) if len(tok) > 1}


class TokenOverlapBackend(RetrievalBackend):
    """Simple in-memory token overlap scoring. No persistence."""

    def __init__(self) -> None:
        self._collections: dict[str, list[dict[str, Any]]] = {}

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        collection: str = "default",
    ) -> None:
        meta_list = metadata or [{}] * len(documents)
        if collection not in self._collections:
            self._collections[collection] = []

        for i, (doc, meta) in enumerate(zip(documents, meta_list)):
            self._collections[collection].append({
                "doc_id": len(self._collections[collection]) + i,
                "text": doc,
                "metadata": meta,
                "tokens": _tokenize(doc),
            })

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        collection: str = "default",
        filter_condition: str | None = None,
    ) -> list[dict[str, Any]]:
        docs = self._collections.get(collection, [])
        if not docs:
            return []

        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored = []
        for doc in docs:
            c_tokens = doc["tokens"]
            if not c_tokens:
                continue
            shared = len(q_tokens & c_tokens)
            score = shared / math.sqrt(len(q_tokens) * len(c_tokens))
            if score > 0:
                scored.append({
                    "text": doc["text"],
                    "score": round(score, 4),
                    "metadata": doc["metadata"],
                    "doc_id": doc["doc_id"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def delete(
        self,
        *,
        collection: str = "default",
        condition: str | None = None,
    ) -> int:
        if collection in self._collections:
            count = len(self._collections[collection])
            del self._collections[collection]
            return count
        return 0
