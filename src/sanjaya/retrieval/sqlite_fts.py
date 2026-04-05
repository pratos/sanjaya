"""SQLite FTS5 full-text search backend — zero external dependencies."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .base import RetrievalBackend


class SQLiteFTSBackend(RetrievalBackend):
    """SQLite FTS5 full-text search. Default backend.

    - Zero external dependencies (sqlite3 is stdlib)
    - Indexing: <100ms for hundreds of segments
    - Persists to disk — cross-run memory out of the box
    - BM25 ranking built into FTS5
    """

    def __init__(self, path: str = ".sanjaya/retrieval.db"):
        if path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            db_path = Path(path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                content,
                content='documents',
                content_rowid='doc_id'
            );
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, content) VALUES (new.doc_id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, content) VALUES ('delete', old.doc_id, old.content);
            END;
        """)
        self._conn.commit()

    def index(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | None = None,
        collection: str = "default",
    ) -> None:
        meta_list = metadata or [{}] * len(documents)
        rows = [
            (collection, doc, json.dumps(meta))
            for doc, meta in zip(documents, meta_list)
        ]
        self._conn.executemany(
            "INSERT INTO documents (collection, content, metadata) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        collection: str = "default",
        filter_condition: str | None = None,
    ) -> list[dict[str, Any]]:
        # Use FTS5 MATCH with BM25 ranking
        sql = """
            SELECT d.doc_id, d.content, d.metadata, rank
            FROM documents_fts f
            JOIN documents d ON d.doc_id = f.rowid
            WHERE documents_fts MATCH ?
              AND d.collection = ?
            ORDER BY rank
            LIMIT ?
        """
        try:
            cursor = self._conn.execute(sql, (query, collection, top_k))
        except sqlite3.OperationalError:
            # Query might have FTS5 syntax issues — return empty
            return []

        results: list[dict[str, Any]] = []
        for row in cursor:
            doc_id, content, metadata_json, rank = row
            results.append({
                "text": content,
                "score": -rank,  # FTS5 rank is negative (lower = better)
                "metadata": json.loads(metadata_json),
                "doc_id": doc_id,
            })
        return results

    def delete(
        self,
        *,
        collection: str = "default",
        condition: str | None = None,
    ) -> int:
        if condition:
            # Simple metadata-based condition (key=value)
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE collection = ? AND metadata LIKE ?",
                (collection, f"%{condition}%"),
            )
        else:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE collection = ?",
                (collection,),
            )
        self._conn.commit()
        return cursor.rowcount

    def collections(self) -> list[str]:
        """List all indexed collections."""
        cursor = self._conn.execute("SELECT DISTINCT collection FROM documents")
        return [row[0] for row in cursor]

    def count(self, collection: str = "default") -> int:
        """Count documents in a collection."""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM documents WHERE collection = ?",
            (collection,),
        )
        return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()
