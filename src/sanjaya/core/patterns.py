"""PatternMemory — learn what code works/fails in the REPL across runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


class PatternMemory:
    """Learn what code works/fails in the REPL across runs.

    Stores patterns in a SQLite database. Successful patterns are
    injected into the system prompt as worked examples. Failed patterns
    are injected as warnings.
    """

    def __init__(self, path: str = ".sanjaya/patterns.db"):
        if path == ":memory:":
            self._conn = sqlite3.connect(":memory:")
        else:
            db_path = Path(path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(db_path))
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS successes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                tools_used TEXT NOT NULL DEFAULT '[]',
                description TEXT,
                timestamp REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                error_type TEXT NOT NULL,
                error_message TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
        """)
        self._conn.commit()

    def record_success(
        self,
        code: str,
        tools_used: list[str],
        description: str | None = None,
    ) -> None:
        """Record a code block that executed successfully."""
        self._conn.execute(
            "INSERT INTO successes (code, tools_used, description, timestamp) VALUES (?, ?, ?, ?)",
            (code, json.dumps(tools_used), description, time.time()),
        )
        self._conn.commit()

    def record_failure(
        self,
        code: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Record a code block that failed."""
        self._conn.execute(
            "INSERT INTO failures (code, error_type, error_message, timestamp) VALUES (?, ?, ?, ?)",
            (code, error_type, error_message, time.time()),
        )
        self._conn.commit()

    def get_examples(
        self,
        tools: list[str],
        limit: int = 3,
    ) -> list[dict]:
        """Get proven code examples relevant to the registered tools."""
        # Get recent successes that used any of the specified tools
        cursor = self._conn.execute(
            "SELECT code, tools_used, description FROM successes ORDER BY timestamp DESC LIMIT ?",
            (limit * 3,),  # Fetch extra to filter
        )

        results = []
        tools_set = set(tools)
        for row in cursor:
            code, tools_json, description = row
            used = set(json.loads(tools_json))
            if used & tools_set:
                results.append({
                    "code": code,
                    "description": description,
                    "tools_used": list(used),
                })
                if len(results) >= limit:
                    break

        return results

    def get_anti_patterns(self, limit: int = 5) -> list[dict]:
        """Get common failure patterns to warn about."""
        cursor = self._conn.execute(
            "SELECT code, error_type, error_message FROM failures ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

        results = []
        for row in cursor:
            code, error_type, error_message = row
            # Truncate code for prompt injection
            code_snippet = code[:200] + "..." if len(code) > 200 else code
            results.append({
                "code_snippet": code_snippet,
                "error": f"{error_type}: {error_message}",
                "hint": f"This code pattern caused {error_type}. Avoid similar approaches.",
            })

        return results

    def close(self) -> None:
        self._conn.close()
