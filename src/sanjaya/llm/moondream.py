"""Moondream vision backend — cheap per-frame VLM queries.

Uses the Moondream Cloud REST API directly (no SDK dependency) so we
avoid the kestrel-native wheel that doesn't build on Python 3.14.

Moondream accepts one image per query, so for multi-frame analysis we
fan out concurrent requests via ThreadPoolExecutor and aggregate the
per-frame answers into a single structured response.

Each image costs a fixed 729 tokens regardless of resolution, making
Moondream ~4-8x cheaper than GPT-4.1-mini for vision-heavy workloads.
"""

from __future__ import annotations

import base64
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MOONDREAM_API_URL = "https://api.moondream.ai/v1/query"
TOKENS_PER_IMAGE = 729


def is_moondream_spec(model: Any) -> bool:
    """Check if a model spec refers to Moondream."""
    if isinstance(model, str):
        return model.startswith("moondream:")
    return isinstance(model, MoondreamVisionClient)


class MoondreamVisionClient:
    """Direct REST client for Moondream Cloud vision queries."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "moondream3-preview",
        max_workers: int = 4,
    ):
        import os

        self._api_key = api_key or os.environ.get("MOONDREAM_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Moondream API key required. Set MOONDREAM_API_KEY env var "
                "or pass api_key= to MoondreamVisionClient."
            )

        self._model = model
        self._max_workers = max_workers
        self._model_name = f"moondream/{model}"

        # Cumulative usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._last_had_metrics = False

    def _query_single(self, image_b64: str, question: str) -> str:
        """Send a single image+question to the Moondream REST API."""
        payload = json.dumps({
            "image_url": f"data:image/jpeg;base64,{image_b64}",
            "question": question,
            "model": self._model,
        }).encode()

        req = urllib.request.Request(
            MOONDREAM_API_URL,
            data=payload,
            headers={
                "X-Moondream-Auth": self._api_key,
                "Content-Type": "application/json",
                "User-Agent": "sanjaya/0.2.0",
            },
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())

        # Track actual token counts from API response
        metrics = data.get("metrics", {})
        if metrics:
            self.total_input_tokens += metrics.get("input_tokens", TOKENS_PER_IMAGE)
            self.total_output_tokens += metrics.get("output_tokens", 0)
            self.total_calls += 1
            self._last_had_metrics = True

        return data.get("answer", data.get("result", str(data)))

    def _load_and_encode(self, path: Path) -> str:
        """Read a JPEG file and return its base64 encoding."""
        return base64.b64encode(path.read_bytes()).decode()

    def query_frames(
        self,
        prompt: str,
        frame_paths: list[str],
        *,
        max_frames: int = 8,
    ) -> str:
        """Query Moondream about multiple frames, aggregating results.

        Single frame: returns the direct answer.
        Multiple frames: returns indexed per-frame answers for the
        orchestrator to synthesize.
        """
        valid = [Path(p) for p in frame_paths if Path(p).exists()][:max_frames]
        if not valid:
            return "No valid frames provided."

        if len(valid) == 1:
            b64 = self._load_and_encode(valid[0])
            answer = self._query_single(b64, prompt)
            self._track_usage(1, answer)
            return answer

        # Fan out concurrent per-frame queries
        def _query_one(idx: int, path: Path) -> tuple[int, str]:
            b64 = self._load_and_encode(path)
            return idx, self._query_single(b64, prompt)

        answers: dict[int, str] = {}
        workers = min(self._max_workers, len(valid))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_query_one, i, p): i
                for i, p in enumerate(valid)
            }
            for fut in as_completed(futures):
                try:
                    idx, answer = fut.result()
                    answers[idx] = answer
                except Exception as e:
                    idx = futures[fut]
                    answers[idx] = f"[error: {e}]"
                    logger.warning("Moondream query failed for frame %d: %s", idx, e)

        self._track_usage(len(valid), *answers.values())

        parts = []
        for i in sorted(answers):
            frame_name = valid[i].stem
            parts.append(f"[Frame {i} ({frame_name})]: {answers[i]}")
        return "\n".join(parts)

    def _track_usage(self, n_images: int, *responses: str) -> None:
        # Skip if _query_single already tracked via API metrics
        if self._last_had_metrics:
            self._last_had_metrics = False
            return

    @property
    def model_name(self) -> str:
        return self._model_name
