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
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MOONDREAM_CLOUD_BASE = "https://api.moondream.ai/v1"
MOONDREAM_STATION_BASE = "http://localhost:2020/v1"
TOKENS_PER_IMAGE = 729

# ── Global concurrency limiter ──────────────────────────────
# Caps total in-flight HTTP requests to Moondream across *all*
# MoondreamVisionClient instances in this process.  Cloud rate-limits
# more aggressively than a local Station, so the default is
# conservative.  Override via MOONDREAM_MAX_CONCURRENT env var.
_MAX_CONCURRENT = int(os.environ.get("MOONDREAM_MAX_CONCURRENT", "2"))
_global_semaphore = threading.BoundedSemaphore(_MAX_CONCURRENT)
logger.debug("Moondream global concurrency limit: %d", _MAX_CONCURRENT)


def is_moondream_spec(model: Any) -> bool:
    """Check if a model spec refers to Moondream (cloud or station)."""
    if isinstance(model, str):
        return model.startswith("moondream:") or model.startswith("moondream-station:")
    return isinstance(model, MoondreamVisionClient)


class MoondreamVisionClient:
    """Direct REST client for Moondream Cloud or local Station.

    Set ``base_url`` to point at a local Moondream Station
    (default ``http://localhost:2020/v1``) or cloud
    (``https://api.moondream.ai/v1``).  When targeting Station
    the API key is optional.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "moondream3-preview",
        max_workers: int | None = None,
        base_url: str | None = None,
        auth_token: str | None = None,
    ):
        import os

        self._api_key = api_key or os.environ.get("MOONDREAM_API_KEY", "")
        # Bearer token for self-hosted proxies (e.g. Modal deployment)
        self._auth_token = auth_token or os.environ.get("MOONDREAM_AUTH_TOKEN", "")
        # Determine base URL: explicit > env > cloud default
        self._base_url = (
            base_url
            or os.environ.get("MOONDREAM_BASE_URL")
            or MOONDREAM_CLOUD_BASE
        )
        self._is_local = "localhost" in self._base_url or "127.0.0.1" in self._base_url
        self._is_proxy = bool(self._auth_token) or (
            not self._is_local and self._base_url != MOONDREAM_CLOUD_BASE
        )

        if not self._api_key and not self._is_local and not self._is_proxy:
            raise ValueError(
                "Moondream API key required for cloud. Set MOONDREAM_API_KEY env var, "
                "pass api_key=, or point base_url= at a local Moondream Station / proxy."
            )

        self._model = model
        # Local station can handle more concurrency; cloud rate-limits aggressively
        self._max_workers = max_workers if max_workers is not None else (4 if self._is_local else 2)
        self._model_name = f"moondream/{model}"

        # Cumulative usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._last_had_metrics = False

    def _make_request(
        self, endpoint: str, body: dict[str, Any], *, retries: int = 2,
    ) -> dict[str, Any]:
        """Send a request to a Moondream API endpoint with retry.

        Acquires the module-level ``_global_semaphore`` before each HTTP
        attempt so the total number of in-flight requests across all
        client instances stays bounded.
        """
        import time as _time

        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        payload = json.dumps(body).encode()

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "User-Agent": "sanjaya/0.2.0",
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        if self._api_key:
            headers["X-Moondream-Auth"] = self._api_key

        last_err: Exception | None = None
        for attempt in range(1 + retries):
            _global_semaphore.acquire()
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as resp:
                    data = json.loads(resp.read())

                # Track actual token counts from API response
                metrics = data.get("metrics", {})
                if metrics:
                    self.total_input_tokens += metrics.get("input_tokens", TOKENS_PER_IMAGE)
                    self.total_output_tokens += metrics.get("output_tokens", 0)
                    self.total_calls += 1
                    self._last_had_metrics = True

                return data
            except (OSError, TimeoutError) as e:
                last_err = e
                if attempt < retries:
                    _time.sleep(1.5 * (attempt + 1))
                    logger.debug("Moondream %s retry %d after %s", endpoint, attempt + 1, e)
            finally:
                _global_semaphore.release()

        raise last_err  # type: ignore[misc]

    def _query_single(self, image_b64: str, question: str) -> str:
        """Send a single image+question to the Moondream query endpoint."""
        data = self._make_request("query", {
            "image_url": f"data:image/jpeg;base64,{image_b64}",
            "question": question,
            "model": self._model,
        })
        return data.get("answer", data.get("result", str(data)))

    def caption_frame(
        self,
        image_path: str | Path,
        *,
        length: str = "normal",
    ) -> str:
        """Caption a single frame using the /caption endpoint.

        Args:
            image_path: Path to a JPEG frame.
            length: Caption detail level — "short" (1-2 sentences) or
                    "normal" (detailed description).
        """
        b64 = self._load_and_encode(Path(image_path))
        data = self._make_request("caption", {
            "image_url": f"data:image/jpeg;base64,{b64}",
            "length": length,
            "model": self._model,
            "stream": False,
        })
        return data.get("caption", data.get("result", str(data)))

    def caption_frames_batch(
        self,
        frame_paths: list[str],
        *,
        length: str = "normal",
    ) -> list[str]:
        """Caption multiple frames concurrently via the /caption endpoint."""
        valid = [Path(p) for p in frame_paths if Path(p).exists()]
        if not valid:
            return []

        def _caption_one(idx: int, path: Path) -> tuple[int, str]:
            return idx, self.caption_frame(str(path), length=length)

        captions: dict[int, str] = {}
        workers = min(self._max_workers, len(valid))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_caption_one, i, p): i
                for i, p in enumerate(valid)
            }
            for fut in as_completed(futures):
                try:
                    idx, caption = fut.result()
                    captions[idx] = caption
                except Exception as e:
                    idx = futures[fut]
                    captions[idx] = f"[caption error: {e}]"
                    logger.warning("Moondream caption failed for frame %d: %s", idx, e)

        return [captions[i] for i in range(len(valid))]

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
