"""Unified LLM client with text, vision, and batched support.

Merges the old LLMClient (text-only) and VideoLLMClient (text + vision)
into a single class. Adds concurrent batched completions.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent

from ..settings import get_settings
from .types import CallMetadata, UsageSnapshot


class LLMClient:
    """Unified LLM client with text, vision, and batched support."""

    def __init__(
        self,
        model: str,
        vision_model: str | None = None,
        fallback_model: str | None = None,
        api_key: str | None = None,
    ):
        self.model = model
        self.vision_model = vision_model or model
        self.fallback_model = fallback_model
        self._api_key = api_key

        self.last_usage: UsageSnapshot | None = None
        self.last_call_metadata: CallMetadata | None = None

    # ── Text completions ────────────────────────────────────

    def completion(self, prompt_or_messages: Any, timeout: int = 300) -> str:
        """Single text completion."""
        prompt = self._as_prompt(prompt_or_messages)
        return self._call(self.model, prompt, timeout=timeout)

    def completion_batched(self, prompts: list[str], timeout: int = 300) -> list[str]:
        """Concurrent text completions via asyncio.gather."""
        return self._run_batched([
            {"model": self.model, "payload": p, "timeout": timeout}
            for p in prompts
        ])

    # ── Vision completions ──────────────────────────────────

    def vision_completion(
        self,
        *,
        prompt: str,
        frame_paths: list[str] | None = None,
        clip_paths: list[str] | None = None,
        timeout: int = 300,
    ) -> str:
        """Single vision completion with image/video attachments."""
        user_content = self._build_vision_content(prompt, frame_paths, clip_paths)
        return self._call(self.vision_model, user_content, timeout=timeout)

    def vision_completion_batched(
        self,
        queries: list[dict[str, Any]],
        timeout: int = 300,
    ) -> list[str]:
        """Concurrent vision completions.

        Each query dict has keys: prompt, frame_paths, clip_paths.
        """
        return self._run_batched([
            {
                "model": self.vision_model,
                "payload": self._build_vision_content(
                    q["prompt"],
                    q.get("frame_paths"),
                    q.get("clip_paths"),
                ),
                "timeout": timeout,
            }
            for q in queries
        ])

    # ── Cost helpers ────────────────────────────────────────

    @property
    def last_cost_usd(self) -> float | None:
        if self.last_call_metadata:
            return self.last_call_metadata.cost_usd
        return None

    # ── Internal ────────────────────────────────────────────

    def _as_prompt(self, payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return str(payload.get("content", payload))
        if isinstance(payload, list):
            parts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    parts.append(f"{role.upper()}:\n{content}")
                else:
                    parts.append(str(item))
            return "\n\n".join(parts)
        return str(payload)

    def _build_vision_content(
        self,
        prompt: str,
        frame_paths: list[str] | None,
        clip_paths: list[str] | None,
    ) -> list[Any]:
        """Build multimodal user content list for vision models."""
        user_content: list[Any] = [prompt]

        valid_frames = [Path(p) for p in (frame_paths or []) if Path(p).exists()]
        valid_clips = [Path(p) for p in (clip_paths or []) if Path(p).exists()]

        for frame_path in valid_frames[:8]:
            user_content.append(
                BinaryContent(data=frame_path.read_bytes(), media_type="image/jpeg")
            )

        if not valid_frames:
            for clip_path in valid_clips[:1]:
                user_content.append(
                    BinaryContent(data=clip_path.read_bytes(), media_type="video/mp4")
                )

        if not valid_frames and not valid_clips:
            user_content.append("No visual attachments were available.")

        return user_content

    def _ensure_api_key(self) -> None:
        """Populate provider API key from explicit arg or settings into env vars."""
        settings = get_settings()
        provider = self.model.split(":", 1)[0] if ":" in self.model else "unknown"

        if provider == "openrouter":
            key = self._api_key or settings.openrouter_api_key
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
        elif provider.startswith("openai"):
            key = self._api_key or settings.openai_api_key
            if key:
                os.environ["OPENAI_API_KEY"] = key
        elif provider.startswith("anthropic"):
            key = self._api_key or settings.anthropic_api_key
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key

    def _run_agent(self, model: str, payload: Any) -> tuple[str, Any]:
        agent = Agent(model=model, output_type=str, retries=1, defer_model_check=True)
        result = agent.run_sync(payload)
        response = result.output if hasattr(result, "output") else str(result)
        return str(response), result

    async def _run_agent_async(self, model: str, payload: Any) -> tuple[str, Any]:
        agent = Agent(model=model, output_type=str, retries=1, defer_model_check=True)
        result = await agent.run(payload)
        response = result.output if hasattr(result, "output") else str(result)
        return str(response), result

    def _resolve_usage(self, usage_obj: Any) -> Any:
        if callable(usage_obj):
            try:
                return usage_obj()
            except Exception:
                return None
        return usage_obj

    def _capture_usage(self, result: Any) -> UsageSnapshot:
        usage = self._resolve_usage(getattr(result, "usage", None))
        if usage is None:
            response = getattr(result, "response", None)
            usage = self._resolve_usage(getattr(response, "usage", None))

        if usage is None:
            snap = UsageSnapshot()
            self.last_usage = snap
            return snap

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))

        snap = UsageSnapshot(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
        self.last_usage = snap
        return snap

    def _extract_cost_usd(self, result: Any) -> float | None:
        response = getattr(result, "response", None)
        if response is None:
            return None

        provider_details = getattr(response, "provider_details", None)
        if isinstance(provider_details, dict):
            for key in ("cost", "total_cost", "upstream_inference_cost"):
                value = provider_details.get(key)
                if isinstance(value, (int, float)):
                    return float(value)

        response_cost = getattr(response, "cost", None)
        if callable(response_cost):
            try:
                computed = response_cost()
                if isinstance(computed, (int, float)):
                    return float(computed)
            except Exception:
                return None

        if isinstance(response_cost, (int, float)):
            return float(response_cost)

        return None

    def _capture_metadata(self, model: str, result: Any, start: float, fallback_used: bool = False) -> CallMetadata:
        response = getattr(result, "response", None)
        model_name = getattr(response, "model_name", None) if response else None
        provider_name = getattr(response, "provider_name", None) if response else None
        provider_response_id = getattr(response, "provider_response_id", None) if response else None
        cost_usd = self._extract_cost_usd(result)

        provider = model.split(":", 1)[0] if ":" in model else "unknown"
        meta = CallMetadata(
            requested_model=model,
            model_used=model_name or model,
            provider=provider_name or provider,
            provider_response_id=provider_response_id,
            fallback_used=fallback_used,
            duration_seconds=round(time.time() - start, 3),
            cost_usd=round(cost_usd, 8) if cost_usd is not None else None,
        )
        self.last_call_metadata = meta
        return meta

    def _call(self, model: str, payload: Any, timeout: int = 300) -> str:
        """Core call with fallback."""
        _ = timeout  # Reserved for future provider-specific timeout support
        start = time.time()
        self.last_usage = None
        self.last_call_metadata = None
        self._ensure_api_key()

        try:
            response, result = self._run_agent(model, payload)
            self._capture_usage(result)
            self._capture_metadata(model, result, start)
            return response
        except Exception as primary_err:
            if not self.fallback_model:
                raise

            # Fallback
            response, result = self._run_agent(self.fallback_model, payload)
            self._capture_usage(result)
            meta = self._capture_metadata(self.fallback_model, result, start, fallback_used=True)
            meta.primary_error = str(primary_err)
            return response

    def _run_batched(self, calls: list[dict[str, Any]]) -> list[str]:
        """Run multiple calls concurrently."""
        self._ensure_api_key()

        async def _gather():
            tasks = []
            for call in calls:
                tasks.append(self._run_agent_async(call["model"], call["payload"]))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in async context — use nest_asyncio pattern or thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _gather())
                raw_results = future.result()
        else:
            raw_results = asyncio.run(_gather())

        responses: list[str] = []
        for r in raw_results:
            if isinstance(r, Exception):
                responses.append(f"Error: {r}")
            else:
                response_text, _ = r
                responses.append(response_text)
        return responses
