"""Unified LLM client with text, vision, and batched support.

Merges the old LLMClient (text-only) and VideoLLMClient (text + vision)
into a single class. Adds concurrent batched completions.

Accepts either a pydantic-ai ``Model`` object (with provider/auth
already configured) or a plain model string like ``"openrouter:openai/gpt-4o"``.
When a string is given, pydantic-ai resolves it using its built-in provider
registry and the corresponding environment variables.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models import Model

from .types import CallMetadata, UsageSnapshot

# Type alias: callers can pass a pre-configured Model *or* a provider string.
ModelSpec = Model | str


def _compress_frame(frame_path: Path, max_dim: int = 768, quality: int = 60) -> bytes:
    """Resize and re-compress a JPEG frame to reduce token cost.

    Downscales the longest side to *max_dim* pixels (preserving aspect ratio)
    and re-encodes at *quality* (1-95, lower = smaller).  Falls back to raw
    bytes if Pillow is not installed.
    """
    try:
        import io

        from PIL import Image

        img = Image.open(frame_path)
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except ImportError:
        return frame_path.read_bytes()


# A persistent background thread + event loop for running async pydantic-ai
# calls from synchronous code (e.g. Jupyter notebooks).  Reusing one loop
# avoids the "Event loop is closed" error that happens when AsyncOpenAI's
# connection pool outlives a throwaway thread's loop.
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    """Return a long-lived event loop running in a background daemon thread."""
    global _bg_loop, _bg_thread
    if _bg_loop is not None and not _bg_loop.is_closed():
        return _bg_loop

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True, name="sanjaya-io")
    thread.start()
    _bg_loop = loop
    _bg_thread = thread
    return loop


def _compute_cost(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    """Compute cost from token counts using genai_prices."""
    try:
        from genai_prices import calc_price
        from genai_prices.types import Usage

        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        # Try model name as-is, strip provider prefix, strip date suffix
        candidates = [model_name]
        if "/" in model_name:
            base = model_name.split("/", 1)[1]
            candidates.append(base)
            # Strip date suffix like "-20260224"
            import re
            stripped = re.sub(r"-\d{8}$", "", base)
            if stripped != base:
                candidates.append(stripped)

        for ref in candidates:
            for provider_id in ("openrouter", "openai", "anthropic"):
                try:
                    result = calc_price(usage, model_ref=ref, provider_id=provider_id)
                    return float(result.total_price)
                except LookupError:
                    continue
    except ImportError:
        pass
    return None


def _has_running_loop() -> bool:
    """Check if we're inside a running event loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _moondream_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute Moondream cost at $0.30/1M input, $2.50/1M output."""
    return (input_tokens * 0.30 + output_tokens * 2.50) / 1_000_000


class LLMClient:
    """Unified LLM client with text, vision, and batched support."""

    def __init__(
        self,
        model: ModelSpec,
        vision_model: ModelSpec | None = None,
        fallback_model: ModelSpec | None = None,
        name: str = "llm",
    ):
        self.model = model
        self.vision_model = vision_model or model
        self.fallback_model = fallback_model
        self.name = name

        self.last_usage: UsageSnapshot | None = None
        self.last_call_metadata: CallMetadata | None = None

        # Moondream direct client (bypasses pydantic-ai for vision)
        self._moondream: Any = None
        from .moondream import MoondreamVisionClient, is_moondream_spec
        if is_moondream_spec(self.vision_model):
            if isinstance(self.vision_model, MoondreamVisionClient):
                self._moondream = self.vision_model
            else:
                # Parse "moondream:model-name" spec
                model_id = self.vision_model.split(":", 1)[1] if ":" in str(self.vision_model) else "moondream3-preview"
                try:
                    self._moondream = MoondreamVisionClient(model=model_id)
                except Exception:
                    pass  # Fall back to pydantic-ai path

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
        # Moondream bypass: use direct SDK instead of pydantic-ai
        if self._moondream is not None:
            resolved_frames = self._resolve_frame_paths(frame_paths, clip_paths)
            start = time.time()
            response = self._moondream.query_frames(prompt, resolved_frames)
            self.last_usage = UsageSnapshot(
                input_tokens=self._moondream.total_input_tokens,
                output_tokens=self._moondream.total_output_tokens,
                total_tokens=self._moondream.total_input_tokens + self._moondream.total_output_tokens,
            )
            self.last_call_metadata = CallMetadata(
                requested_model=self._moondream.model_name,
                model_used=self._moondream.model_name,
                provider="moondream",
                duration_seconds=round(time.time() - start, 3),
                cost_usd=_moondream_cost(self._moondream.total_input_tokens, self._moondream.total_output_tokens),
            )
            return response

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
        if self._moondream is not None:
            return [
                self.vision_completion(
                    prompt=q["prompt"],
                    frame_paths=q.get("frame_paths"),
                    clip_paths=q.get("clip_paths"),
                )
                for q in queries
            ]

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

    def _resolve_frame_paths(
        self,
        frame_paths: list[str] | None,
        clip_paths: list[str] | None,
    ) -> list[str]:
        """Resolve frame/clip paths into a flat list of existing frame paths."""
        valid_frames = [p for p in (frame_paths or []) if Path(p).exists()]

        if not valid_frames and clip_paths:
            valid_clips = [p for p in clip_paths if Path(p).exists()]
            if valid_clips:
                try:
                    import tempfile

                    from ..tools.video.media import sample_frames as _sample
                    from ..tools.video.media import video_duration_seconds
                    clip = valid_clips[0]
                    dur = video_duration_seconds(clip)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        sampled = _sample(clip, start_s=0, end_s=dur, output_dir=tmpdir, max_frames=8)
                        import shutil
                        persist_dir = Path(clip).parent / "_auto_frames"
                        persist_dir.mkdir(exist_ok=True)
                        for p in sampled:
                            dst = persist_dir / Path(p).name
                            shutil.copy2(p, dst)
                            valid_frames.append(str(dst))
                except Exception:
                    pass

        return valid_frames

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
        """Build multimodal user content list for vision models.

        Only JPEG frames are sent — video clips are never sent as raw
        binary because most providers (OpenAI Chat Completions, OpenRouter)
        reject ``video/mp4`` payloads.  If clips are provided without
        frames, we auto-sample frames from the first clip via ffmpeg.
        """
        user_content: list[Any] = [prompt]

        valid_frames = [Path(p) for p in (frame_paths or []) if Path(p).exists()]
        valid_clips = [Path(p) for p in (clip_paths or []) if Path(p).exists()]

        # If we have clips but no frames, auto-sample frames from the first clip.
        if not valid_frames and valid_clips:
            try:
                import tempfile

                from ..tools.video.media import sample_frames as _sample
                from ..tools.video.media import video_duration_seconds
                clip = str(valid_clips[0])
                dur = video_duration_seconds(clip)
                with tempfile.TemporaryDirectory() as tmpdir:
                    sampled = _sample(clip, start_s=0, end_s=dur, output_dir=tmpdir, max_frames=8)
                    # Copy to a persistent location (tmpdir is cleaned up)
                    import shutil
                    persist_dir = valid_clips[0].parent / "_auto_frames"
                    persist_dir.mkdir(exist_ok=True)
                    persisted = []
                    for p in sampled:
                        dst = persist_dir / Path(p).name
                        shutil.copy2(p, dst)
                        persisted.append(dst)
                    valid_frames = persisted
            except Exception:
                pass

        for frame_path in valid_frames[:8]:
            data = _compress_frame(frame_path, max_dim=768, quality=60)
            user_content.append(
                BinaryContent(data=data, media_type="image/jpeg")
            )

        if not valid_frames:
            user_content.append("No visual attachments were available.")

        return user_content

    def _run_agent(self, model: ModelSpec, payload: Any) -> tuple[str, Any]:
        model_label = model if isinstance(model, str) else getattr(model, "model_name", "unknown")
        agent = Agent(model=model, output_type=str, retries=1, defer_model_check=True, name=f"sanjaya:{self.name}:{model_label}")

        if _has_running_loop():
            # Inside Jupyter: schedule the coroutine on a persistent background
            # loop so the AsyncOpenAI client's connection pool stays alive.
            # Propagate OTel context so spans nest correctly under the caller.
            from opentelemetry import context as otel_context

            parent_ctx = otel_context.get_current()

            async def _run_with_context():
                token = otel_context.attach(parent_ctx)
                try:
                    return await agent.run(payload)
                finally:
                    otel_context.detach(token)

            loop = _get_bg_loop()
            future = asyncio.run_coroutine_threadsafe(_run_with_context(), loop)
            result = future.result()
        else:
            result = agent.run_sync(payload)

        response = result.output if hasattr(result, "output") else str(result)
        return str(response), result

    async def _run_agent_async(self, model: ModelSpec, payload: Any) -> tuple[str, Any]:
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
                pass

        if isinstance(response_cost, (int, float)):
            return float(response_cost)

        # Fallback: compute cost from token counts using genai_prices
        if self.last_usage and (self.last_usage.input_tokens or self.last_usage.output_tokens):
            model_name = getattr(response, "model_name", None) if response else None
            if model_name:
                cost = _compute_cost(model_name, self.last_usage.input_tokens, self.last_usage.output_tokens)
                if cost is not None:
                    return cost

        return None

    def _capture_metadata(self, model: ModelSpec, result: Any, start: float, fallback_used: bool = False) -> CallMetadata:
        response = getattr(result, "response", None)
        model_name = getattr(response, "model_name", None) if response else None
        provider_name = getattr(response, "provider_name", None) if response else None
        provider_response_id = getattr(response, "provider_response_id", None) if response else None
        cost_usd = self._extract_cost_usd(result)

        if isinstance(model, str):
            provider = model.split(":", 1)[0] if ":" in model else "unknown"
        else:
            provider = type(model).__name__
        model_str = model if isinstance(model, str) else getattr(model, "model_name", type(model).__name__)
        meta = CallMetadata(
            requested_model=model_str,
            model_used=model_name or model_str,
            provider=provider_name or provider,
            provider_response_id=provider_response_id,
            fallback_used=fallback_used,
            duration_seconds=round(time.time() - start, 3),
            cost_usd=round(cost_usd, 8) if cost_usd is not None else None,
        )
        self.last_call_metadata = meta
        return meta

    def _call(self, model: ModelSpec, payload: Any, timeout: int = 300) -> str:
        """Core call with fallback."""
        _ = timeout  # Reserved for future provider-specific timeout support
        start = time.time()
        self.last_usage = None
        self.last_call_metadata = None

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

        if _has_running_loop():
            from opentelemetry import context as otel_context
            parent_ctx = otel_context.get_current()

            async def _gather():
                token = otel_context.attach(parent_ctx)
                try:
                    tasks = [self._run_agent_async(c["model"], c["payload"]) for c in calls]
                    return await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    otel_context.detach(token)

            loop = _get_bg_loop()
            future = asyncio.run_coroutine_threadsafe(_gather(), loop)
            raw_results = future.result()
        else:
            async def _gather():
                tasks = [self._run_agent_async(c["model"], c["payload"]) for c in calls]
                return await asyncio.gather(*tasks, return_exceptions=True)

            raw_results = asyncio.run(_gather())

        responses: list[str] = []
        for r in raw_results:
            if isinstance(r, Exception):
                responses.append(f"Error: {r}")
            else:
                response_text, _ = r
                responses.append(response_text)
        return responses
