"""LLM client wrapper built on pydantic-ai, using OpenRouter as the primary provider."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent

from ..settings import get_settings


@dataclass
class UsageSnapshot:
    """Token usage snapshot for the last model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class LLMClient:
    """Thin wrapper around pydantic-ai Agent for OpenRouter-first calls.

    Optionally accepts a fallback model tried when the primary model errors.
    """

    def __init__(
        self,
        model: str = "openrouter:openai/gpt-4.1-mini",
        api_key: str | None = None,
        fallback_model: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.fallback_model = fallback_model
        self.last_usage: UsageSnapshot | None = None
        self.last_call_metadata: dict[str, Any] | None = None

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

    def _ensure_api_key(self) -> None:
        """Populate provider API key from explicit arg or settings into env vars."""
        settings = get_settings()
        provider = self.model.split(":", 1)[0] if ":" in self.model else "unknown"

        if provider == "openrouter":
            key = self.api_key or settings.openrouter_api_key
            if key:
                os.environ["OPENROUTER_API_KEY"] = key
        elif provider.startswith("openai"):
            key = self.api_key or settings.openai_api_key
            if key:
                os.environ["OPENAI_API_KEY"] = key
        elif provider.startswith("anthropic"):
            key = self.api_key or settings.anthropic_api_key
            if key:
                os.environ["ANTHROPIC_API_KEY"] = key

    def _run_agent(self, model: str, prompt: str) -> tuple[str, Any]:
        agent = Agent(model=model, output_type=str, retries=1, defer_model_check=True)
        result = agent.run_sync(prompt)
        response = result.output if hasattr(result, "output") else str(result)
        return str(response), result

    def _resolve_usage(self, usage_obj: Any) -> Any:
        """Resolve usage attribute that may be a value or a callable."""
        if callable(usage_obj):
            try:
                return usage_obj()
            except Exception:
                return None
        return usage_obj

    def _capture_usage(self, result: Any) -> None:
        usage = self._resolve_usage(getattr(result, "usage", None))
        if usage is None:
            response = getattr(result, "response", None)
            usage = self._resolve_usage(getattr(response, "usage", None))

        if usage is None:
            self.last_usage = UsageSnapshot()
            return

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (input_tokens + output_tokens))

        self.last_usage = UsageSnapshot(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _extract_cost_usd(self, result: Any) -> float | None:
        """Best-effort extraction of model cost in USD from result metadata."""
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

    def _capture_result_metadata(self, result: Any) -> dict[str, Any]:
        response = getattr(result, "response", None)
        metadata: dict[str, Any] = {}

        if response is not None:
            model_name = getattr(response, "model_name", None)
            provider_name = getattr(response, "provider_name", None)
            provider_response_id = getattr(response, "provider_response_id", None)
            if model_name:
                metadata["model_used"] = model_name
            if provider_name:
                metadata["provider"] = provider_name
            if provider_response_id:
                metadata["provider_response_id"] = provider_response_id

        cost_usd = self._extract_cost_usd(result)
        if cost_usd is not None:
            metadata["cost_usd"] = round(cost_usd, 8)

        return metadata

    def completion(self, prompt_or_messages: Any, timeout: int | None = None) -> str:
        _ = timeout  # Reserved for future provider-specific timeout support
        prompt = self._as_prompt(prompt_or_messages)

        start = time.time()
        self.last_usage = None
        self._ensure_api_key()

        provider = self.model.split(":", 1)[0] if ":" in self.model else "openrouter"
        self.last_call_metadata = {
            "requested_model": self.model,
            "provider": provider,
            "fallback_used": False,
        }

        # Primary call
        try:
            response, result = self._run_agent(self.model, prompt)
            self._capture_usage(result)
            result_metadata = self._capture_result_metadata(result)
            self.last_call_metadata.update(
                {
                    "model_used": getattr(getattr(result, "response", None), "model_name", self.model),
                    "duration_seconds": round(time.time() - start, 3),
                    **result_metadata,
                }
            )
            return response
        except Exception as primary_err:
            if not self.fallback_model:
                raise

            self.last_call_metadata["primary_error"] = str(primary_err)

        # Fallback call (single level, same provider)
        response, result = self._run_agent(self.fallback_model, prompt)
        self._capture_usage(result)
        result_metadata = self._capture_result_metadata(result)
        self.last_call_metadata.update(
            {
                "model_used": self.fallback_model,
                "fallback_used": True,
                "fallback_reason": "primary_error",
                "duration_seconds": round(time.time() - start, 3),
                **result_metadata,
            }
        )
        return response
