"""LLM client helpers for text + vision-style completions in VideoRLM."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from sanjaya.utils.llm import LLMClient


class VideoLLMClient:
    """Wrapper that keeps text completion and adds multimodal-safe calls."""

    def __init__(
        self,
        model: str,
        vision_model: str | None = None,
        api_key: str | None = None,
        fallback_model: str | None = None,
    ):
        self.text_client = LLMClient(model=model, api_key=api_key)
        self.vision_client = LLMClient(
            model=vision_model or model,
            api_key=api_key,
            fallback_model=fallback_model,
        )

    @property
    def last_usage(self):
        return self.text_client.last_usage

    @property
    def last_call_metadata(self):
        return self.text_client.last_call_metadata

    @property
    def last_vision_usage(self):
        return self.vision_client.last_usage

    @property
    def last_vision_call_metadata(self):
        return self.vision_client.last_call_metadata

    def completion(self, prompt_or_messages: Any, timeout: int | None = None) -> str:
        """Text completion passthrough for orchestrator calls."""
        return self.text_client.completion(prompt_or_messages, timeout=timeout)

    def vision_completion(
        self,
        *,
        question: str,
        frame_paths: list[str] | None = None,
        clip_paths: list[str] | None = None,
        extra_context: str | None = None,
        timeout: int | None = None,
    ) -> str:
        """Best-effort multimodal completion that safely handles local artifacts."""
        frame_paths = frame_paths or []
        clip_paths = clip_paths or []

        attachment_lines: list[str] = []
        for path in frame_paths:
            attachment_lines.append(self._encode_file_line(path, media_type="image/jpeg"))

        for path in clip_paths:
            attachment_lines.append(self._encode_file_line(path, media_type="video/mp4", max_bytes=128_000))

        payload_parts = [
            "You are analyzing evidence from a long video QA workflow.",
            f"Question: {question}",
        ]

        if extra_context:
            payload_parts.append(f"Context: {extra_context}")

        if attachment_lines:
            payload_parts.append("Attachments (data URLs; may be truncated):")
            payload_parts.extend(attachment_lines)
        else:
            payload_parts.append("Attachments: none")

        payload_parts.append(
            "Return a concise textual evidence summary with likely actions/scenes and confidence caveats."
        )
        prompt = "\n".join(payload_parts)

        try:
            return self.vision_client.completion(prompt, timeout=timeout)
        except Exception as exc:
            attachment_preview = ", ".join(
                [Path(p).name for p in frame_paths[:3]] + [Path(p).name for p in clip_paths[:2]]
            )
            return (
                "Vision model unavailable; fallback summary: "
                f"question='{question}', attachments=[{attachment_preview}], error='{exc}'"
            )

    @staticmethod
    def _encode_file_line(path: str, *, media_type: str, max_bytes: int = 64_000) -> str:
        file_path = Path(path)
        if not file_path.exists():
            return f"- missing: {path}"

        data = file_path.read_bytes()[:max_bytes]
        encoded = base64.b64encode(data).decode("ascii")
        truncated = " (truncated)" if file_path.stat().st_size > len(data) else ""
        return f"- {file_path.name}{truncated}: data:{media_type};base64,{encoded}"
