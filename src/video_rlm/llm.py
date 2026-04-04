"""LLM client helpers for text + vision-style completions in VideoRLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic_ai.messages import BinaryContent

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
        """Best-effort multimodal completion for local frame/clip artifacts."""
        frame_paths = frame_paths or []
        clip_paths = clip_paths or []

        prompt_parts = [
            "You are analyzing evidence from a long video QA workflow.",
            f"Question: {question}",
            "Summarize what is visually present and include confidence caveats.",
        ]
        if extra_context:
            prompt_parts.append(f"Context: {extra_context}")

        user_content: list[Any] = ["\n".join(prompt_parts)]

        valid_frames = [Path(path) for path in frame_paths if Path(path).exists()]
        valid_clips = [Path(path) for path in clip_paths if Path(path).exists()]

        for frame_path in valid_frames[:8]:
            user_content.append(BinaryContent(data=frame_path.read_bytes(), media_type="image/jpeg"))

        if not valid_frames:
            for clip_path in valid_clips[:1]:
                user_content.append(BinaryContent(data=clip_path.read_bytes(), media_type="video/mp4"))

        if not valid_frames and not valid_clips:
            user_content.append("No attachments were available.")

        try:
            return self.vision_client.completion_with_user_content(user_content, timeout=timeout)
        except Exception as exc:
            attachment_preview = ", ".join(
                [Path(p).name for p in frame_paths[:3]] + [Path(p).name for p in clip_paths[:2]]
            )
            return (
                "Vision model unavailable; fallback summary: "
                f"question='{question}', attachments=[{attachment_preview}], error='{exc}'"
            )
