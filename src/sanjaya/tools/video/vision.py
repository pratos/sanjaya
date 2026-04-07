"""Vision query tools — query multimodal LLMs about video frames/clips."""

from __future__ import annotations

from typing import Any


def make_vision_query_fn(
    *,
    llm_client: Any,  # LLMClient
    get_clips: Any,  # Callable that returns clips dict
    get_question: Any,  # Callable that returns question string
    get_tracer: Any = None,  # Callable that returns Tracer or None
) -> Any:
    """Create a vision_query closure bound to toolkit state."""

    def vision_query(
        *,
        prompt: str | None = None,
        clip_id: str | None = None,
        frame_paths: list[str] | None = None,
    ) -> str:
        """Query a vision model about video frames or a clip.

        Sends frames (preferred) or a clip to a multimodal LLM.
        If no prompt given, uses the original question.

        Args:
            prompt: What to ask about the visual content.
            clip_id: ID from extract_clip() (will use its frames).
            frame_paths: Direct frame paths (alternative to clip_id).
        """
        collected_frames = list(frame_paths or [])
        collected_clips: list[str] = []
        clips = get_clips()

        if clip_id:
            artifact = clips.get(clip_id)
            if artifact is None:
                raise ValueError(f"Unknown clip_id: {clip_id}")
            collected_clips.append(artifact["clip_path"])
            collected_frames.extend(artifact.get("frame_paths", []))

        if not collected_frames and not collected_clips and clips:
            latest = next(reversed(clips.values()))
            collected_clips.append(latest["clip_path"])
            collected_frames.extend(latest.get("frame_paths", []))

        effective_prompt = prompt or get_question()

        tracer = get_tracer() if get_tracer else None
        n_frames = len(collected_frames)
        vision_model = getattr(llm_client, "vision_model", "unknown")
        model_label = vision_model if isinstance(vision_model, str) else getattr(vision_model, "model_name", "unknown")

        if tracer:
            with tracer._span(
                "sanjaya.sub_llm_call.vision",
                model=model_label,
                prompt_chars=len(effective_prompt),
                n_frames=n_frames,
                clip_id=clip_id or "",
            ) as ctx:
                result = llm_client.vision_completion(
                    prompt=effective_prompt,
                    frame_paths=collected_frames if collected_frames else None,
                    clip_paths=collected_clips if collected_clips else None,
                )
                ctx.record_response(result)
                usage = llm_client.last_usage
                if usage:
                    ctx.record_usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                metadata = llm_client.last_call_metadata
                if metadata:
                    ctx.record(cost_usd=metadata.cost_usd, duration_seconds=metadata.duration_seconds)
                return result
        else:
            return llm_client.vision_completion(
                prompt=effective_prompt,
                frame_paths=collected_frames if collected_frames else None,
                clip_paths=collected_clips if collected_clips else None,
            )

    return vision_query


def make_vision_query_batched_fn(
    *,
    llm_client: Any,  # LLMClient
    get_clips: Any,
    get_question: Any,
    get_tracer: Any = None,  # Callable that returns Tracer or None
) -> Any:
    """Create a vision_query_batched closure bound to toolkit state."""

    def vision_query_batched(queries: list[dict]) -> list[str]:
        """Run multiple vision queries concurrently.

        Args:
            queries: List of dicts with keys matching vision_query params:
                [{"prompt": "...", "clip_id": "..."}, ...]
        """
        clips = get_clips()
        batch: list[dict] = []

        for q in queries:
            frame_paths = list(q.get("frame_paths") or [])
            clip_paths: list[str] = []

            clip_id = q.get("clip_id")
            if clip_id:
                artifact = clips.get(clip_id)
                if artifact:
                    clip_paths.append(artifact["clip_path"])
                    frame_paths.extend(artifact.get("frame_paths", []))

            batch.append({
                "prompt": q.get("prompt") or get_question(),
                "frame_paths": frame_paths if frame_paths else None,
                "clip_paths": clip_paths if clip_paths else None,
            })

        tracer = get_tracer() if get_tracer else None
        vision_model = getattr(llm_client, "vision_model", "unknown")
        model_label = vision_model if isinstance(vision_model, str) else getattr(vision_model, "model_name", "unknown")

        if tracer:
            with tracer._span(
                "sanjaya.sub_llm_call.vision",
                model=model_label,
                n_queries=len(batch),
                batched=True,
            ) as ctx:
                results = llm_client.vision_completion_batched(batch)
                ctx.record(n_results=len(results))
                usage = llm_client.last_usage
                if usage:
                    ctx.record_usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                return results
        else:
            return llm_client.vision_completion_batched(batch)

    return vision_query_batched
