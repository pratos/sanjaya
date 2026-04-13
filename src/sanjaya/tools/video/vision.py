"""Vision query tools — query multimodal LLMs about video frames/clips."""

from __future__ import annotations

from typing import Any

_CAPTION_PROMPT = (
    "Describe what you see in this video frame in 1-2 sentences. "
    "Focus on: people, actions, text on screen, UI elements, "
    "diagrams, code, and setting. Be specific about any text you can read."
)


def _record_vision_budget(llm_client: Any, get_budget: Any) -> None:
    """Record vision call cost in the budget tracker if available."""
    budget = get_budget() if get_budget else None
    if budget is None:
        return
    usage = getattr(llm_client, "last_usage", None)
    if usage:
        cost = getattr(llm_client, "last_cost_usd", None) or 0.0
        model = getattr(llm_client, "model", None)
        budget.record(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
            model=str(model) if model else "vision",
        )


def make_vision_query_fn(
    *,
    llm_client: Any,  # LLMClient
    get_clips: Any,  # Callable that returns clips dict
    get_question: Any,  # Callable that returns question string
    get_tracer: Any = None,  # Callable that returns Tracer or None
    get_budget: Any = None,  # Callable that returns BudgetTracker or None
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
                frame_paths=collected_frames[:12],
                clip_paths=collected_clips,
            ) as ctx:
                result = llm_client.vision_completion(
                    prompt=effective_prompt,
                    frame_paths=collected_frames if collected_frames else None,
                    clip_paths=collected_clips if collected_clips else None,
                )
                ctx.record_response(result)
                metadata = llm_client.last_call_metadata
                if metadata:
                    ctx.record(cost_usd=metadata.cost_usd, duration_seconds=metadata.duration_seconds)
                _record_vision_budget(llm_client, get_budget)
                return result
        else:
            result = llm_client.vision_completion(
                prompt=effective_prompt,
                frame_paths=collected_frames if collected_frames else None,
                clip_paths=collected_clips if collected_clips else None,
            )
            _record_vision_budget(llm_client, get_budget)
            return result

    return vision_query


def make_caption_frames_fn(
    *,
    llm_client: Any,  # LLMClient
    captioner: Any = None,  # Dedicated captioner (e.g. MoondreamVisionClient)
    get_clips: Any,  # Callable that returns clips dict
    get_tracer: Any = None,  # Callable that returns Tracer or None
    get_budget: Any = None,  # Callable that returns BudgetTracker or None
) -> Any:
    """Create a caption_frames closure that captions each frame individually."""

    def caption_frames(
        *,
        clip_id: str,
        prompt: str | None = None,
    ) -> list[str]:
        """Caption each frame in a clip individually using the vision model.

        Returns a list of "[timestamp]s caption" strings. Feed these to
        llm_query() for reasoning — cheaper and more thorough than sending
        all frames in a single vision_query() call.

        Args:
            clip_id: ID from extract_clip().
            prompt: Custom captioning prompt (default: focused scene description).
        """
        clips = get_clips()
        artifact = clips.get(clip_id)
        if artifact is None:
            raise ValueError(f"Unknown clip_id: {clip_id}")

        frame_paths = artifact.get("frame_paths", [])
        if not frame_paths:
            raise ValueError(
                f"No frames sampled for {clip_id}. Call sample_frames() first."
            )

        clip_start = artifact["start_s"]
        clip_end = artifact["end_s"]
        clip_duration = clip_end - clip_start
        n_frames = len(frame_paths)

        effective_prompt = prompt or _CAPTION_PROMPT

        tracer = get_tracer() if get_tracer else None
        # Prefer dedicated captioner (set via Agent caption_model=),
        # fall back to moondream on llm_client, then generic vision.
        moondream = captioner or getattr(llm_client, "_moondream", None)

        if moondream is not None:
            # Fast path: use Moondream /caption endpoint (per-frame, concurrent)
            model_label = getattr(moondream, "model_name", "moondream")

            tokens_before = getattr(moondream, "total_input_tokens", 0)
            out_before = getattr(moondream, "total_output_tokens", 0)

            if tracer:
                with tracer._span(
                    "sanjaya.sub_llm_call.caption_frames",
                    model=model_label,
                    n_frames=n_frames,
                    clip_id=clip_id,
                    backend="moondream_caption",
                ) as ctx:
                    captions = moondream.caption_frames_batch(
                        frame_paths, length="normal",
                    )
                    in_delta = getattr(moondream, "total_input_tokens", 0) - tokens_before
                    out_delta = getattr(moondream, "total_output_tokens", 0) - out_before
                    ctx.record(n_captions=len(captions), sanjaya_input_tokens=in_delta, sanjaya_output_tokens=out_delta)
            else:
                captions = moondream.caption_frames_batch(
                    frame_paths, length="normal",
                )
                in_delta = getattr(moondream, "total_input_tokens", 0) - tokens_before
                out_delta = getattr(moondream, "total_output_tokens", 0) - out_before

            # Record moondream cost directly into the budget
            budget = get_budget() if get_budget else None
            if budget is not None:
                from ...llm.pricing import moondream_cost

                cost = moondream_cost(in_delta, out_delta)
                budget.record(
                    input_tokens=in_delta,
                    output_tokens=out_delta,
                    cost_usd=cost,
                    model=model_label,
                )
        else:
            # Fallback: use generic vision model with captioning prompt
            queries = [
                {
                    "prompt": effective_prompt,
                    "frame_paths": [fp],
                    "clip_paths": None,
                }
                for fp in frame_paths
            ]

            vision_model = getattr(llm_client, "vision_model", "unknown")
            model_label = (
                vision_model
                if isinstance(vision_model, str)
                else getattr(vision_model, "model_name", "unknown")
            )

            if tracer:
                with tracer._span(
                    "sanjaya.sub_llm_call.caption_frames",
                    model=model_label,
                    n_frames=n_frames,
                    clip_id=clip_id,
                    backend="vision_completion",
                ) as ctx:
                    captions = llm_client.vision_completion_batched(queries)
                    ctx.record(n_captions=len(captions))
                    _record_vision_budget(llm_client, get_budget)
            else:
                captions = llm_client.vision_completion_batched(queries)
                _record_vision_budget(llm_client, get_budget)

        # Compute timestamp for each frame (uniform spacing within clip)
        results = []
        for i, caption in enumerate(captions):
            if n_frames == 1:
                ts = clip_start + clip_duration / 2
            else:
                ts = clip_start + (i * clip_duration / (n_frames - 1))
            results.append(f"[{ts:.1f}s] {caption.strip()}")

        return results

    return caption_frames


def make_vision_query_batched_fn(
    *,
    llm_client: Any,  # LLMClient
    get_clips: Any,
    get_question: Any,
    get_tracer: Any = None,  # Callable that returns Tracer or None
    get_budget: Any = None,  # Callable that returns BudgetTracker or None
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
                _record_vision_budget(llm_client, get_budget)
                return results
        else:
            results = llm_client.vision_completion_batched(batch)
            _record_vision_budget(llm_client, get_budget)
            return results

    return vision_query_batched
