"""Prompt templates for VideoRLM orchestration."""

VIDEO_DEFAULT_QUERY = "Answer the user question grounded in video evidence."


def build_video_system_prompt() -> list[dict[str, str]]:
    """System prompt for the video-capable orchestrator."""
    tool_contract = (
        "You are a recursive orchestrator with Python REPL tool use. "
        "Use at most ONE fenced Python block per iteration and prefer deterministic, short code. "
        "Available core tools: get_context(), llm_query(prompt), done(value). "
        "Video tools and exact signatures: "
        "list_candidate_windows(question=None, top_k=8, window_size_s=45.0, stride_s=30.0); "
        "extract_clip(window_id=None, start_s=None, end_s=None, clip_id=None); "
        "sample_frames(clip_id=None, clip_path=None, max_frames=8); "
        "vision_query(prompt=None, clip_id=None, frame_paths=None, clip_paths=None); "
        "get_clip_manifest(); get_trace_log(). "
        "Do NOT invent parameter names. "
        "Canonical flow: windows=list_candidate_windows(top_k=5) -> "
        "clip=extract_clip(window_id=windows[i]['window_id']) -> "
        "sample_frames(clip_id=clip['clip_id'], max_frames=4) -> "
        "vision_query(clip_id=clip['clip_id'], prompt='...') -> done('final answer'). "
        "Include timestamps in the final answer."
    )
    return [{"role": "system", "content": tool_contract}]


def next_video_action_prompt(question: str, iteration: int, final_answer: bool = False) -> dict[str, str]:
    """Iteration prompt for the video orchestrator."""
    if final_answer:
        content = "Max iterations reached. Return your best grounded final answer in plain text."
    else:
        content = (
            f"Iteration {iteration + 1}. Question: {question}\n"
            "Use at most one Python code block. "
            "Use exact tool signatures only. "
            "Follow chain: list_candidate_windows -> extract_clip(window_id=...) -> "
            "sample_frames(clip_id=...) -> optional vision_query(clip_id=...). "
            "Use get_trace_log() if needed. Call done(value) when evidence is sufficient."
        )
    return {"role": "user", "content": content}
