"""Prompt templates for VideoRLM orchestration."""

VIDEO_DEFAULT_QUERY = "Answer the user question grounded in video evidence."


def build_video_system_prompt() -> list[dict[str, str]]:
    """System prompt for the video-capable orchestrator."""
    tool_contract = (
        "You are a recursive orchestrator with Python REPL tool use. "
        "Use fenced Python code blocks when you need tools. "
        "Available core tools: get_context(), llm_query(prompt), done(value). "
        "Video tools: list_candidate_windows(...), extract_clip(...), sample_frames(...), "
        "get_clip_manifest(...), vision_query(...), get_trace_log(). "
        "Policy: 1) start with list_candidate_windows, 2) inspect top windows via extract_clip/sample_frames, "
        "3) call vision_query when textual cues are insufficient or visual confirmation is needed, "
        "4) cite timestamps and artifacts before done(value)."
    )
    return [{"role": "system", "content": tool_contract}]


def next_video_action_prompt(question: str, iteration: int, final_answer: bool = False) -> dict[str, str]:
    """Iteration prompt for the video orchestrator."""
    if final_answer:
        content = "Max iterations reached. Return your best grounded final answer in plain text."
    else:
        content = (
            f"Iteration {iteration + 1}. Question: {question}\n"
            "Use code blocks to call tools when needed. "
            "Follow tool chain: list_candidate_windows -> extract_clip -> sample_frames -> optional vision_query. "
            "Use get_trace_log() if you need prior tool events. Call done(value) when evidence is sufficient."
        )
    return {"role": "user", "content": content}
