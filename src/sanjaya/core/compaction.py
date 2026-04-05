"""Context compaction — summarize message history when approaching context limit."""

from __future__ import annotations

from typing import Any

_COMPACTION_PROMPT = """\
You are summarizing an in-progress analysis session. The agent has been iterating \
on a problem and needs to continue, but the conversation history is getting long.

Summarize the progress so far in 1-3 paragraphs. Preserve:
- Key intermediate results and findings
- What approaches have been tried (successes and failures)
- What the agent was working on most recently
- Any important variable values or data discovered

Be concise but don't lose critical information the agent needs to continue."""


def compact_history(
    *,
    messages: list[dict[str, str]],
    llm: Any,  # LLMClient — avoid circular import
    system_prompt: str,
    threshold_pct: float = 0.85,
    model_context_limit: int = 200_000,
) -> list[dict[str, str]]:
    """Summarize message history when approaching context limit.

    Asks the LLM to summarize progress, then replaces history with
    [system_prompt, summary, "continue from summary"].

    Returns the original messages if under threshold.
    """
    # Estimate token count (rough: 4 chars per token)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    estimated_tokens = total_chars / 4
    threshold = model_context_limit * threshold_pct

    if estimated_tokens < threshold:
        return messages

    # Build summary request from the conversation so far
    history_text = "\n\n".join(
        f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:2000]}"
        for m in messages[1:]  # Skip system prompt
    )

    summary_prompt = f"{_COMPACTION_PROMPT}\n\n---\n\nConversation history:\n{history_text}"

    try:
        summary = llm.completion(summary_prompt)
    except Exception:
        # If compaction fails, return original — better to overflow than crash
        return messages

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Session summary (compacted from {len(messages)} messages):\n\n{summary}"},
        {"role": "user", "content": "Continue your analysis from where you left off."},
    ]
