"""Prompt helpers for RLM orchestration."""

DEFAULT_QUERY = "Answer the user query from the provided context."


def build_system_prompt() -> list[dict[str, str]]:
    """Initial orchestrator instructions."""
    return [
        {
            "role": "system",
            "content": (
                "You are an orchestrator that can reason and execute Python code in a REPL. "
                "If you need tools, return Python code inside fenced code blocks. "
                "Use get_context() to inspect context, llm_query(prompt) for sub-LLM calls, "
                "and done(value) when final answer is known."
            ),
        }
    ]


def next_action_prompt(query: str, iteration: int, final_answer: bool = False) -> dict[str, str]:
    """Prompt the orchestrator for the next step."""
    if final_answer:
        content = (
            "Max iterations reached. Provide only the final answer in plain text using your best estimate."
        )
    else:
        content = (
            f"Iteration {iteration + 1}. User query: {query}\n"
            "Return Python code blocks if you need to compute/search the context. "
            "Call done(value) once you find the final answer."
        )

    return {"role": "user", "content": content}
