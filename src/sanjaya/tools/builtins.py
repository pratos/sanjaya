"""Built-in tools available in every agent REPL session.

These are registered automatically by the Agent — not by user code.
They wrap closures that capture agent/REPL state at runtime.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import Tool, ToolParam


def make_context_tool(get_context_fn: Callable[[], Any]) -> Tool:
    """Create the `context` tool that returns the current context."""
    return Tool(
        name="get_context",
        description="Return the context data provided to agent.ask().",
        fn=get_context_fn,
        parameters={},
        return_type="Any",
    )


def make_llm_query_tool(llm_query_fn: Callable[[str], str]) -> Tool:
    """Create the `llm_query` tool for sub-LLM calls."""
    return Tool(
        name="llm_query",
        description=(
            "Query a sub-LLM with a text prompt. Use this when you need "
            "the LLM to analyze, summarize, or reason about data you've gathered."
        ),
        fn=llm_query_fn,
        parameters={
            "prompt": ToolParam(
                name="prompt",
                type_hint="str",
                description="The text prompt to send to the sub-LLM.",
            ),
        },
        return_type="str",
    )


def make_llm_query_batched_tool(llm_query_batched_fn: Callable[[list[str]], list[str]]) -> Tool:
    """Create the `llm_query_batched` tool for concurrent sub-LLM calls."""
    return Tool(
        name="llm_query_batched",
        description=(
            "Run multiple LLM queries concurrently. Much faster than "
            "sequential llm_query() calls for independent analyses."
        ),
        fn=llm_query_batched_fn,
        parameters={
            "prompts": ToolParam(
                name="prompts",
                type_hint="list[str]",
                description="List of text prompts to send concurrently.",
            ),
        },
        return_type="list[str]",
    )


def make_done_tool(done_fn: Callable[[Any], Any]) -> Tool:
    """Create the `done` tool that signals the final answer."""
    return Tool(
        name="done",
        description=(
            "Signal that you have the final answer. Call this with your answer "
            "to end the loop. The value you pass becomes the agent's response."
        ),
        fn=done_fn,
        parameters={
            "value": ToolParam(
                name="value",
                type_hint="Any",
                description="The final answer value.",
            ),
        },
        return_type="Any",
    )


def make_get_state_tool(get_state_fn: Callable[[], dict[str, Any]]) -> Tool:
    """Create the `get_state` tool for inspecting accumulated state."""
    return Tool(
        name="get_state",
        description=(
            "Inspect the current agent state: registered tools, toolkit states, "
            "iteration count, budget usage, and accumulated artifacts."
        ),
        fn=get_state_fn,
        parameters={},
        return_type="dict",
    )
