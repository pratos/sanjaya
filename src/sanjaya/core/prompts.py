"""System prompt builder for the agent."""

from __future__ import annotations

from typing import Any

from ..tools.registry import ToolRegistry

_CORE_INSTRUCTIONS = """\
You are an RLM (Recursive Language Model) agent that solves problems by writing Python code in a sandboxed REPL.

## How it works
1. You receive a question and optional context.
2. You write Python code in fenced code blocks to investigate, compute, and reason.
3. The code executes in a sandbox (pydantic-monty). You see stdout, stderr, and return values.
4. You iterate: read results, write more code, until you have the answer.
5. Call `done(value)` with your final answer to end the loop.

## Built-in functions
- `get_context()` — returns the context data provided to the agent
- `llm_query(prompt: str) -> str` — query a sub-LLM for analysis/summarization
- `llm_query_batched(prompts: list[str]) -> list[str]` — concurrent sub-LLM queries (faster)
- `done(value)` — signal the final answer and end the loop
- `get_state() -> dict` — inspect agent state, toolkit states, and accumulated artifacts

## Sandbox constraints
Available: list, dict, set, tuple, str, int, float, bool, None, math, re, json, collections, itertools, functools, string operations, f-strings, list comprehensions, slicing, unpacking.

NOT available: os, sys, subprocess, pathlib, importlib, open(), file I/O, network access, eval(), exec(). Use the provided tools for all external operations.

## Strategy
- Start by understanding the context: call get_context() or inspect provided data.
- Break complex problems into steps. Use intermediate variables.
- Use llm_query() when you need the LLM to analyze, summarize, or reason about gathered data.
- Use llm_query_batched() when you have multiple independent analyses — it's much faster.
- Call done(value) as soon as you have confidence in your answer.
"""

_NEXT_ACTION_TEMPLATE = """\
Iteration {iteration}. User query: {query}
Return Python code blocks if you need to compute/search the context. \
Call done(value) once you find the final answer."""

_FORCE_FINAL = """\
Max iterations reached. Provide only the final answer in plain text using your best estimate."""


def build_system_prompt(
    *,
    registry: ToolRegistry,
    context_metadata: dict[str, Any] | None = None,
    toolkit_sections: list[str] | None = None,
) -> str:
    """Build the full system prompt.

    Structure:
    1. Core RLM instructions
    2. Auto-generated tool docs (from registry)
    3. Toolkit-specific strategy sections
    4. Context metadata
    """
    parts = [_CORE_INSTRUCTIONS]

    # Tool docs
    tool_docs = registry.generate_tool_docs()
    if tool_docs:
        parts.append(f"\n## Additional tools\n{tool_docs}")

    # Toolkit strategy sections
    if toolkit_sections:
        for section in toolkit_sections:
            if section:
                parts.append(f"\n{section}")

    # Context metadata
    if context_metadata:
        meta_lines = ["## Context metadata"]
        for key, value in context_metadata.items():
            meta_lines.append(f"- {key}: {value}")
        parts.append("\n".join(meta_lines))

    return "\n".join(parts)


def next_action_prompt(query: str, iteration: int, final_answer: bool = False) -> dict[str, str]:
    """Prompt the orchestrator for the next step."""
    if final_answer:
        return {"role": "user", "content": _FORCE_FINAL}
    return {
        "role": "user",
        "content": _NEXT_ACTION_TEMPLATE.format(iteration=iteration + 1, query=query),
    }
