"""Code block extraction, execution feedback formatting, and final answer detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionResult:
    """Result of code execution in the REPL."""

    stdout: str
    stderr: str
    result: Any
    execution_time: float
    final_answer: Any | None = None
    llm_queries: list[tuple[str, str]] = field(default_factory=list)


def extract_code_blocks(response: str) -> list[str]:
    """Extract ```python or ```repl code blocks from LLM response."""
    pattern = re.compile(r"```(?:python|repl)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in pattern.findall(response)]


_MAX_STDOUT_CHARS = 2_000
_MAX_STDERR_CHARS = 1_000


def _format_llm_queries(
    llm_queries: list[tuple[str, str]],
    *,
    max_prompt_chars: int = 500,
    max_response_chars: int = 1_500,
) -> str:
    """Render sub-LLM calls into compact feedback text for the orchestrator."""
    if not llm_queries:
        return ""

    parts = [f"SUB_LLM_CALLS: {len(llm_queries)}"]
    for idx, (prompt, response) in enumerate(llm_queries, start=1):
        prompt_text = prompt.strip()
        response_text = response.strip()

        if len(prompt_text) > max_prompt_chars:
            prompt_text = f"{prompt_text[:max_prompt_chars]}\n...[prompt truncated]"
        if len(response_text) > max_response_chars:
            response_text = f"{response_text[:max_response_chars]}\n...[response truncated]"

        parts.append(
            (
                f"SUB_LLM_CALL_{idx}_PROMPT:\n{prompt_text}\n\n"
                f"SUB_LLM_CALL_{idx}_RESPONSE:\n{response_text}"
            )
        )

    return "\n\n".join(parts)


def format_execution_feedback(result: ExecutionResult, block_index: int, block_total: int) -> str:
    """Format REPL output as a user message for the next iteration."""
    parts = [f"Code block {block_index}/{block_total} executed."]
    if result.stdout:
        stdout = result.stdout
        if len(stdout) > _MAX_STDOUT_CHARS:
            stdout = stdout[:_MAX_STDOUT_CHARS] + "\n...[stdout truncated]"
        parts.append(f"STDOUT:\n{stdout}")
    if result.stderr:
        stderr = result.stderr
        if len(stderr) > _MAX_STDERR_CHARS:
            stderr = stderr[:_MAX_STDERR_CHARS] + "\n...[stderr truncated]"
        parts.append(f"STDERR:\n{stderr}")
    if result.result is not None:
        # Format dicts/lists as readable JSON for the orchestrator
        if isinstance(result.result, (dict, list)):
            try:
                formatted = json.dumps(result.result, indent=2, default=str)
                if len(formatted) > 2000:
                    formatted = formatted[:2000] + "\n...[truncated]"
                parts.append(f"RESULT:\n{formatted}")
            except (TypeError, ValueError):
                parts.append(f"RESULT: {result.result!r}")
        else:
            parts.append(f"RESULT: {result.result!r}")

    llm_feedback = _format_llm_queries(result.llm_queries)
    if llm_feedback:
        parts.append(llm_feedback)

    if result.final_answer is not None:
        parts.append(f"FINAL_ANSWER: {result.final_answer}")
    return "\n\n".join(parts)


def extract_final_answer(result: ExecutionResult | None, response: str) -> Any | None:
    """Check for done() signal, FINAL(...), or 'final answer:' in response."""
    # Check done() from REPL execution
    if result is not None and result.final_answer is not None:
        return result.final_answer

    # Check FINAL(...) directive
    final_match = re.search(r"FINAL\((.*?)\)", response, re.DOTALL)
    if final_match:
        return final_match.group(1).strip().strip("'\"")

    # Check "final answer:" pattern
    done_match = re.search(r"final answer\s*[:=]\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    if done_match:
        return done_match.group(1).strip()

    return None
