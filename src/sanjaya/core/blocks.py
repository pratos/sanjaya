"""Code block extraction, execution feedback formatting, and final answer detection."""

from __future__ import annotations

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


def format_execution_feedback(result: ExecutionResult, block_index: int, block_total: int) -> str:
    """Format REPL output as a user message for the next iteration."""
    parts = [f"Code block {block_index}/{block_total} executed."]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.result is not None:
        parts.append(f"RESULT: {result.result!r}")
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
