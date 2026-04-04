"""Core utility functions used by RLM_REPL."""

from __future__ import annotations

import json
import re
from typing import Any

from ..repl import ExecutionResult, MontyREPL


def convert_context_for_repl(context: Any) -> tuple[Any | None, str]:
    """Normalize input context for the REPL environment."""
    if isinstance(context, str):
        return None, context

    if isinstance(context, (list, dict)):
        try:
            return context, json.dumps(context, ensure_ascii=False)
        except TypeError:
            return None, str(context)

    return None, str(context)


def find_code_blocks(response: str) -> list[str]:
    """Extract fenced code blocks from model output."""
    pattern = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    return [block.strip() for block in pattern.findall(response)]


def _format_execution_feedback(result: ExecutionResult, index: int, total: int) -> str:
    parts = [f"Code block {index}/{total} executed."]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if result.result is not None:
        parts.append(f"RESULT: {result.result!r}")
    if result.final_answer is not None:
        parts.append(f"FINAL_ANSWER: {result.final_answer}")
    return "\n\n".join(parts)


def process_code_execution(
    response: str,
    messages: list[dict[str, str]],
    repl_env: MontyREPL,
    repl_env_logger: Any,
    logger: Any,
    *,
    run_id: str | None = None,
    iteration: int | None = None,
) -> tuple[list[dict[str, str]], ExecutionResult | None]:
    """Execute all code blocks in a model response and append feedback to message history."""
    code_blocks = find_code_blocks(response)
    messages.append({"role": "assistant", "content": response})

    last_result: ExecutionResult | None = None
    total = len(code_blocks)

    for idx, code in enumerate(code_blocks, start=1):
        repl_env_logger.log_execution_start(idx, total, code)
        last_result = repl_env.code_execution(
            code,
            run_id=run_id,
            iteration=iteration,
            code_block_index=idx,
            code_block_total=total,
        )
        repl_env_logger.log_execution_end(idx, total, last_result.execution_time)
        logger.log_execution_result(idx, total, last_result.stdout, last_result.stderr, last_result.result)

        feedback = _format_execution_feedback(last_result, idx, total)
        messages.append({"role": "user", "content": feedback})

    return messages, last_result


def check_for_final_answer(
    response: str,
    repl_env: MontyREPL | None,
    logger: Any,
    execution_result: ExecutionResult | None = None,
) -> str | None:
    """Determine whether a final answer is available."""
    _ = repl_env, logger

    if execution_result is not None and execution_result.final_answer is not None:
        return str(execution_result.final_answer)

    final_match = re.search(r"FINAL\((.*?)\)", response, re.DOTALL)
    if final_match:
        return final_match.group(1).strip().strip("'\"")

    done_match = re.search(r"final answer\s*[:=]\s*(.+)$", response, re.IGNORECASE | re.MULTILINE)
    if done_match:
        return done_match.group(1).strip()

    return None
