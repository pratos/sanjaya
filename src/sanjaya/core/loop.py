"""The RLM iteration loop — the core engine that drives the agent."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from .blocks import ExecutionResult, extract_code_blocks, extract_final_answer, format_execution_feedback
from .budget import BudgetTracker
from .compaction import compact_history
from .prompts import next_action_prompt
from .repl import AgentREPL

_console = Console()


@dataclass
class LoopConfig:
    max_iterations: int = 20
    max_budget_usd: float | None = None
    max_timeout_s: float | None = None
    compaction_threshold: float = 0.85


@dataclass
class LoopResult:
    raw_answer: Any
    iterations_used: int
    messages: list[dict[str, str]]
    budget: BudgetTracker
    wall_time_s: float


def run_loop(
    *,
    orchestrator: Any,  # LLMClient
    repl: AgentREPL,
    system_prompt: str,
    question: str,
    config: LoopConfig,
    budget: BudgetTracker,
    tracer: Any | None = None,
) -> LoopResult:
    """The RLM iteration loop.

    1. Build initial messages [system, user]
    2. For each iteration:
       a. Check budget/timeout
       b. Call orchestrator LLM
       c. Extract code blocks
       d. Execute each block in REPL
       e. Format feedback, append to messages
       f. Check for done() signal
       g. If approaching context limit, compact history
    3. If max iterations reached, force final answer
    4. Return LoopResult
    """
    start_time = time.time()

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    for iteration in range(config.max_iterations):
        # Check budget/timeout
        if budget.should_stop():
            reason = "budget" if budget.budget_exceeded else "timeout"
            _console.print(f"[yellow]⚠️  Stopping: {reason} limit reached[/]")
            break

        _console.print(f"\n[bold blue]{'=' * 60}[/]")
        _console.print(f"[bold blue]ITERATION {iteration + 1}/{config.max_iterations}[/]")
        _console.print(f"[bold blue]{'=' * 60}[/]")

        # Build orchestrator prompt
        orchestrator_messages = messages + [next_action_prompt(question, iteration)]
        prompt = "\n\n".join(m.get("content", "") for m in orchestrator_messages)

        # Call orchestrator LLM
        _console.print("[dim]Querying orchestrator LLM...[/]")
        response = orchestrator.completion(prompt)

        # Track usage
        usage = orchestrator.last_usage
        if usage:
            cost = orchestrator.last_cost_usd or 0.0
            budget.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                model=orchestrator.model,
            )

        # Extract and execute code blocks
        code_blocks = extract_code_blocks(response)

        if code_blocks:
            _console.print(f"[cyan]📝 Found {len(code_blocks)} code block(s) to execute[/]")
            messages.append({"role": "assistant", "content": response})

            last_result: ExecutionResult | None = None
            for idx, code in enumerate(code_blocks, start=1):
                last_result = repl.execute(
                    code,
                    iteration=iteration + 1,
                    block_index=idx,
                    block_total=len(code_blocks),
                )
                feedback = format_execution_feedback(last_result, idx, len(code_blocks))
                messages.append({"role": "user", "content": feedback})

            # Check for final answer
            final_answer = extract_final_answer(last_result, response)
            if final_answer is not None:
                _console.print("[bold green]✅ Final answer found![/]")
                return LoopResult(
                    raw_answer=final_answer,
                    iterations_used=iteration + 1,
                    messages=messages,
                    budget=budget,
                    wall_time_s=time.time() - start_time,
                )
        else:
            _console.print("[dim yellow]No code blocks found in response[/]")
            messages.append({"role": "assistant", "content": response})

            # Check for inline final answer
            final_answer = extract_final_answer(None, response)
            if final_answer is not None:
                _console.print("[bold green]✅ Final answer found (inline)![/]")
                return LoopResult(
                    raw_answer=final_answer,
                    iterations_used=iteration + 1,
                    messages=messages,
                    budget=budget,
                    wall_time_s=time.time() - start_time,
                )

        # Compact history if approaching context limit
        if config.compaction_threshold > 0:
            messages = compact_history(
                messages=messages,
                llm=orchestrator,
                system_prompt=system_prompt,
                threshold_pct=config.compaction_threshold,
            )

    # Max iterations reached — force final answer
    _console.print("[yellow]⚠️  Max iterations reached, forcing final answer[/]")
    messages.append(next_action_prompt(question, config.max_iterations - 1, final_answer=True))
    forced_answer = orchestrator.completion(
        "\n\n".join(m.get("content", "") for m in messages)
    )

    return LoopResult(
        raw_answer=forced_answer,
        iterations_used=config.max_iterations,
        messages=messages,
        budget=budget,
        wall_time_s=time.time() - start_time,
    )
