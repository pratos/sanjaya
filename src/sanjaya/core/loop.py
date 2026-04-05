"""The RLM iteration loop — the core engine that drives the agent."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from rich.console import Console

from contextlib import contextmanager

from .blocks import ExecutionResult, extract_code_blocks, extract_final_answer, format_execution_feedback
from .budget import BudgetTracker
from .compaction import compact_history
from .prompts import next_action_prompt
from .repl import AgentREPL
from ..tracing.tracer import Tracer

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


def _run_iteration(
    *,
    orchestrator: Any,
    repl: AgentREPL,
    messages: list[dict[str, str]],
    question: str,
    iteration: int,
    config: LoopConfig,
    budget: BudgetTracker,
    tracer: Tracer | None,
    model_name: str,
    start_time: float,
) -> LoopResult | None:
    """Run a single iteration, wrapped in tracer spans. Returns LoopResult if done."""

    with tracer.iteration(iteration=iteration + 1) if tracer else _nullctx() as iter_trace:
        # Build orchestrator prompt
        orchestrator_messages = messages + [next_action_prompt(question, iteration)]
        prompt = "\n\n".join(m.get("content", "") for m in orchestrator_messages)

        # Call orchestrator LLM
        _console.print("[dim]Querying orchestrator LLM...[/]")

        with tracer.orchestrator_call(model=model_name) if tracer else _nullctx() as orch_trace:
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
                if orch_trace:
                    orch_trace.record_usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                    orch_trace.record_llm_cost(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        model_name=model_name,
                    )

        # Extract and execute code blocks
        code_blocks = extract_code_blocks(response)

        if code_blocks:
            _console.print(f"[cyan]📝 Found {len(code_blocks)} code block(s) to execute[/]")
            messages.append({"role": "assistant", "content": response})

            last_result: ExecutionResult | None = None
            for idx, code in enumerate(code_blocks, start=1):
                with tracer.code_execution(code=code, block_index=idx) if tracer else _nullctx():
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
                if iter_trace:
                    iter_trace.record_final_answer(str(final_answer))
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
                if iter_trace:
                    iter_trace.record_final_answer(str(final_answer))
                return LoopResult(
                    raw_answer=final_answer,
                    iterations_used=iteration + 1,
                    messages=messages,
                    budget=budget,
                    wall_time_s=time.time() - start_time,
                )

        # Compact history if approaching context limit
        if config.compaction_threshold > 0:
            messages[:] = compact_history(
                messages=messages,
                llm=orchestrator,
                system_prompt=messages[0]["content"] if messages else "",
                threshold_pct=config.compaction_threshold,
            )

    return None


@contextmanager
def _nullctx():
    yield None


def _model_label(model: Any) -> str:
    """Extract a human-readable model name string."""
    if isinstance(model, str):
        return model
    return getattr(model, "model_name", type(model).__name__)


def run_loop(
    *,
    orchestrator: Any,  # LLMClient
    repl: AgentREPL,
    system_prompt: str,
    question: str,
    config: LoopConfig,
    budget: BudgetTracker,
    tracer: Tracer | None = None,
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
    model_name = _model_label(orchestrator.model)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
    ]

    for iteration in range(config.max_iterations):
        # Check budget/timeout
        if budget.should_stop():
            reason = "budget" if budget.budget_exceeded else "timeout"
            _console.print(f"[yellow]⚠️  Stopping: {reason} limit reached[/]")
            break

        _console.print(f"\n[bold cyan]{'=' * 60}[/]")
        _console.print(f"[bold cyan]ITERATION {iteration + 1}/{config.max_iterations}[/]")
        _console.print(f"[bold cyan]{'=' * 60}[/]")

        result = _run_iteration(
            orchestrator=orchestrator,
            repl=repl,
            messages=messages,
            question=question,
            iteration=iteration,
            config=config,
            budget=budget,
            tracer=tracer,
            model_name=model_name,
            start_time=start_time,
        )
        if result is not None:
            return result

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
