"""Monty-based REPL environment for recursive language model operations."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic_monty import MontyRepl
from rich.console import Console

from .rlm import RLM
from .tracing import get_tracer
from .utils.llm import LLMClient

_console = Console()


class Sub_RLM(RLM):
    """Simplified RLM implementation for recursive sub-calls within the REPL."""

    def __init__(self, model: str = "openai-responses:gpt-5.4-mini"):
        """Initialize Sub_RLM.

        Args:
            model: Pydantic AI model string (e.g., 'openai-responses:gpt-5.4-mini')
        """
        self.model = model
        self.client = LLMClient(model=model)

    def completion(
        self,
        context: Any,
        query: str = "",
        trace_meta: dict[str, Any] | None = None,
    ) -> str:
        """Execute a simple LLM query.

        For Sub_RLM, the context is treated as the prompt directly.
        The query parameter is ignored (for interface compatibility).

        Args:
            context: Prompt to send to the LLM (str, dict, or list)
            query: Ignored (for RLM interface compatibility)
            trace_meta: Optional tracing metadata (run/iteration/code-block context)

        Returns:
            LLM response or error message
        """
        tracer = get_tracer()
        ctx_str = str(context)
        span_meta = trace_meta or {}

        with tracer.sub_llm_call(model=self.model, context=ctx_str, **span_meta) as t:
            t.record_content(prompt=ctx_str)

            try:
                response = self.client.completion(context, timeout=300)
                t.record_response(response)

                usage = self.client.last_usage
                if usage is not None:
                    t.record_usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )

                metadata = self.client.last_call_metadata or {}
                t.record_duration(metadata.get("duration_seconds"))
                t.record(
                    requested_model=metadata.get("requested_model"),
                    model_used=metadata.get("model_used"),
                    provider=metadata.get("provider"),
                    fallback_used=metadata.get("fallback_used"),
                )

                return response
            except Exception as e:
                error_msg = f"Error in Sub_RLM completion: {str(e)}"
                t.record_error(e)
                return error_msg

    def cost_summary(self) -> dict[str, float]:
        raise NotImplementedError("Cost tracking not implemented for Sub_RLM")

    def reset(self) -> None:
        raise NotImplementedError("Reset not implemented for Sub_RLM")


@dataclass
class ExecutionResult:
    """Result of code execution in MontyREPL."""

    stdout: str
    stderr: str
    result: Any
    execution_time: float
    final_answer: Optional[Any] = None
    llm_queries: list[tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"ExecutionResult(stdout={self.stdout!r}, stderr={self.stderr!r}, "
            f"result={self.result!r}, execution_time={self.execution_time}, "
            f"final_answer={self.final_answer!r})"
        )


class MontyREPL:
    """Sandboxed Python REPL using pydantic-monty with LLM integration."""

    def __init__(
        self,
        recursive_model: str = "openai-responses:gpt-5.4-mini",
        context: Any = None,
    ):
        """Initialize MontyREPL.

        Args:
            recursive_model: Pydantic AI model string for sub-LLM queries
            context: Context data accessible via get_context()
        """
        self.repl = MontyRepl()
        self.sub_rlm = Sub_RLM(model=recursive_model)
        self.context = context

        # Execution state (reset per code_execution call)
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._final_value: Optional[Any] = None
        self._is_done: bool = False
        self._llm_queries: list[tuple[str, str]] = []

        # Trace correlation metadata (updated by code_execution calls)
        self._trace_run_id: str | None = None
        self._trace_iteration: int | None = None
        self._trace_code_block_index: int | None = None
        self._trace_code_block_total: int | None = None

    def _reset_execution_state(self) -> None:
        """Reset per-execution state."""
        self._stdout_lines = []
        self._stderr_lines = []
        self._final_value = None
        self._is_done = False
        self._llm_queries = []
        self._trace_code_block_index = None
        self._trace_code_block_total = None

    def _print_callback(self, stream: str, text: str) -> None:
        """Capture print output."""
        if stream == "stdout":
            self._stdout_lines.append(text)
        else:
            self._stderr_lines.append(text)

    def _current_trace_meta(self) -> dict[str, Any]:
        """Return trace correlation metadata for nested spans."""
        meta: dict[str, Any] = {}
        if self._trace_run_id is not None:
            meta["run_id"] = self._trace_run_id
        if self._trace_iteration is not None:
            meta["iteration"] = self._trace_iteration
        if self._trace_code_block_index is not None:
            meta["code_block_index"] = self._trace_code_block_index
        if self._trace_code_block_total is not None:
            meta["code_block_total"] = self._trace_code_block_total
        return meta

    def _get_context(self) -> Any:
        """External function: return context data."""
        ctx = self.context
        ctx_len = len(ctx) if hasattr(ctx, "__len__") else "N/A"
        ctx_type = type(ctx).__name__
        _console.print(f"[yellow]📄 get_context() called → {ctx_type}, length: {ctx_len:,}[/]")
        return ctx

    def _llm_query(self, prompt: str) -> str:
        """External function: query sub-LLM."""
        prompt_preview = prompt[:100].replace("\n", "\\n") + ("..." if len(prompt) > 100 else "")
        _console.print(
            f"[magenta]🤖 llm_query() called ({len(prompt):,} chars)[/]\n[dim magenta]   Preview: {prompt_preview}[/]"
        )
        start_time = time.time()
        response = self.sub_rlm.completion(
            prompt,
            trace_meta=self._current_trace_meta(),
        )  # context=prompt for Sub_RLM
        elapsed = time.time() - start_time
        _console.print(f"[magenta]✓ llm_query() returned in {elapsed:.1f}s ({len(response):,} chars)[/]")
        self._llm_queries.append((prompt, response))
        return response

    def _done(self, value: Any) -> Any:
        """External function: signal final answer."""
        _console.print(f"[bold green]🏁 done() called with: {value!r}[/]")
        self._is_done = True
        self._final_value = value
        return value  # Return so LLM can use it in expressions

    def code_execution(
        self,
        code: str,
        run_id: str | None = None,
        iteration: int | None = None,
        code_block_index: int | None = None,
        code_block_total: int | None = None,
    ) -> ExecutionResult:
        """Execute Python code in the sandboxed environment.

        Args:
            code: Python code to execute
            run_id: Trace run identifier
            iteration: Current orchestration iteration
            code_block_index: 1-based code block index in current response
            code_block_total: Total code blocks in current response

        Returns:
            ExecutionResult with stdout, result, and optional final_answer
        """
        self._reset_execution_state()

        if run_id is not None:
            self._trace_run_id = run_id
        if iteration is not None:
            self._trace_iteration = iteration
        if code_block_index is not None:
            self._trace_code_block_index = code_block_index
        if code_block_total is not None:
            self._trace_code_block_total = code_block_total

        trace_meta = self._current_trace_meta()

        start_time = time.time()
        tracer = get_tracer()

        result = None
        stderr = ""

        with tracer.code_execution(code=code, **trace_meta) as t:
            try:
                result = self.repl.feed_run(
                    code,
                    external_functions={
                        "get_context": self._get_context,
                        "llm_query": self._llm_query,
                        "done": self._done,
                    },
                    print_callback=self._print_callback,
                )
            except Exception as e:
                stderr = str(e)
                self._stderr_lines.append(str(e))
                t.record_error(e)

            execution_time = time.time() - start_time

            exec_result = ExecutionResult(
                stdout="".join(self._stdout_lines),
                stderr=stderr or "".join(self._stderr_lines),
                result=result,
                execution_time=execution_time,
                final_answer=self._final_value if self._is_done else None,
                llm_queries=self._llm_queries.copy(),
            )

            t.record(
                execution_time=round(execution_time, 2),
                has_final_answer=self._is_done,
                llm_queries_count=len(self._llm_queries),
                stdout_preview=exec_result.stdout[:200] if exec_result.stdout else None,
                run_id=self._trace_run_id,
                iteration=self._trace_iteration,
                code_block_index=self._trace_code_block_index,
                code_block_total=self._trace_code_block_total,
            )

        return exec_result

    @property
    def locals(self) -> dict:
        """Compatibility property - returns empty dict.

        Note: Monty doesn't expose locals. Use done() or return values instead.
        """
        return {}
