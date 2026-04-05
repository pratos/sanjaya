"""AgentREPL — sandboxed Python REPL with dynamic tool injection."""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic_monty import MontyRepl

from ..tools.registry import ToolRegistry
from .blocks import ExecutionResult


class AgentREPL:
    """Sandboxed Python REPL with dynamic tool injection.

    Replaces both MontyREPL and VideoMontyREPL.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        context: Any = None,
        os_access: Any | None = None,
    ):
        self.monty = MontyRepl()
        self.registry = registry
        self.context = context
        self._os_access = os_access

        # Per-execution state
        self._stdout_lines: list[str] = []
        self._stderr_lines: list[str] = []
        self._final_value: Any | None = None
        self._is_done: bool = False
        self._llm_queries: list[tuple[str, str]] = []

    def _reset_execution_state(self) -> None:
        self._stdout_lines = []
        self._stderr_lines = []
        self._final_value = None
        self._is_done = False
        self._llm_queries = []

    def _print_callback(self, stream: str, text: str) -> None:
        if stream == "stdout":
            self._stdout_lines.append(text)
        else:
            self._stderr_lines.append(text)

    def set_context(self, context: Any) -> None:
        """Update the context variable."""
        self.context = context

    def set_os_access(self, os_access: Any | None) -> None:
        """Update the Monty filesystem mount."""
        self._os_access = os_access

    def _build_external_functions(
        self,
        extra_builtins: dict[str, Callable[..., Any]] | None = None,
    ) -> dict[str, Callable[..., Any]]:
        """Build the external_functions dict from registry + builtins."""
        fns = self.registry.build_external_functions()
        if extra_builtins:
            fns.update(extra_builtins)
        return fns

    def execute(
        self,
        code: str,
        *,
        extra_builtins: dict[str, Callable[..., Any]] | None = None,
        iteration: int | None = None,
        block_index: int | None = None,
        block_total: int | None = None,
    ) -> ExecutionResult:
        """Execute a code block in the sandbox.

        Injects all registered tools + any extra builtins as external_functions.
        """
        self._reset_execution_state()

        external_functions = self._build_external_functions(extra_builtins)

        start_time = time.time()
        result = None
        stderr = ""

        try:
            feed_kwargs: dict[str, Any] = {
                "external_functions": external_functions,
                "print_callback": self._print_callback,
            }
            if self._os_access is not None:
                feed_kwargs["os"] = self._os_access

            result = self.monty.feed_run(code, **feed_kwargs)
        except Exception as e:
            stderr = str(e)
            self._stderr_lines.append(str(e))

        execution_time = time.time() - start_time

        return ExecutionResult(
            stdout="".join(self._stdout_lines),
            stderr=stderr or "".join(self._stderr_lines),
            result=result,
            execution_time=execution_time,
            final_answer=self._final_value if self._is_done else None,
            llm_queries=self._llm_queries.copy(),
        )

    def mark_done(self, value: Any) -> Any:
        """Called by the done() builtin to signal final answer."""
        self._is_done = True
        self._final_value = value
        return value

    def record_llm_query(self, prompt: str, response: str) -> None:
        """Record a sub-LLM query for tracking."""
        self._llm_queries.append((prompt, response))
