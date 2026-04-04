"""RLM_REPL - Main orchestrator for recursive language model operations."""

from typing import Any, Optional
from uuid import uuid4

from rich.console import Console

from .logger.repl_logger import REPLEnvLogger
from .logger.root_logger import ColorfulLogger
from .repl import ExecutionResult, MontyREPL
from .rlm import RLM
from .tracing import get_tracer
from .utils import utils
from .utils.llm import LLMClient
from .utils.prompts import DEFAULT_QUERY, build_system_prompt, next_action_prompt

_console = Console()


class RLM_REPL(RLM):
    """Main orchestrator for recursive language model operations using a REPL environment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai-responses:gpt-5.3-codex",
        recursive_model: str = "openai-responses:gpt-5.4-mini",
        max_iterations: int = 20,
        depth: int = 0,
        enable_logging: bool = False,
    ):
        """Initialize RLM_REPL.

        Args:
            api_key: API key (defaults to environment variable if None)
            model: Pydantic AI model string for orchestration (e.g., 'openai-responses:gpt-5.3-codex')
            recursive_model: Pydantic AI model string for sub-LLM calls
            max_iterations: Maximum number of completion loop iterations
            depth: Recursion depth (unused in current implementation)
            enable_logging: Enable colorful logging output
        """
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = LLMClient(model=model, api_key=api_key)
        self.repl_env: Optional[MontyREPL] = None
        self.depth = depth  # Unused in this version
        self._max_iterations = max_iterations

        self.logger = ColorfulLogger(enabled=enable_logging)
        self.repl_env_logger = REPLEnvLogger(enabled=enable_logging)

        self.messages: list[dict[str, str]] = []
        self.query: Optional[str] = None
        self._last_execution_result: Optional[ExecutionResult] = None

    def setup_context(
        self,
        context: list[str] | str | list[dict[str, str]],
        query: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Initialize the RLM context, conversation history, and REPL environment.

        Args:
            context: Large context to analyze (messages, string, or dict)
            query: User's question (defaults to DEFAULT_QUERY if None)

        Returns:
            Initialized message list with system prompt
        """
        if query is None:
            query = DEFAULT_QUERY
        self.query = query

        self.logger.log_query_start(query)

        self.messages = build_system_prompt()
        self.logger.log_initial_messages(self.messages)

        # Convert context and create MontyREPL with unified context
        context_data, context_str = utils.convert_context_for_repl(context)
        # MontyREPL takes context directly (no separate json/str params)
        unified_context = context_data if context_data is not None else context_str
        self.repl_env = MontyREPL(
            recursive_model=self.recursive_model,
            context=unified_context,
        )
        self._last_execution_result = None

        return self.messages

    def completion(
        self,
        context: Any,
        query: str = "",
    ) -> str:
        """Execute the main recursive completion loop.

        Args:
            context: Context data to analyze
            query: User's question

        Returns:
            Final answer extracted from LLM response or REPL variable
        """
        self.setup_context(context, query or None)
        current_query = self.query or ""

        # Log context size
        ctx_len = len(context) if hasattr(context, "__len__") else 0
        _console.print("\n[bold]🚀 Starting RLM completion[/]")
        _console.print(f"[dim]   Context size: {ctx_len:,} chars[/]")
        _console.print(f"[dim]   Query: {current_query[:80]}{'...' if len(current_query) > 80 else ''}[/]")
        _console.print(f"[dim]   Max iterations: {self._max_iterations}[/]")

        tracer = get_tracer()
        run_id = uuid4().hex[:12]
        total_code_blocks_executed = 0
        total_sub_llm_calls = 0

        with tracer.rlm_completion(
            query=current_query,
            context_size=ctx_len if isinstance(ctx_len, int) else 0,
            max_iterations=self._max_iterations,
            orchestrator_model=self.model,
            sub_llm_model=self.recursive_model,
            run_id=run_id,
        ) as completion_ctx:
            for iteration in range(self._max_iterations):
                with tracer.iteration(
                    iteration=iteration + 1,
                    max_iterations=self._max_iterations,
                    message_count=len(self.messages),
                    run_id=run_id,
                ) as iter_ctx:
                    _console.print(f"\n[bold blue]{'=' * 60}[/]")
                    _console.print(f"[bold blue]ITERATION {iteration + 1}/{self._max_iterations}[/]")
                    _console.print(f"[bold blue]{'=' * 60}[/]")

                    # Generate action prompt and query LLM
                    _console.print("[dim]Querying orchestrator LLM...[/]")

                    orchestrator_messages = self.messages + [next_action_prompt(current_query, iteration)]
                    orchestrator_prompt = "\n\n".join(msg.get("content", "") for msg in orchestrator_messages)
                    prompt_chars = len(orchestrator_prompt)

                    with tracer.orchestrator_call(
                        model=self.model,
                        prompt_chars=prompt_chars,
                        run_id=run_id,
                        iteration=iteration + 1,
                        prompt_content=orchestrator_prompt,
                    ) as orchestrator_ctx:
                        orchestrator_ctx.record_content(prompt=orchestrator_prompt)
                        response = self.llm.completion(orchestrator_messages)
                        orchestrator_ctx.record_content(response=response)

                        usage = self.llm.last_usage
                        if usage is not None:
                            orchestrator_ctx.record_usage(
                                input_tokens=usage.input_tokens,
                                output_tokens=usage.output_tokens,
                            )

                        metadata = self.llm.last_call_metadata or {}
                        orchestrator_ctx.record_duration(metadata.get("duration_seconds"))
                        orchestrator_ctx.record(
                            requested_model=metadata.get("requested_model"),
                            model_used=metadata.get("model_used"),
                            provider=metadata.get("provider"),
                            fallback_used=metadata.get("fallback_used"),
                        )

                    code_blocks = utils.find_code_blocks(response)
                    has_tool_calls = code_blocks is not None and len(code_blocks) > 0
                    self.logger.log_model_response(response, has_tool_calls)

                    iter_ctx.record(
                        response_chars=len(response),
                        code_blocks_count=len(code_blocks) if code_blocks else 0,
                        has_tool_calls=has_tool_calls,
                    )

                    if has_tool_calls:
                        code_block_count = len(code_blocks)
                        total_code_blocks_executed += code_block_count

                        _console.print(f"[cyan]📝 Found {code_block_count} code block(s) to execute[/]")
                        self.messages, self._last_execution_result = utils.process_code_execution(
                            response,
                            self.messages,
                            self.repl_env,
                            self.repl_env_logger,
                            self.logger,
                            run_id=run_id,
                            iteration=iteration + 1,
                        )

                        if self._last_execution_result:
                            sub_llm_calls = len(self._last_execution_result.llm_queries)
                            total_sub_llm_calls += sub_llm_calls

                            iter_ctx.record(
                                sub_llm_calls=sub_llm_calls,
                                has_final_answer=self._last_execution_result.final_answer is not None,
                            )
                    else:
                        _console.print("[dim yellow]No code blocks found in response[/]")
                        self.messages.append({"role": "assistant", "content": f"You responded with: {response}"})
                        self._last_execution_result = None
                        iter_ctx.record(sub_llm_calls=0)

                    # Check for final answer (from done() call or FINAL() directive)
                    final_answer = utils.check_for_final_answer(
                        response, self.repl_env, self.logger, self._last_execution_result
                    )
                    if final_answer:
                        _console.print("[bold green]✅ Final answer found![/]")
                        iter_ctx.record(iteration_status="final_answer", messages_after_iteration=len(self.messages))

                        tracer.log_final_answer(str(final_answer), iteration + 1, run_id=run_id)
                        completion_ctx.record(
                            final_iteration=iteration + 1,
                            iterations_used=iteration + 1,
                            total_code_blocks_executed=total_code_blocks_executed,
                            total_sub_llm_calls=total_sub_llm_calls,
                            completion_status="final_answer",
                        )
                        completion_ctx.record_final_answer(str(final_answer)[:500], forced=False)
                        self.logger.log_final_response(final_answer)
                        return final_answer
                    else:
                        iter_ctx.record(iteration_status="continue", messages_after_iteration=len(self.messages))
                        _console.print("[dim]No final answer yet, continuing...[/]")

            # Max iterations reached - force final answer
            print("No final answer found in any iteration")
            tracer.log_max_iterations_reached(self._max_iterations, run_id=run_id)
            self.messages.append(next_action_prompt(current_query, self._max_iterations - 1, final_answer=True))
            final_answer = self.llm.completion(self.messages)

            forced_usage = self.llm.last_usage
            if forced_usage is not None:
                completion_ctx.record_usage(
                    input_tokens=forced_usage.input_tokens,
                    output_tokens=forced_usage.output_tokens,
                )

            forced_metadata = self.llm.last_call_metadata or {}
            completion_ctx.record_duration(forced_metadata.get("duration_seconds"))
            completion_ctx.record(
                forced_model_used=forced_metadata.get("model_used"),
                forced_provider=forced_metadata.get("provider"),
                forced_fallback_used=forced_metadata.get("fallback_used"),
            )

            completion_ctx.record(
                forced_answer=True,
                iterations_used=self._max_iterations,
                total_code_blocks_executed=total_code_blocks_executed,
                total_sub_llm_calls=total_sub_llm_calls,
                completion_status="forced_final_answer",
            )
            completion_ctx.record_final_answer(str(final_answer)[:500], forced=True)
            self.logger.log_final_response(final_answer)
            return final_answer

    def cost_summary(self) -> dict[str, Any]:
        """Retrieve cost summary - use Logfire dashboard for cost tracking."""
        raise NotImplementedError("Cost tracking handled by Logfire dashboard")

    def reset(self) -> None:
        """Reset the REPL environment and message history."""
        self.repl_env = MontyREPL(recursive_model=self.recursive_model)
        self.messages = []
        self.query = None
        self._last_execution_result = None
