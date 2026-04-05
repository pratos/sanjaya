"""Agent — the single entry point for sanjaya.

Replaces both RLM_REPL and VideoRLM_REPL with a unified, extensible API.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.openai import OpenAIProvider

from .answer import Answer, Evidence
from .core.budget import BudgetTracker
from .core.loop import LoopConfig, _model_label, run_loop
from .core.prompts import build_system_prompt
from .core.repl import AgentREPL
from .llm.client import LLMClient, ModelSpec
from .tools.base import Tool, Toolkit
from .tools.builtins import (
    make_context_tool,
    make_done_tool,
    make_get_state_tool,
    make_llm_query_batched_tool,
    make_llm_query_tool,
)
from .tools.registry import ToolRegistry
from .tracing import Tracer


def _resolve_model(
    spec: ModelSpec,
    provider: Provider | None,
    primary: ModelSpec | None = None,
) -> ModelSpec:
    """Resolve a model spec into a concrete Model when possible.

    Resolution order:
    1. Already a Model object — return as-is.
    2. A string + explicit ``provider`` — build an ``OpenAIResponsesModel``
       with that provider (the Responses API is OpenAI's preferred API).
    3. A string + a ``primary`` Model object — reuse the primary's provider
       to build a sibling of the same Model class.
    4. A plain provider-prefixed string like ``"openrouter:openai/gpt-4o"`` —
       return as-is and let pydantic-ai resolve it from env vars.
    """
    if isinstance(spec, Model):
        return spec

    # Strip provider prefix for model name (e.g. "openrouter:openai/gpt-4.1-mini" → "openai/gpt-4.1-mini")
    model_name = spec.split(":", 1)[1] if ":" in spec else spec

    # Explicit provider takes precedence.
    # Use Responses API for direct OpenAI, Chat Completions for everything else
    # (OpenRouter, custom endpoints don't support Responses API).
    if provider is not None:
        try:
            model_cls = OpenAIResponsesModel if isinstance(provider, OpenAIProvider) else OpenAIChatModel
            return model_cls(model_name, provider=provider)
        except Exception:
            return spec

    # Inherit from primary model's provider
    if primary is not None and isinstance(primary, Model):
        inherited = getattr(primary, "_provider", None)
        if inherited is not None:
            try:
                return type(primary)(model_name, provider=inherited)
            except Exception:
                pass

    return spec


class Agent:
    """RLM agent that solves problems by writing code in a sandboxed REPL."""

    def __init__(
        self,
        model: ModelSpec = "openrouter:openai/gpt-5.3-codex",
        sub_model: ModelSpec = "openrouter:openai/gpt-4.1-mini",
        vision_model: ModelSpec | None = None,
        fallback_model: ModelSpec | None = "openrouter:vikhyatk/moondream2",
        *,
        provider: Provider | None = None,
        max_iterations: int = 5,
        max_budget_usd: float | None = None,
        max_timeout_s: float | None = None,
        compaction_threshold: float = 0.85,
        tracing: bool = True,
    ):
        # Resolve all model specs through the provider chain.
        # The primary model is resolved first so siblings can inherit from it.
        model = _resolve_model(model, provider)
        sub_model = _resolve_model(sub_model, provider, primary=model)
        if vision_model is not None:
            vision_model = _resolve_model(vision_model, provider, primary=model)
        if fallback_model is not None:
            fallback_model = _resolve_model(fallback_model, provider, primary=model)

        self.model = model
        self.sub_model = sub_model
        self.vision_model = vision_model
        self.fallback_model = fallback_model
        self.max_iterations = max_iterations
        self.max_budget_usd = max_budget_usd
        self.max_timeout_s = max_timeout_s
        self.compaction_threshold = compaction_threshold

        # LLM clients
        self._orchestrator = LLMClient(
            model=model,
            fallback_model=fallback_model,
            name="orchestrator",
        )
        self._sub_llm = LLMClient(
            model=sub_model,
            vision_model=vision_model or sub_model,
            fallback_model=fallback_model,
            name="sub",
        )

        # Tool registry
        self._registry = ToolRegistry()

        # Tracing
        self._tracer = Tracer(enabled=tracing, track_events=True)

        # Budget tracking (cumulative across ask() calls)
        self._budget = BudgetTracker(
            max_budget_usd=max_budget_usd,
            max_timeout_s=max_timeout_s,
        )

        # State
        self._last_answer: Answer | None = None

    def use(self, *tools_or_toolkits: Tool | Toolkit) -> "Agent":
        """Register tools or toolkits. Chainable."""
        for item in tools_or_toolkits:
            if isinstance(item, Toolkit):
                # Inject LLM client for vision-capable toolkits
                if hasattr(item, "_llm_client"):
                    item._llm_client = self._sub_llm
                self._registry.register_toolkit(item)
            elif isinstance(item, Tool):
                self._registry.register(item)
            else:
                raise TypeError(f"Expected Tool or Toolkit, got {type(item).__name__}")
        return self

    def ask(
        self,
        question: str,
        *,
        context: Any = None,
        video: str | None = None,
        subtitle: str | None = None,
    ) -> Answer:
        """Run the RLM loop and return a structured answer."""
        start_time = time.time()

        # Auto-register VideoToolkit if video= provided and none registered
        if video and not self._has_video_toolkit():
            from .tools.video import VideoToolkit
            vt = VideoToolkit()
            vt._llm_client = self._sub_llm
            self._registry.register_toolkit(vt)

        # Build context dict for toolkits
        context_dict: dict[str, Any] = {
            "question": question,
            "context": context,
            "video": video,
            "subtitle": subtitle,
        }

        # Setup toolkits
        for toolkit in self._registry.toolkits:
            toolkit.setup(context_dict)

        # Build a fresh registry for this run with builtins
        run_registry = ToolRegistry()

        # Copy all registered tools
        for tool in self._registry.all_tools():
            run_registry.register(tool)

        # Create REPL
        repl = AgentREPL(
            registry=run_registry,
            context=context,
        )

        # Set up OS access from video toolkit if available
        for toolkit in self._registry.toolkits:
            if hasattr(toolkit, "get_os_access"):
                os_access = toolkit.get_os_access()
                if os_access is not None:
                    repl.set_os_access(os_access)

        # Create builtin tool closures
        sub_model_name = _model_label(self.sub_model)

        def _llm_query(prompt: str) -> str:
            with self._tracer.llm_call(model=sub_model_name, prompt=prompt) as llm_trace:
                response = self._sub_llm.completion(prompt)
                repl.record_llm_query(prompt, response)

                llm_trace.record_response(response)

                usage = self._sub_llm.last_usage
                if usage:
                    llm_trace.record_usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                    llm_trace.record_llm_cost(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        model_name=sub_model_name,
                    )

                metadata = self._sub_llm.last_call_metadata
                if metadata:
                    llm_trace.record(
                        model_used=metadata.model_used,
                        provider=metadata.provider,
                        duration_seconds=metadata.duration_seconds,
                        fallback_used=metadata.fallback_used,
                        cost_usd=metadata.cost_usd,
                    )

            return response

        def _llm_query_batched(prompts: list[str]) -> list[str]:
            responses = self._sub_llm.completion_batched(prompts)
            for prompt, response in zip(prompts, responses):
                repl.record_llm_query(prompt, response)
                with self._tracer.llm_call(model=sub_model_name, prompt=prompt, batched=True) as llm_trace:
                    llm_trace.record_response(response)
            return responses

        def _get_state() -> dict[str, Any]:
            state: dict[str, Any] = {
                "tools": [t.name for t in run_registry.all_tools()],
                "iteration": "in_progress",
                "budget": self._budget.summary(),
            }
            for toolkit in self._registry.toolkits:
                toolkit_state = toolkit.get_state()
                if toolkit_state:
                    state.update(toolkit_state)
            return state

        # Register builtins
        run_registry.register(make_context_tool(lambda: repl.context))
        run_registry.register(make_llm_query_tool(_llm_query))
        run_registry.register(make_llm_query_batched_tool(_llm_query_batched))
        run_registry.register(make_done_tool(repl.mark_done))
        run_registry.register(make_get_state_tool(_get_state))

        # Build system prompt
        toolkit_sections = [
            tk.prompt_section()
            for tk in self._registry.toolkits
            if tk.prompt_section()
        ]

        context_metadata: dict[str, Any] = {}
        if context is not None:
            context_metadata["context_type"] = type(context).__name__
            if hasattr(context, "__len__"):
                context_metadata["context_length"] = len(context)
        if video:
            context_metadata["video"] = video

        system_prompt = build_system_prompt(
            registry=run_registry,
            context_metadata=context_metadata,
            toolkit_sections=toolkit_sections,
        )

        # Run the loop
        config = LoopConfig(
            max_iterations=self.max_iterations,
            max_budget_usd=self.max_budget_usd,
            max_timeout_s=self.max_timeout_s,
            compaction_threshold=self.compaction_threshold,
        )

        model_name = _model_label(self.model)
        with self._tracer.completion(question=question, model=model_name) as comp_trace:
            loop_result = run_loop(
                orchestrator=self._orchestrator,
                repl=repl,
                system_prompt=system_prompt,
                question=question,
                config=config,
                budget=self._budget,
                tracer=self._tracer,
            )

            # Collect evidence from toolkits
            evidence: list[Evidence] = []
            for toolkit in self._registry.toolkits:
                evidence.extend(toolkit.build_evidence())

            # Teardown toolkits
            for toolkit in self._registry.toolkits:
                toolkit.teardown()

            # Build Answer
            answer = Answer(
                question=question,
                text=str(loop_result.raw_answer),
                evidence=evidence,
                iterations=loop_result.iterations_used,
                cost_usd=self._budget.total_cost_usd,
                input_tokens=self._budget.total_input_tokens,
                output_tokens=self._budget.total_output_tokens,
                wall_time_s=round(time.time() - start_time, 2),
            )

            # Record cost on the top-level span
            comp_trace.record_usage(
                input_tokens=self._budget.total_input_tokens,
                output_tokens=self._budget.total_output_tokens,
            )
            comp_trace.record_llm_cost(
                input_tokens=self._budget.total_input_tokens,
                output_tokens=self._budget.total_output_tokens,
                model_name=model_name,
            )

        self._last_answer = answer
        return answer

    @property
    def last_answer(self) -> Answer | None:
        """Most recent answer, for notebook inspection."""
        return self._last_answer

    @property
    def cost_so_far(self) -> float:
        """Cumulative USD spent across all ask() calls."""
        return self._budget.total_cost_usd

    def reset(self) -> None:
        """Clear all state (budget, history, workspace)."""
        self._budget = BudgetTracker(
            max_budget_usd=self.max_budget_usd,
            max_timeout_s=self.max_timeout_s,
        )
        self._last_answer = None
        self._registry = ToolRegistry()
        self._tracer = Tracer(enabled=self._tracer._enabled_requested, track_events=True)

    def _has_video_toolkit(self) -> bool:
        """Check if a VideoToolkit is already registered."""
        try:
            from .tools.video import VideoToolkit
            return any(isinstance(tk, VideoToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False
