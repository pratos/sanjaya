"""Agent — the single entry point for sanjaya.

Replaces both RLM_REPL and VideoRLM_REPL with a unified, extensible API.
"""

from __future__ import annotations

import time
from typing import Any

from .answer import Answer, Evidence
from .core.budget import BudgetTracker
from .core.loop import LoopConfig, run_loop
from .core.prompts import build_system_prompt
from .core.repl import AgentREPL
from .llm.client import LLMClient
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


class Agent:
    """RLM agent that solves problems by writing code in a sandboxed REPL."""

    def __init__(
        self,
        model: str = "openrouter:openai/gpt-5.3-codex",
        sub_model: str = "openrouter:openai/gpt-4.1-mini",
        vision_model: str | None = None,
        fallback_model: str | None = "openrouter:vikhyatk/moondream2",
        max_iterations: int = 20,
        max_budget_usd: float | None = None,
        max_timeout_s: float | None = None,
        compaction_threshold: float = 0.85,
        tracing: bool = True,
    ):
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
        )
        self._sub_llm = LLMClient(
            model=sub_model,
            vision_model=vision_model or sub_model,
            fallback_model=fallback_model,
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
        def _llm_query(prompt: str) -> str:
            response = self._sub_llm.completion(prompt)
            repl.record_llm_query(prompt, response)
            return response

        def _llm_query_batched(prompts: list[str]) -> list[str]:
            return self._sub_llm.completion_batched(prompts)

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
