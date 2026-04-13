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
from .core.loop import LoopConfig, LoopResult, _model_label, run_loop
from .core.prompts import build_system_prompt
from .core.repl import AgentREPL
from .llm.client import LLMClient, ModelSpec
from .prompts import PromptConfig
from .tools.base import Tool, Toolkit
from .tools.builtins import (
    make_context_tool,
    make_done_tool,
    make_get_state_tool,
    make_llm_query_batched_tool,
    make_llm_query_tool,
    make_rlm_query_batched_tool,
    make_rlm_query_tool,
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
        model: ModelSpec = "openrouter:z-ai/glm-5.1",
        sub_model: ModelSpec = "openrouter:openai/gpt-4.1-mini",
        vision_model: ModelSpec | None = "moondream-station:moondream3-preview",
        caption_model: ModelSpec | None = "moondream-station:moondream3-preview",
        fallback_model: ModelSpec | None = None,
        critic_model: ModelSpec | None = "openrouter:qwen/qwen3-30b-a3b-thinking-2507",
        *,
        prompts: PromptConfig | None = None,
        provider: Provider | None = None,
        max_iterations: int = 8,
        max_depth: int = 1,
        max_budget_usd: float | None = None,
        max_timeout_s: float | None = None,
        compaction_threshold: float = 0.85,
        critic_threshold: int = 70,
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

        self._prompts = prompts or PromptConfig()
        self.model = model
        self.sub_model = sub_model
        self.vision_model = vision_model
        self.caption_model = caption_model
        self.fallback_model = fallback_model
        self.max_iterations = max_iterations
        self._max_depth = max_depth
        self._depth = 0  # always 0 for user-created agents
        self.max_budget_usd = max_budget_usd
        self.max_timeout_s = max_timeout_s
        self.compaction_threshold = compaction_threshold
        self.critic_threshold = critic_threshold

        # LLM clients
        self._orchestrator = LLMClient(
            model=model,
            fallback_model=fallback_model,
            name="root_llm",
        )
        self._sub_llm = LLMClient(
            model=sub_model,
            vision_model=vision_model or sub_model,
            fallback_model=fallback_model,
            name="sub_llm",
        )
        self._critic = LLMClient(model=critic_model, name="critic") if critic_model else None

        # Captioner (separate from sub_llm — used only by caption_frames)
        self._captioner: Any = None
        if caption_model is not None:
            from .llm.moondream import MOONDREAM_STATION_BASE, MoondreamVisionClient, is_moondream_spec
            if is_moondream_spec(caption_model):
                spec = str(caption_model)
                use_station = spec.startswith("moondream-station:")
                model_id = spec.split(":", 1)[1] if ":" in spec else "moondream3-preview"
                try:
                    self._captioner = MoondreamVisionClient(
                        model=model_id,
                        base_url=MOONDREAM_STATION_BASE if use_station else None,
                    )
                except Exception:
                    pass

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
                # Inject LLM client and tracer for vision-capable toolkits
                if hasattr(item, "_llm_client"):
                    item._llm_client = self._sub_llm
                if hasattr(item, "_tracer"):
                    item._tracer = self._tracer
                if hasattr(item, "_budget"):
                    item._budget = self._budget
                if hasattr(item, "_captioner") and self._captioner is not None:
                    item._captioner = self._captioner
                item._prompt_config = self._prompts
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
        document: str | list[str] | None = None,
        image: str | list[str] | None = None,
    ) -> Answer:
        """Run the RLM loop and return a structured answer."""
        start_time = time.time()

        # Auto-register VideoToolkit if video= provided and none registered
        if video and not self._has_video_toolkit():
            from .tools.video import VideoToolkit
            vt = VideoToolkit()
            vt._llm_client = self._sub_llm
            vt._tracer = self._tracer
            vt._budget = self._budget
            if self._captioner is not None:
                vt._captioner = self._captioner
            vt._prompt_config = self._prompts
            self._registry.register_toolkit(vt)

        # Auto-register DocumentToolkit if document= provided and none registered
        if document and not self._has_document_toolkit():
            from .tools.document import DocumentToolkit
            dt = DocumentToolkit()
            dt._llm_client = self._sub_llm
            dt._tracer = self._tracer
            dt._budget = self._budget
            dt._prompt_config = self._prompts
            self._registry.register_toolkit(dt)

        # Auto-register ImageToolkit if image= provided and none registered
        if image and not self._has_image_toolkit():
            from .tools.image import ImageToolkit
            it = ImageToolkit()
            it._llm_client = self._sub_llm
            it._tracer = self._tracer
            it._budget = self._budget
            it._prompt_config = self._prompts
            if self._captioner is not None:
                it._captioner = self._captioner
            self._registry.register_toolkit(it)

        # Build context dict for toolkits (modality classified later, inside the
        # completion span, so the sub_llm call shows up under sanjaya.completion)
        context_dict: dict[str, Any] = {
            "question": question,
            "context": context,
            "video": video,
            "subtitle": subtitle,
            "document": document,
            "image": image,
            "modality": "balanced",
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
                    cost = self._sub_llm.last_cost_usd or 0.0
                    self._budget.record(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=cost,
                        model=sub_model_name,
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

        # Register rlm_query builtins (only when recursion is enabled)
        if self._max_depth > 1:
            def _rlm_query(prompt: str) -> str:
                return self._subcall(
                    prompt,
                    depth=1,
                    parent_run_registry=run_registry,
                    parent_context=context,
                )

            def _rlm_query_batched(prompts: list[str]) -> list[str]:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = [
                        pool.submit(
                            self._subcall,
                            p,
                            depth=1,
                            parent_run_registry=run_registry,
                            parent_context=context,
                        )
                        for p in prompts
                    ]
                    return [f.result() for f in futures]

            run_registry.register(make_rlm_query_tool(_rlm_query))
            run_registry.register(make_rlm_query_batched_tool(_rlm_query_batched))

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
            max_depth=self._max_depth,
        )

        # Run the loop
        config = LoopConfig(
            max_iterations=self.max_iterations,
            max_budget_usd=self.max_budget_usd,
            max_timeout_s=self.max_timeout_s,
            compaction_threshold=self.compaction_threshold,
            critic_threshold=self.critic_threshold,
        )

        model_name = _model_label(self.model)
        with self._tracer.completion(question=question, model=model_name) as comp_trace:
            # Classify question modality inside the span so the sub_llm call
            # is nested under sanjaya.completion in traces.
            if video:
                from .core.schema import classify_question_modality
                modality = classify_question_modality(question, self._sub_llm)
                context_dict["modality"] = modality
                # Update toolkit modality and rebuild the system prompt
                # so the correct strategy prompt is used.
                for toolkit in self._registry.toolkits:
                    if hasattr(toolkit, "_modality"):
                        toolkit._modality = modality
                toolkit_sections = [
                    tk.prompt_section()
                    for tk in self._registry.toolkits
                    if tk.prompt_section()
                ]
                system_prompt = build_system_prompt(
                    registry=run_registry,
                    context_metadata=context_metadata,
                    toolkit_sections=toolkit_sections,
                    max_depth=self._max_depth,
                )

            # Generate or use provided answer schema
            from .core.schema import generate_answer_schema, schema_to_prompt_section

            if self._prompts.answer_schema is not None:
                answer_schema = self._prompts.answer_schema
            else:
                with self._tracer._span("sanjaya.schema_generation", question_chars=len(question)):
                    answer_schema = generate_answer_schema(
                        question=question,
                        llm_client=self._sub_llm,
                    )
            self._current_schema = answer_schema
            system_prompt = system_prompt + "\n\n" + schema_to_prompt_section(answer_schema)

            loop_result = run_loop(
                orchestrator=self._orchestrator,
                repl=repl,
                system_prompt=system_prompt,
                question=question,
                config=config,
                budget=self._budget,
                tracer=self._tracer,
                critic=self._critic,
                answer_schema=answer_schema,
                critic_prompt=self._prompts.critic,
            )

            # Collect evidence from toolkits
            evidence: list[Evidence] = []
            for toolkit in self._registry.toolkits:
                evidence.extend(toolkit.build_evidence())

            # Teardown toolkits
            for toolkit in self._registry.toolkits:
                toolkit.teardown()

            # Auto-persist trace
            self._persist_trace(
                question=question,
                loop_result=loop_result,
                evidence=evidence,
            )

            # Build Answer
            raw = loop_result.raw_answer
            if isinstance(raw, dict):
                text = raw.get("summary") or raw.get("answer") or str(raw)
                data = raw
            else:
                text = str(raw)
                data = None

            answer = Answer(
                question=question,
                text=text,
                data=data,
                evidence=evidence,
                iterations=loop_result.iterations_used,
                cost_usd=self._budget.total_cost_usd,
                input_tokens=self._budget.total_input_tokens,
                output_tokens=self._budget.total_output_tokens,
                wall_time_s=round(time.time() - start_time, 2),
            )

            # Record final answer on the completion span so run_end SSE
            # event includes it (the UI reads final_answer from run_end).
            is_forced = loop_result.iterations_used >= config.max_iterations
            comp_trace.record_final_answer(text, forced=is_forced)
            comp_trace.record(answer_preview=text[:200])

            # Record tokens and cost on the top-level span.
            # Use the budget's accumulated cost (which prices each call
            # with its actual model) instead of re-pricing all tokens
            # under the orchestrator model — that caused cost divergence
            # between the UI and Logfire.
            comp_trace.record(
                sanjaya_cost_usd=self._budget.total_cost_usd,
                sanjaya_input_tokens=self._budget.total_input_tokens,
                sanjaya_output_tokens=self._budget.total_output_tokens,
            )

        self._last_answer = answer
        return answer

    # Names of builtin tools that are re-created per child (not inherited)
    _BUILTIN_NAMES = frozenset({
        "llm_query", "llm_query_batched",
        "rlm_query", "rlm_query_batched",
        "done", "get_context", "get_state",
    })

    def _subcall(
        self,
        prompt: str,
        *,
        depth: int,
        parent_run_registry: ToolRegistry,
        parent_context: Any = None,
    ) -> str:
        """Run a recursive RLM sub-call with its own REPL and loop.

        At leaf depth (depth >= max_depth), falls back to a plain LLM
        completion with no REPL.
        """
        from rich.console import Console
        _console = Console()

        # Leaf node: plain LLM call, no REPL
        if depth >= self._max_depth:
            _console.print(f"[dim]rlm_query d{depth}: leaf node, falling back to llm_query[/]")
            return self._sub_llm.completion(prompt)

        _console.print(f"[bold magenta]>>> rlm_query d{depth}: spawning child loop[/]")

        # --- Build child registry (inherit non-builtin tools) ---
        child_registry = ToolRegistry()
        for t in parent_run_registry.all_tools():
            if t.name not in self._BUILTIN_NAMES:
                child_registry.register(t)

        # --- Create child REPL ---
        repl = AgentREPL(registry=child_registry, context=parent_context)

        # Inherit OS access from parent (for video toolkit)
        for toolkit in self._registry.toolkits:
            if hasattr(toolkit, "get_os_access"):
                os_access = toolkit.get_os_access()
                if os_access is not None:
                    repl.set_os_access(os_access)

        # --- Child builtins ---
        def _child_llm_query(p: str) -> str:
            response = self._sub_llm.completion(p)
            repl.record_llm_query(p, response)
            usage = self._sub_llm.last_usage
            if usage:
                cost = self._sub_llm.last_cost_usd or 0.0
                self._budget.record(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cost_usd=cost,
                    model=_model_label(self.sub_model),
                )
            return response

        def _child_llm_query_batched(prompts: list[str]) -> list[str]:
            responses = self._sub_llm.completion_batched(prompts)
            for p, r in zip(prompts, responses):
                repl.record_llm_query(p, r)
            return responses

        def _child_rlm_query(p: str) -> str:
            return self._subcall(
                p,
                depth=depth + 1,
                parent_run_registry=child_registry,
                parent_context=parent_context,
            )

        def _child_rlm_query_batched(prompts: list[str]) -> list[str]:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(
                        self._subcall,
                        p,
                        depth=depth + 1,
                        parent_run_registry=child_registry,
                        parent_context=parent_context,
                    )
                    for p in prompts
                ]
                return [f.result() for f in futures]

        def _child_get_state() -> dict[str, Any]:
            return {
                "depth": depth,
                "max_depth": self._max_depth,
                "tools": [t.name for t in child_registry.all_tools()],
                "budget": self._budget.summary(),
            }

        child_registry.register(make_context_tool(lambda: repl.context))
        child_registry.register(make_llm_query_tool(_child_llm_query))
        child_registry.register(make_llm_query_batched_tool(_child_llm_query_batched))
        child_registry.register(make_done_tool(repl.mark_done))
        child_registry.register(make_get_state_tool(_child_get_state))
        child_registry.register(make_rlm_query_tool(_child_rlm_query))
        child_registry.register(make_rlm_query_batched_tool(_child_rlm_query_batched))

        # --- System prompt ---
        system_prompt = build_system_prompt(registry=child_registry, max_depth=self._max_depth)

        # --- Child orchestrator (uses sub_model) ---
        child_orchestrator = LLMClient(
            model=self.sub_model,
            fallback_model=self.fallback_model,
            name=f"child_rlm_d{depth}",
        )

        # --- Budget: remaining headroom from parent ---
        remaining_budget = None
        if self._budget.max_budget_usd is not None:
            remaining_budget = max(0.0, self._budget.max_budget_usd - self._budget.total_cost_usd)

        remaining_timeout = None
        if self._budget.max_timeout_s is not None:
            remaining_timeout = max(0.0, self._budget.max_timeout_s - self._budget.elapsed_s)

        child_budget = BudgetTracker(
            max_budget_usd=remaining_budget,
            max_timeout_s=remaining_timeout,
        )

        # Cap child iterations: children should find evidence quickly,
        # not spin for 20 iterations hallucinating content.
        child_max_iters = min(self.max_iterations, 10)

        child_config = LoopConfig(
            max_iterations=child_max_iters,
            max_budget_usd=remaining_budget,
            max_timeout_s=remaining_timeout,
            compaction_threshold=self.compaction_threshold,
        )

        # --- Run child loop (no critic, no schema) ---
        result = run_loop(
            orchestrator=child_orchestrator,
            repl=repl,
            system_prompt=system_prompt,
            question=prompt,
            config=child_config,
            budget=child_budget,
            tracer=self._tracer,
            critic=None,
            answer_schema=None,
        )

        # --- Merge child costs into parent budget ---
        self._budget.record(
            input_tokens=child_budget.total_input_tokens,
            output_tokens=child_budget.total_output_tokens,
            cost_usd=child_budget.total_cost_usd,
            model=f"child_rlm_d{depth}",
        )

        _console.print(f"[bold magenta]<<< rlm_query d{depth}: child done ({result.iterations_used} iters)[/]")

        # --- Extract answer string ---
        raw = result.raw_answer
        if isinstance(raw, dict):
            return str(raw.get("summary") or raw.get("answer") or raw)
        return str(raw)

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

    def _persist_trace(
        self,
        question: str,
        loop_result: LoopResult,
        evidence: list[Evidence],
    ) -> None:
        """Write trace.json alongside clips/frames in the workspace."""
        import json

        workspace = None
        for toolkit in self._registry.toolkits:
            if hasattr(toolkit, "_workspace") and toolkit._workspace is not None:
                workspace = toolkit._workspace
                break

        if workspace is None:
            return

        model_name = _model_label(self.model)
        sub_model_name = _model_label(self.sub_model)
        vision_label = _model_label(self.vision_model) if self.vision_model else sub_model_name

        raw = loop_result.raw_answer
        trace = {
            "run_id": workspace.run_id,
            "question": question,
            "model": model_name,
            "sub_model": sub_model_name,
            "vision_model": vision_label,
            "answer": str(raw),
            "answer_data": raw if isinstance(raw, dict) else None,
            "answer_schema": getattr(self, "_current_schema", None),
            "iterations": loop_result.iterations_used,
            "wall_time_s": round(loop_result.wall_time_s, 2),
            "cost": self._budget.summary(),
            "evidence_count": len(evidence),
            "events": self._tracer.dump_events(),
            "messages": loop_result.messages,
        }

        trace_path = workspace.run_dir / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, default=str), encoding="utf-8")

        workspace.record_trace_events(self._tracer.dump_events())

    def _has_video_toolkit(self) -> bool:
        """Check if a VideoToolkit is already registered."""
        try:
            from .tools.video import VideoToolkit
            return any(isinstance(tk, VideoToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False

    def _has_document_toolkit(self) -> bool:
        """Check if a DocumentToolkit is already registered."""
        try:
            from .tools.document import DocumentToolkit
            return any(isinstance(tk, DocumentToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False

    def _has_image_toolkit(self) -> bool:
        """Check if an ImageToolkit is already registered."""
        try:
            from .tools.image import ImageToolkit
            return any(isinstance(tk, ImageToolkit) for tk in self._registry.toolkits)
        except ImportError:
            return False
