# Fix Token/Cost Accounting: Budget Underreporting + Logfire Cost Inflation

## Overview

Two bugs cause token/cost numbers to diverge between local UI, Logfire, and OpenRouter billing:

1. **Budget tracker misses batched calls** — `_run_batched()` discards per-call usage, so 124/156 API calls (~50% of tokens, ~34% of cost) are lost from the budget.
2. **Logfire cost is double-counted** — manual tracer spans record tokens/cost that duplicate pydantic-ai auto-instrumentation, inflating Logfire cost by ~2.5x.

Logfire token counts are actually correct (they match OpenRouter). The budget tracker is the one that's wrong.

## Current State Analysis

**Run 20260408-185606** ("Write me a 200 words summary of this match"):

| Source | Calls | Input Tokens | Output Tokens | Cost |
|---|---|---|---|---|
| **OpenRouter (truth)** | **156** | **583,309** | **25,484** | **$0.373** |
| Logfire | ~365 spans | 583,310 | 25,480 | $0.941 |
| Budget tracker | 32 | 306,658 | 14,625 | $0.245 |

### Root cause 1: `_run_batched()` loses usage

`LLMClient._run_batched()` (`client.py:495-519`) runs concurrent `Agent.run()` calls but discards the result objects:

```python
response_text, _ = r  # result object with usage is thrown away
```

This affects `vision_completion_batched()` which is called by `caption_frames` for per-frame captioning. In this run: 126 per-frame calls (~1,520 tokens each) were made to gpt-4.1-mini but never recorded in the budget.

### Root cause 2: Manual span token recording duplicates auto-instrumentation

`logfire.instrument_pydantic_ai()` auto-instruments every `Agent.run()` call with token/cost attributes on OpenTelemetry spans. The manual tracer spans (`sanjaya.root_llm_call`, `sanjaya.sub_llm_call.*`) also call `record_usage()` and `record_llm_cost()` with the same data. Logfire aggregates both, inflating cost.

- Logfire tokens (583K) match OpenRouter — auto-instrumentation is correct
- Logfire cost ($0.94) is ~2.5x OpenRouter ($0.37) — manual spans add duplicate cost

### Key Discoveries:
- gpt-5.3-codex has prompt caching: 39,424 cached tokens saved $0.062 (OpenRouter `cost_cache`)
- genai_prices has no entry for `qwen/qwen3-30b-a3b-thinking-2507` ($0.08/$0.40 on OpenRouter)
- genai_prices has no `openrouter` provider entry for `gpt-5.3-codex` (only `openai` direct)
- Per-frame caption calls are ~1,520 tokens each (126 calls = ~191K tokens lost from budget)

## Desired End State

After fix:
- Budget tracker matches OpenRouter within 5% (accounting for cache discounts)
- Logfire cost matches OpenRouter within 20% (auto-instrumentation pricing may differ slightly)
- All 17 tests in `tests/test_token_cost_accounting.py` pass
- Tests in `TestBudgetVsOpenRouter` should be INVERTED (budget matches OR instead of undercounting)

## What We're NOT Doing

- Not changing the event buffer (SSE events for local UI)
- Not removing `logfire.instrument_pydantic_ai()` (it provides correct token tracking)
- Not changing compaction or prompt construction
- Not fixing genai_prices missing model entries (upstream dependency)

## Phase 1: Fix `_run_batched()` Usage Accumulation

### Changes Required:

#### 1. `src/sanjaya/llm/client.py` — `_run_batched()` (lines 495-519)

Capture usage from each result and accumulate into `last_usage`. The results contain pydantic-ai `RunResult` objects with `.usage()`.

```python
def _run_batched(self, calls: list[dict[str, Any]]) -> list[str]:
    """Run multiple calls concurrently, accumulating usage."""
    from opentelemetry import context as otel_context
    parent_ctx = otel_context.get_current()

    async def _gather():
        token = otel_context.attach(parent_ctx)
        try:
            tasks = [self._run_agent_async(c["model"], c["payload"]) for c in calls]
            return await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            otel_context.detach(token)

    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(_gather(), loop)
    raw_results = future.result()

    responses: list[str] = []
    total_input = 0
    total_output = 0
    total_cost = 0.0
    for r in raw_results:
        if isinstance(r, Exception):
            responses.append(f"Error: {r}")
        else:
            response_text, result = r
            responses.append(response_text)
            # Accumulate usage from each call
            snap = self._capture_usage(result)
            total_input += snap.input_tokens
            total_output += snap.output_tokens
            cost = self._extract_cost_usd(result)
            if cost:
                total_cost += cost

    # Set accumulated totals as last_usage
    self.last_usage = UsageSnapshot(
        input_tokens=total_input,
        output_tokens=total_output,
        total_tokens=total_input + total_output,
    )
    self.last_call_metadata = CallMetadata(
        requested_model=str(calls[0]["model"]) if calls else "unknown",
        model_used="batched",
        provider="batched",
        duration_seconds=0,
        cost_usd=total_cost if total_cost > 0 else None,
    )
    return responses
```

#### 2. `src/sanjaya/llm/client.py` — `_run_agent_async()` (lines 364-368)

Add the agent `name` parameter (currently missing vs `_run_agent`), so pydantic-ai auto-instrumentation shows meaningful span names for batched calls too.

```python
async def _run_agent_async(self, model: ModelSpec, payload: Any) -> tuple[str, Any]:
    model_label = model if isinstance(model, str) else getattr(model, "model_name", "unknown")
    agent = Agent(model=model, output_type=str, retries=1, defer_model_check=True, name=f"sanjaya:{self.name}:{model_label}")
    result = await agent.run(payload)
    response = result.output if hasattr(result, "output") else str(result)
    return str(response), result
```

### Success Criteria:

#### Automated Verification:
- [ ] `TestBudgetVsOpenRouter` tests should be updated to assert budget MATCHES OpenRouter (not undercounts)
- [ ] No import errors: `uv run python3 -c "import sanjaya"`

#### Manual Verification:
- [ ] Run a prompt with caption_frames, check budget totals match OpenRouter activity CSV
- [ ] Local UI shows correct elevated token/cost (was $0.24, should be ~$0.37)

---

## Phase 2: Remove Manual Span Token/Cost Recording

### Changes Required:

#### 1. `src/sanjaya/agent.py` — sub-LLM query closure (lines 252-260)
Remove `record_usage()` and `record_llm_cost()`:

```python
# DELETE these lines:
                    llm_trace.record_usage(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                    )
                    llm_trace.record_llm_cost(
                        input_tokens=usage.input_tokens,
                        output_tokens=usage.output_tokens,
                        model_name=sub_model_name,
                    )
```

#### 2. `src/sanjaya/agent.py` — completion span (lines 432-443)
Replace `record_usage()` + `operation.cost` with custom-prefixed attributes:

```python
# REPLACE lines 432-443 with:
            comp_trace.record(
                sanjaya_cost_usd=cost_usd,
                sanjaya_input_tokens=self._budget.total_input_tokens,
                sanjaya_output_tokens=self._budget.total_output_tokens,
            )
```

#### 3. `src/sanjaya/core/loop.py` — orchestrator call (lines 129-135)
Remove `record_usage()` and `record_llm_cost()`:

```python
# DELETE these lines:
                if orch_trace:
                    orch_trace.record_usage(...)
                    orch_trace.record_llm_cost(...)
```

#### 4. `src/sanjaya/tools/video/vision.py` — vision spans (lines 98, 185, 234-237)
Remove `record_usage()` from vision_query and caption_frames spans.

### Success Criteria:

#### Automated Verification:
- [ ] `TestLogfireInflation::test_logfire_cost_inflated_vs_openrouter` should be updated to assert cost is NOT inflated

#### Manual Verification:
- [ ] Check Logfire after a real run — cost on `sanjaya.completion` should be close to OpenRouter
- [ ] pydantic-ai auto-instrumented child spans still show per-call tokens in Logfire

## Testing Strategy

### Automated:
- `tests/test_token_cost_accounting.py` — 17 tests covering all scenarios
- After Phase 1: update `TestBudgetVsOpenRouter` assertions to expect matching
- After Phase 2: update `TestLogfireInflation` assertions to expect no inflation

### Manual:
1. Run a video prompt, download OpenRouter CSV
2. Compare: budget ≈ OpenRouter ≈ Logfire tokens (within 5%)
3. Compare: budget cost ≈ OpenRouter cost (within 10%, cache discounts cause small variance)

## References

- Test: `tests/test_token_cost_accounting.py`
- Trace: `sanjaya_artifacts/20260408-185606/trace.json`
- OpenRouter CSV: `~/Downloads/openrouter_activity_2026-04-08.csv`
- OpenRouter pricing: gpt-5.3-codex $1.75/$14, gpt-4.1-mini $0.40/$1.60, qwen3-30b $0.08/$0.40
