"""Budget tracking for cost and token limits."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetTracker:
    """Tracks cumulative cost and tokens across a run."""

    max_budget_usd: float | None = None
    max_timeout_s: float | None = None

    _total_cost_usd: float = field(default=0.0, init=False)
    _total_input_tokens: int = field(default=0, init=False)
    _total_output_tokens: int = field(default=0, init=False)
    _start_time: float = field(default_factory=time.time, init=False)
    _calls: list[dict] = field(default_factory=list, init=False)

    def record(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        model: str | None = None,
    ) -> None:
        """Record usage from an LLM call."""
        self._total_cost_usd += cost_usd
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "model": model,
        })

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def elapsed_s(self) -> float:
        return time.time() - self._start_time

    @property
    def budget_exceeded(self) -> bool:
        if self.max_budget_usd is None:
            return False
        return self._total_cost_usd >= self.max_budget_usd

    @property
    def timeout_exceeded(self) -> bool:
        if self.max_timeout_s is None:
            return False
        return self.elapsed_s >= self.max_timeout_s

    def should_stop(self) -> bool:
        """True if any limit is exceeded."""
        return self.budget_exceeded or self.timeout_exceeded

    def summary(self) -> dict:
        return {
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "elapsed_s": round(self.elapsed_s, 2),
            "calls": len(self._calls),
        }
