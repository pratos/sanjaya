"""Core engine — REPL, loop, compaction, budget, prompts."""

from .blocks import ExecutionResult, extract_code_blocks, extract_final_answer, format_execution_feedback
from .budget import BudgetTracker
from .loop import LoopConfig, LoopResult, run_loop
from .repl import AgentREPL

__all__ = [
    "AgentREPL",
    "BudgetTracker",
    "ExecutionResult",
    "LoopConfig",
    "LoopResult",
    "extract_code_blocks",
    "extract_final_answer",
    "format_execution_feedback",
    "run_loop",
]
