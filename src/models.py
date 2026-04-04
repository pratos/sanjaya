"""Pydantic models for LLM responses."""

from pydantic import BaseModel, Field


class CodeResponse(BaseModel):
    """Structured response from orchestrator LLM."""

    reasoning: str = Field(description="Your step-by-step thinking about what to do next")
    code_blocks: list[str] = Field(
        default_factory=list, description="Python code blocks to execute in the REPL. Empty if no code to run."
    )
    is_done: bool = Field(default=False, description="Set to true when you have the final answer")
    final_answer: str | None = Field(
        default=None, description="The final answer to the user's query (only when is_done=True)"
    )
