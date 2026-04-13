"""Tool, ToolParam, Toolkit base class, and @tool decorator."""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from ..answer import Evidence


@dataclass
class ToolParam:
    """A single parameter of a tool function."""

    name: str
    type_hint: str
    default: Any = inspect.Parameter.empty
    description: str = ""


@dataclass
class Tool:
    """A callable tool available in the agent's REPL environment."""

    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, ToolParam] = field(default_factory=dict)
    return_type: str = "Any"


class Toolkit(ABC):
    """Bundle of related tools with shared state."""

    _prompt_config: Any = None  # Injected by Agent, type is PromptConfig

    @abstractmethod
    def tools(self) -> list[Tool]:
        """Return all tools this toolkit provides."""

    def setup(self, context: dict[str, Any]) -> None:
        """Called once before the RLM loop starts."""

    def teardown(self) -> None:
        """Called after the RLM loop ends."""

    def get_state(self) -> dict[str, Any]:
        """Return toolkit state for introspection."""
        return {}

    def build_evidence(self) -> list[Evidence]:
        """Convert toolkit artifacts into Evidence items."""
        return []

    def prompt_section(self) -> str | None:
        """Optional extra prompt text for the system prompt."""
        return None


def _parse_docstring_args(docstring: str) -> dict[str, str]:
    """Parse Args section from a Google-style docstring into {param: description}."""
    result: dict[str, str] = {}
    args_match = re.search(r"Args:\s*\n((?:\s+\S.*\n?)*)", docstring)
    if not args_match:
        return result

    args_block = args_match.group(1)
    current_param: str | None = None
    current_desc_lines: list[str] = []

    for line in args_block.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # New param line: "param_name: description" or "param_name (type): description"
        param_match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
        if param_match:
            if current_param:
                result[current_param] = " ".join(current_desc_lines).strip()
            current_param = param_match.group(1)
            current_desc_lines = [param_match.group(2)]
        elif current_param:
            current_desc_lines.append(stripped)

    if current_param:
        result[current_param] = " ".join(current_desc_lines).strip()

    return result


def _get_type_hint_str(annotation: Any) -> str:
    """Convert a type annotation to a readable string."""
    if annotation is inspect.Parameter.empty:
        return "Any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _extract_description(docstring: str) -> str:
    """Extract the description (everything before Args/Returns/Raises sections)."""
    lines: list[str] = []
    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.startswith(("Args:", "Returns:", "Raises:", "Example:", "Examples:")):
            break
        lines.append(stripped)
    return " ".join(line for line in lines if line).strip()


def tool(fn: Callable[..., Any]) -> Tool:
    """Decorator that converts a function into a Tool.

    Inspects the function's signature and docstring to produce
    a fully-described Tool object.
    """
    sig = inspect.signature(fn)
    docstring = inspect.getdoc(fn) or ""
    arg_docs = _parse_docstring_args(docstring)

    parameters: dict[str, ToolParam] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        parameters[param_name] = ToolParam(
            name=param_name,
            type_hint=_get_type_hint_str(param.annotation),
            default=param.default,
            description=arg_docs.get(param_name, ""),
        )

    return_annotation = sig.return_annotation
    return_type = _get_type_hint_str(return_annotation)

    return Tool(
        name=fn.__name__,
        description=_extract_description(docstring),
        fn=fn,
        parameters=parameters,
        return_type=return_type,
    )
