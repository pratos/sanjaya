"""ToolRegistry: manages tool registration and generates prompt documentation."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from .base import Tool, Toolkit


class ToolRegistry:
    """Manages tool registration and generates prompt documentation."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._toolkits: list[Toolkit] = []

    def register(self, tool: Tool) -> None:
        """Register a single tool."""
        self._tools[tool.name] = tool

    def register_toolkit(self, toolkit: Toolkit) -> None:
        """Register all tools from a toolkit."""
        self._toolkits.append(toolkit)
        for t in toolkit.tools():
            self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def all_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    @property
    def toolkits(self) -> list[Toolkit]:
        """Return all registered toolkits."""
        return list(self._toolkits)

    def build_external_functions(self) -> dict[str, Callable[..., Any]]:
        """Build the external_functions dict for MontyRepl.feed_run()."""
        return {name: tool.fn for name, tool in self._tools.items()}

    def generate_tool_docs(self) -> str:
        """Auto-generate the tool contract section of the system prompt."""
        if not self._tools:
            return ""

        lines = ["Available tools in the REPL:"]
        for tool in self._tools.values():
            sig_parts: list[str] = []
            for param in tool.parameters.values():
                part = f"{param.name}: {param.type_hint}"
                if param.default is not inspect.Parameter.empty:
                    part += f" = {param.default!r}"
                sig_parts.append(part)
            signature = ", ".join(sig_parts)
            lines.append(f"- `{tool.name}({signature}) -> {tool.return_type}`")
            if tool.description:
                lines.append(f"  {tool.description}")
            # Add parameter descriptions if any
            for param in tool.parameters.values():
                if param.description:
                    default_str = ""
                    if param.default is not inspect.Parameter.empty:
                        default_str = f" (default: {param.default!r})"
                    lines.append(f"    - {param.name}: {param.description}{default_str}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        tool_names = list(self._tools.keys())
        return f"ToolRegistry(tools={tool_names})"
