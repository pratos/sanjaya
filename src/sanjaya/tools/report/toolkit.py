"""ReportToolkit — save analysis outputs and build cross-run memory."""

from __future__ import annotations

import time
from typing import Any

from ..base import Tool, Toolkit, ToolParam
from .writers import save_data as _save_data
from .writers import save_note as _save_note
from .writers import save_qmd as _save_qmd


class ReportToolkit(Toolkit):
    """Save analysis outputs and build cross-run memory."""

    def __init__(
        self,
        output_dir: str = "./sanjaya_reports",
        retrieval: Any | None = None,
    ):
        self.output_dir = output_dir
        self.retrieval = retrieval
        self._run_id: str | None = None

    def setup(self, context: dict[str, Any]) -> None:
        self._run_id = context.get("run_id")

    def teardown(self) -> None:
        pass

    def tools(self) -> list[Tool]:
        return [
            self._make_save_note_tool(),
            self._make_save_qmd_tool(),
            self._make_save_data_tool(),
        ]

    def _make_save_note_tool(self) -> Tool:
        toolkit = self

        def save_note(content: str, filename: str) -> str:
            """Save a markdown/text note to the reports directory."""
            path = _save_note(content, filename, toolkit.output_dir)
            if toolkit.retrieval:
                toolkit.retrieval.index(
                    documents=[content],
                    metadata=[{
                        "filename": filename,
                        "run_id": toolkit._run_id,
                        "timestamp": time.time(),
                    }],
                    collection="reports",
                )
            return path

        return Tool(
            name="save_note",
            description="Save a markdown/text note to the reports directory. Auto-indexed for cross-run retrieval.",
            fn=save_note,
            parameters={
                "content": ToolParam(name="content", type_hint="str", description="The text content to save."),
                "filename": ToolParam(name="filename", type_hint="str", description="Filename (e.g., 'analysis.md')."),
            },
            return_type="str",
        )

    def _make_save_qmd_tool(self) -> Tool:
        toolkit = self

        def save_qmd(content: str, filename: str) -> str:
            """Save a Quarto markdown document to the reports directory."""
            path = _save_qmd(content, filename, toolkit.output_dir)
            if toolkit.retrieval:
                toolkit.retrieval.index(
                    documents=[content],
                    metadata=[{
                        "filename": filename,
                        "run_id": toolkit._run_id,
                        "timestamp": time.time(),
                    }],
                    collection="reports",
                )
            return path

        return Tool(
            name="save_qmd",
            description="Save a Quarto markdown document to the reports directory. Auto-indexed for cross-run retrieval.",
            fn=save_qmd,
            parameters={
                "content": ToolParam(name="content", type_hint="str", description="Full QMD content including frontmatter."),
                "filename": ToolParam(name="filename", type_hint="str", description="Filename (e.g., 'report.qmd')."),
            },
            return_type="str",
        )

    def _make_save_data_tool(self) -> Tool:
        toolkit = self

        def save_data(data: Any, filename: str) -> str:
            """Save structured data (JSON, CSV) to the reports directory."""
            return _save_data(data, filename, toolkit.output_dir)

        return Tool(
            name="save_data",
            description="Save structured data (JSON, CSV) to the reports directory.",
            fn=save_data,
            parameters={
                "data": ToolParam(name="data", type_hint="Any", description="Dict/list for JSON, or list[list] for CSV."),
                "filename": ToolParam(name="filename", type_hint="str", description="Filename with extension (e.g., 'results.json')."),
            },
            return_type="str",
        )
