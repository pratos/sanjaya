"""Report writer implementations — save_note, save_qmd, save_data."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any


def save_note(content: str, filename: str, output_dir: str) -> str:
    """Save a markdown/text note to the reports directory."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def save_qmd(content: str, filename: str, output_dir: str) -> str:
    """Save a Quarto markdown document to the reports directory."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return str(path.resolve())


def save_data(data: Any, filename: str, output_dir: str) -> str:
    """Save structured data (JSON, CSV) to the reports directory."""
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)

    if filename.endswith(".csv"):
        if isinstance(data, list) and all(isinstance(row, (list, tuple)) for row in data):
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(data)
            path.write_text(output.getvalue(), encoding="utf-8")
        else:
            raise ValueError("CSV data must be a list of lists/tuples")
    else:
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    return str(path.resolve())
