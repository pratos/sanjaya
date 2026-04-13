"""Workspace manager for image analysis artifacts (crops, normalized images)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ImageWorkspace:
    """Run-local workspace for image crops and normalized copies."""

    def __init__(self, base_dir: str = "./sanjaya_artifacts"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"image_{timestamp}"
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        self.crops_dir = self.run_dir / "crops"
        self.normalized_dir = self.run_dir / "normalized"

        self._manifest: dict[str, Any] = {
            "run_id": self.run_id,
            "images": {},
            "crops": {},
            "trace_events": [],
        }

        self.ensure()

    def ensure(self) -> None:
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)

    def crop_path(self, image_id: str, crop_index: int) -> Path:
        """Deterministic path for a cropped region."""
        return self.crops_dir / f"{image_id}_crop_{crop_index:03d}.jpg"

    def normalized_path(self, image_id: str) -> Path:
        """Path for the normalized (resized) version of an image."""
        return self.normalized_dir / f"{image_id}.jpg"

    def record_trace_events(self, trace_events: list[dict[str, Any]]) -> None:
        """Store trace events in the manifest."""
        self._manifest["trace_events"] = trace_events
        self._flush_manifest()

    def _flush_manifest(self) -> None:
        manifest_path = self.run_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, default=str),
            encoding="utf-8",
        )
