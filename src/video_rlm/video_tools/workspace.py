"""Workspace manager for VideoRLM-generated artifacts."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from video_rlm.video_models import CandidateWindow, ClipArtifact


class ArtifactWorkspace:
    """Run-local workspace for clips, frames, and manifests."""

    def __init__(self, base_dir: str = "data/longvideobench/artifacts", run_id: str | None = None):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = run_id or timestamp
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / self.run_id
        self.clips_dir = self.run_dir / "clips"
        self.frames_dir = self.run_dir / "frames"
        self.manifest_path = self.run_dir / "manifest.json"

        self._manifest: dict = {
            "run_id": self.run_id,
            "clips": {},
            "candidate_windows": [],
            "trace_events": [],
        }

        self.ensure()

    def ensure(self) -> None:
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._flush_manifest()

    def clip_path(self, clip_id: str) -> Path:
        return self.clips_dir / f"{clip_id}.mp4"

    def frame_dir(self, clip_id: str) -> Path:
        return self.frames_dir / clip_id

    def record_windows(self, windows: list[CandidateWindow]) -> None:
        self._manifest["candidate_windows"] = [w.model_dump() for w in windows]
        self._flush_manifest()

    def record_clip(self, artifact: ClipArtifact, window_id: str | None = None) -> None:
        payload = artifact.model_dump()
        if window_id is not None:
            payload["window_id"] = window_id
        self._manifest["clips"][artifact.clip_id] = payload
        self._flush_manifest()

    def update_frames(self, clip_id: str, frame_paths: list[str]) -> None:
        clip = self._manifest["clips"].get(clip_id)
        if clip is None:
            return
        clip["frame_paths"] = frame_paths
        self._flush_manifest()

    def record_trace_events(self, trace_events: list[dict]) -> None:
        self._manifest["trace_events"] = trace_events
        self._flush_manifest()

    def get_manifest(self) -> dict:
        return self._manifest

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return self._manifest

    def _flush_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2), encoding="utf-8")
