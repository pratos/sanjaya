"""Background orchestrator that runs Agent in a thread."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sanjaya import Agent, Answer
from sanjaya.tools.video import VideoToolkit
from sanjaya.tracing import Tracer


@dataclass
class RunRecord:
    """In-memory record of a single orchestration run."""

    run_id: str
    status: str = "pending"  # pending | running | complete | error
    tracer: Tracer | None = None
    answer: Answer | None = None
    error: str | None = None
    thread: threading.Thread | None = None


_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class OrchestratorService:
    """Manages background Agent runs and exposes their tracers for SSE polling."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def start_run(
        self,
        video_path: str,
        question: str,
        subtitle_path: str | None = None,
        subtitle_mode: str = "none",
        subtitle_api_model: str = "gpt-4o-transcribe-diarize",
        max_iterations: int = 20,
    ) -> str:
        """Start a new orchestration run in a background thread, return run_id."""
        run_id = uuid4().hex[:12]
        record = RunRecord(run_id=run_id)

        with self._lock:
            self._runs[run_id] = record

        thread = threading.Thread(
            target=self._run_completion,
            args=(record, video_path, question, subtitle_path, subtitle_mode, max_iterations),
            daemon=True,
        )
        record.thread = thread
        thread.start()
        return run_id

    def get_run(self, run_id: str) -> RunRecord | None:
        """Get a run record by ID."""
        with self._lock:
            return self._runs.get(run_id)

    @staticmethod
    def _resolve_video_path(video_path: str) -> str:
        """Resolve a relative video path against the data/ directory."""
        p = Path(video_path)
        if not p.is_absolute():
            p = _DATA_DIR / p
        return str(p)

    def _run_completion(
        self,
        record: RunRecord,
        video_path: str,
        question: str,
        subtitle_path: str | None,
        subtitle_mode: str,
        max_iterations: int,
    ) -> None:
        """Execute Agent.ask() in a background thread."""
        record.status = "running"

        tracer = Tracer(track_events=True)
        record.tracer = tracer

        try:
            agent = Agent(max_iterations=max_iterations, tracing=True)
            agent._tracer = tracer
            agent.use(VideoToolkit(subtitle_mode=subtitle_mode))

            resolved_video = self._resolve_video_path(video_path)
            answer = agent.ask(
                question,
                video=resolved_video,
                subtitle=subtitle_path,
            )

            record.answer = answer
            record.status = "complete"
        except Exception as exc:
            record.error = str(exc)
            record.status = "error"
