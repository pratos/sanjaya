# Sanjaya API — Backend

FastAPI bridge wrapping `VideoRLM_REPL` for the Sanjaya HUD dashboard. Runs orchestrations in background threads and streams trace events via SSE.

## Setup

```bash
cd api
uv sync
uv run uvicorn sanjaya_api.main:app --port 8000
```

Or from project root: `just dev` (starts both API and UI via Overmind).

## Environment Variables

Inherits from parent project `.env`:

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | Primary LLM provider |
| `OPENAI_API_KEY` | Optional | OpenAI provider |
| `ANTHROPIC_API_KEY` | Optional | Anthropic provider |
| `LOGFIRE_TOKEN` | Optional | Observability |

## API Endpoints

### `GET /health`
Health check.
```json
{"status": "ok"}
```

### `POST /runs`
Start a new VideoRLM orchestration run.

**Request:**
```json
{
  "video_path": "/path/to/video.mp4",
  "question": "What is happening in the video?",
  "subtitle_path": null,
  "subtitle_mode": "auto"
}
```

**Response:**
```json
{"run_id": "a1b2c3d4e5f6"}
```

### `GET /runs/{run_id}/events`
SSE stream of trace events for a run.

**Event types:** `run_start`, `transcription`, `root_response`, `code_instruction`, `code_execution`, `retrieval`, `clip`, `frames`, `vision`, `sub_llm`, `run_end`, `heartbeat`, `stream_end`, `stream_error`

**Event format:**
```
event: root_response
data: {"kind": "root_response", "timestamp": 1712234567.89, "payload": {...}}
```

## Architecture

- `OrchestratorService` manages background threads for `VideoRLM_REPL.completion()`
- SSE polling reads from `Tracer.events` (append-only list, GIL-safe)
- Heartbeat sent every 2s to keep connections alive
- Stream closes after `run_end` event + `stream_end` sentinel
