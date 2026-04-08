set shell := ["bash", "-cu"]

# Start API (port 8000) + UI (port 5100) together via Overmind.
dev:
	overmind start

# Start only the FastAPI backend (port 8000).
api:
	uv run --project api uvicorn sanjaya_api.main:app --port 8000 --reload

# Start only the Next.js UI (port 5100).
ui:
	cd ui && bun dev --port 5100

# Run VideoRLM on a local sample video (override args as needed).
video-qna \
  video="data/longvideobench/videos/7F9IrtSHmc0.mp4" \
  question="In a room with a wall tiger and a map on the wall, what is the man doing?" \
  subtitle_mode="none" \
  format="markdown" \
  subtitle_api_model="whisper-1":
	uv run python -u scripts/run_video_qna.py \
	  --video "{{video}}" \
	  --question "{{question}}" \
	  --subtitle-mode "{{subtitle_mode}}" \
	  --subtitle-api-model "{{subtitle_api_model}}" \
	  --format "{{format}}"

# Generate subtitle sidecars with parakeet-mlx (pass video paths, or --all-youtube).
subtitles *args:
	uv run --with parakeet-mlx python scripts/generate_subtitles.py {{ if args == "" { "--all-youtube" } else { args } }}

# Run demo prompts against test videos (pass prompt IDs to run specific ones).
demo *ids:
	uv run python scripts/run_demo_prompts.py {{ if ids == "" { "" } else { "--prompt " + ids } }}

# Inspect latest persisted trace (or pass manifest/run_id).
video-trace manifest="" run_id="":
	if [[ -n "{{manifest}}" ]]; then uv run python scripts/inspect_video_trace.py --manifest "{{manifest}}"; elif [[ -n "{{run_id}}" ]]; then uv run python scripts/inspect_video_trace.py --run-id "{{run_id}}"; else uv run python scripts/inspect_video_trace.py; fi
