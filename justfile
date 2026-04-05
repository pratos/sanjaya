set shell := ["bash", "-cu"]

# Start API (port 8000) + UI (port 5100) together via Overmind.
dev:
	overmind start

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

# Inspect latest persisted trace (or pass manifest/run_id).
video-trace manifest="" run_id="":
	if [[ -n "{{manifest}}" ]]; then uv run python scripts/inspect_video_trace.py --manifest "{{manifest}}"; elif [[ -n "{{run_id}}" ]]; then uv run python scripts/inspect_video_trace.py --run-id "{{run_id}}"; else uv run python scripts/inspect_video_trace.py; fi
