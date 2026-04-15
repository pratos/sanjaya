"""Test batch captioning and querying via MoondreamVisionClient.

Creates synthetic test images, then exercises:
1. _resolve_model fix     (moondream specs survive resolution)
2. caption_frames_batch   (Modal: /batch/caption, Cloud: ThreadPool fallback)
3. query_frames            (Modal: /batch/query,   Cloud: ThreadPool fallback)
4. query_batch             (Modal: /batch/query,   Cloud: sequential fallback)
5. LLMClient vision paths  (Moondream + OpenRouter/OpenAI vision models)

Usage:
    uv run python scripts/test_batch_endpoints.py              # Modal (default)
    uv run python scripts/test_batch_endpoints.py --cloud      # Moondream Cloud
    uv run python scripts/test_batch_endpoints.py --openrouter # OpenRouter vision
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")

# Load .env
_dotenv = Path(__file__).resolve().parent.parent / ".env"
if _dotenv.exists():
    for _line in _dotenv.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            _k, _v = _k.strip(), _v.strip().strip("'\"")
            # Override empty shell env vars too
            if _v and not os.environ.get(_k):
                os.environ[_k] = _v

MODAL_BASE_URL = "https://prthamesh-sarang--sanjaya-moondream-server.modal.run/v1"


def make_test_images(n: int = 4) -> list[str]:
    """Create N solid-color JPEG test images in a temp dir."""
    from PIL import Image

    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "orange"]
    tmpdir = Path(tempfile.mkdtemp(prefix="sanjaya_test_"))
    paths = []
    for i in range(n):
        img = Image.new("RGB", (128, 128), colors[i % len(colors)])
        p = tmpdir / f"frame_{i:03d}.jpg"
        img.save(p, "JPEG")
        paths.append(str(p))
    return paths


# ── Unit tests ──────────────────────────────────────────────


def test_resolve_model() -> None:
    """Verify moondream specs pass through _resolve_model unchanged."""
    from sanjaya.agent import _resolve_model

    for spec in ("moondream:moondream3-preview", "moondream-station:moondream3-preview"):
        result = _resolve_model(spec, None)
        assert result == spec, f"Expected {spec!r}, got {result!r}"

    # Non-moondream specs should pass through as-is too
    spec_or = "openrouter:openai/gpt-4.1-mini"
    result_or = _resolve_model(spec_or, None)
    assert result_or == spec_or, f"Expected {spec_or!r}, got {result_or!r}"

    print("[PASS] _resolve_model preserves moondream + openrouter specs")


# ── Moondream tests (Modal or Cloud) ────────────────────────


def _make_moondream_client(use_modal: bool):
    from sanjaya.llm.moondream import MoondreamVisionClient

    if use_modal:
        auth_token = os.environ.get("SANJAYA_MODAL_AUTH_TOKEN", "")
        return MoondreamVisionClient(base_url=MODAL_BASE_URL, auth_token=auth_token)
    else:
        return MoondreamVisionClient()


def test_caption_batch(paths: list[str], use_modal: bool) -> None:
    client = _make_moondream_client(use_modal)
    label = "Modal" if use_modal else "Cloud"
    print(f"\n[TEST] caption_frames_batch ({label}) — {len(paths)} images")

    t0 = time.time()
    captions = client.caption_frames_batch(paths, length="short")
    elapsed = time.time() - t0

    assert len(captions) == len(paths), f"Expected {len(paths)} captions, got {len(captions)}"
    for i, cap in enumerate(captions):
        print(f"  [{i}] {cap[:80]}")
    print(f"  [{elapsed:.2f}s] tokens: in={client.total_input_tokens} out={client.total_output_tokens}")
    print(f"[PASS] caption_frames_batch ({label})")


def test_query_frames(paths: list[str], use_modal: bool) -> None:
    client = _make_moondream_client(use_modal)
    label = "Modal" if use_modal else "Cloud"
    print(f"\n[TEST] query_frames ({label}) — {len(paths)} images")

    t0 = time.time()
    result = client.query_frames("What color is this image?", paths)
    elapsed = time.time() - t0

    print(f"  {result[:200]}")
    print(f"  [{elapsed:.2f}s] tokens: in={client.total_input_tokens} out={client.total_output_tokens}")
    print(f"[PASS] query_frames ({label})")


def test_query_batch(paths: list[str], use_modal: bool) -> None:
    client = _make_moondream_client(use_modal)
    label = "Modal" if use_modal else "Cloud"
    items = [
        {"frame_paths": [paths[0], paths[1]], "question": "What color is this?"},
        {"frame_paths": [paths[2], paths[3]], "question": "Describe this image."},
    ]
    print(f"\n[TEST] query_batch ({label}) — {len(items)} groups")

    t0 = time.time()
    results = client.query_batch(items)
    elapsed = time.time() - t0

    assert len(results) == len(items), f"Expected {len(items)} results, got {len(results)}"
    for i, r in enumerate(results):
        print(f"  group[{i}]: {r[:120]}")
    print(f"  [{elapsed:.2f}s] tokens: in={client.total_input_tokens} out={client.total_output_tokens}")
    print(f"[PASS] query_batch ({label})")


# ── LLMClient integration test (OpenRouter/OpenAI vision) ───


def test_llm_client_vision(paths: list[str]) -> None:
    """Test vision_completion + vision_completion_batched via LLMClient.

    Uses the default sub_model + vision_model from Agent defaults.
    This exercises the pydantic-ai path (non-Moondream).
    """
    from sanjaya.llm.client import LLMClient

    # Use OpenRouter gpt-4.1-mini as vision model (same as Agent default sub_model)
    vision_model = "openrouter:openai/gpt-4.1-mini"
    client = LLMClient(model=vision_model, vision_model=vision_model, name="test")

    print(f"\n[TEST] LLMClient.vision_completion (OpenRouter) — 1 image")
    t0 = time.time()
    result = client.vision_completion(prompt="What color is this image?", frame_paths=[paths[0]])
    elapsed = time.time() - t0
    print(f"  {result[:120]}")
    print(f"  [{elapsed:.2f}s] usage={client.last_usage}")
    print("[PASS] LLMClient.vision_completion (OpenRouter)")

    print(f"\n[TEST] LLMClient.vision_completion_batched (OpenRouter) — 2 queries")
    queries = [
        {"prompt": "What color is this?", "frame_paths": [paths[0]], "clip_paths": None},
        {"prompt": "Describe this image.", "frame_paths": [paths[1]], "clip_paths": None},
    ]
    t0 = time.time()
    results = client.vision_completion_batched(queries)
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        print(f"  [{i}] {r[:120]}")
    print(f"  [{elapsed:.2f}s] usage={client.last_usage}")
    print("[PASS] LLMClient.vision_completion_batched (OpenRouter)")


def test_llm_client_moondream_vision(paths: list[str], use_modal: bool) -> None:
    """Test LLMClient with Moondream as vision_model."""
    from sanjaya.llm.client import LLMClient

    label = "Modal" if use_modal else "Cloud"

    if use_modal:
        os.environ["MOONDREAM_BASE_URL"] = MODAL_BASE_URL
        os.environ["MOONDREAM_AUTH_TOKEN"] = os.environ.get("SANJAYA_MODAL_AUTH_TOKEN", "")

    client = LLMClient(
        model="openrouter:openai/gpt-4.1-mini",
        vision_model="moondream:moondream3-preview",
        name="test",
    )

    print(f"\n[TEST] LLMClient vision_completion_batched (Moondream {label}) — 2 queries")
    queries = [
        {"prompt": "What color is this?", "frame_paths": [paths[0]], "clip_paths": None},
        {"prompt": "Describe this image.", "frame_paths": [paths[1]], "clip_paths": None},
    ]
    t0 = time.time()
    results = client.vision_completion_batched(queries)
    elapsed = time.time() - t0
    for i, r in enumerate(results):
        print(f"  [{i}] {r[:120]}")
    print(f"  [{elapsed:.2f}s]")
    print(f"[PASS] LLMClient vision_completion_batched (Moondream {label})")

    if use_modal:
        os.environ.pop("MOONDREAM_BASE_URL", None)
        os.environ.pop("MOONDREAM_AUTH_TOKEN", None)


# ── Main ────────────────────────────────────────────────────


def main() -> None:
    use_modal = "--cloud" not in sys.argv
    run_openrouter = "--openrouter" in sys.argv or "--all" in sys.argv

    target = "Modal" if use_modal else "Moondream Cloud"
    print("=" * 60)
    print(f"Batch endpoint tests — target: {target}")
    print("=" * 60)

    test_resolve_model()

    paths = make_test_images(4)
    print(f"\nCreated {len(paths)} test images in {Path(paths[0]).parent}")

    # Moondream direct client tests
    test_caption_batch(paths, use_modal)
    test_query_frames(paths, use_modal)
    test_query_batch(paths, use_modal)

    # LLMClient integration with Moondream
    test_llm_client_moondream_vision(paths, use_modal)

    # OpenRouter vision (optional — costs real tokens)
    if run_openrouter:
        test_llm_client_vision(paths)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
