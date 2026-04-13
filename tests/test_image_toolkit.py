"""Tests for ImageToolkit — uses mocked LLM client and synthetic images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sanjaya.tools.image.media import crop_image, load_image, normalize_for_vision
from sanjaya.tools.image.toolkit import ImageToolkit, _parse_bbox_response

# ── Helpers ──────────────────────────────────────────────────


def _make_test_image(path: Path, width: int = 200, height: int = 150, color: str = "red") -> str:
    """Create a small test image with Pillow."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color)
    img.save(str(path), format="JPEG")
    return str(path)


def _make_png_with_alpha(path: Path, width: int = 100, height: int = 100) -> str:
    """Create a PNG with transparency."""
    from PIL import Image

    img = Image.new("RGBA", (width, height), (255, 0, 0, 128))
    img.save(str(path), format="PNG")
    return str(path)


class MockLLMClient:
    """Mock LLM client for testing vision_completion calls."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or ["Mock vision response"])
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []
        self.last_usage = None
        self.last_call_metadata = None
        self.last_cost_usd = None
        self.vision_model = "mock-vision-model"
        self.model = "mock-model"

    def vision_completion(self, *, prompt: str, frame_paths: list[str] | None = None, clip_paths: list[str] | None = None) -> str:
        self.calls.append({
            "prompt": prompt,
            "frame_paths": frame_paths,
            "clip_paths": clip_paths,
        })
        response = self._responses[min(self._call_index, len(self._responses) - 1)]
        self._call_index += 1
        return response


# ── Phase 1: media.py tests ─────────────────────────────────


def test_load_image_jpeg(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg", 300, 200)
    info = load_image(img_path)

    assert info.width == 300
    assert info.height == 200
    assert info.format == "JPEG"
    assert info.image_id == "photo"
    assert info.file_size_bytes > 0


def test_load_image_png(tmp_path: Path):
    from PIL import Image

    p = tmp_path / "chart.png"
    Image.new("RGB", (400, 300), "blue").save(str(p), format="PNG")
    info = load_image(str(p))

    assert info.width == 400
    assert info.height == 300
    assert info.format == "PNG"


def test_load_image_not_found():
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/image.jpg")


def test_load_image_unsupported_format(tmp_path: Path):
    p = tmp_path / "file.xyz"
    p.write_text("not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_image(str(p))


def test_normalize_for_vision_resize(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "big.jpg", 4000, 3000)
    data = normalize_for_vision(img_path, max_dim=1536)

    assert isinstance(data, bytes)
    assert len(data) < 200_000  # should be well under 200KB

    # Verify dimensions were reduced
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    assert max(img.size) <= 1536


def test_normalize_for_vision_rgba(tmp_path: Path):
    img_path = _make_png_with_alpha(tmp_path / "alpha.png")
    data = normalize_for_vision(img_path)

    import io

    from PIL import Image

    img = Image.open(io.BytesIO(data))
    assert img.mode == "RGB"  # alpha composited away


def test_crop_image(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "source.jpg", 400, 300)
    out_path = str(tmp_path / "cropped.jpg")

    result = crop_image(img_path, (50, 50, 200, 200), out_path)

    assert Path(result).exists()
    from PIL import Image

    cropped = Image.open(result)
    assert cropped.size == (150, 150)


def test_crop_image_clamps_bounds(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "source.jpg", 100, 100)
    out_path = str(tmp_path / "cropped.jpg")

    # bbox extends beyond image bounds — should clamp
    result = crop_image(img_path, (-10, -10, 200, 200), out_path)
    assert Path(result).exists()

    from PIL import Image

    cropped = Image.open(result)
    assert cropped.size == (100, 100)


# ── Phase 2: Core toolkit tests ─────────────────────────────


def test_setup_single_image(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "test.jpg")
    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit.setup({"image": [img_path], "question": "describe"})

    tools = {t.name for t in toolkit.tools()}
    assert "list_images" in tools
    assert "vision_query" in tools

    list_fn = next(t.fn for t in toolkit.tools() if t.name == "list_images")
    result = list_fn()
    assert len(result) == 1
    assert result[0]["image_id"] == "test"
    assert result[0]["width"] == 200


def test_setup_multiple_images(tmp_path: Path):
    paths = [
        _make_test_image(tmp_path / "a.jpg", 100, 100),
        _make_test_image(tmp_path / "b.jpg", 200, 200),
        _make_test_image(tmp_path / "a.jpg", 300, 300),  # duplicate name
    ]
    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit.setup({"image": paths, "question": "describe"})

    list_fn = next(t.fn for t in toolkit.tools() if t.name == "list_images")
    result = list_fn()
    assert len(result) == 3

    ids = {r["image_id"] for r in result}
    assert "a" in ids
    assert "b" in ids
    assert "a_1" in ids  # dedup


def test_get_image_info_valid(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg", 640, 480)
    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit.setup({"image": [img_path], "question": "test"})

    info_fn = next(t.fn for t in toolkit.tools() if t.name == "get_image_info")
    result = info_fn("photo")
    assert result["width"] == 640
    assert result["height"] == 480
    assert result["format"] == "JPEG"
    assert "error" not in result


def test_get_image_info_invalid(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg")
    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit.setup({"image": [img_path], "question": "test"})

    info_fn = next(t.fn for t in toolkit.tools() if t.name == "get_image_info")
    result = info_fn("nonexistent")
    assert "error" in result
    assert "list_images()" in result["error"]


def test_vision_query_single(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg")
    mock_llm = MockLLMClient(["A red square image"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": [img_path], "question": "describe"})

    query_fn = next(t.fn for t in toolkit.tools() if t.name == "vision_query")
    result = query_fn(prompt="What do you see?", image_id="photo")

    assert result == "A red square image"
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[0]["frame_paths"] == [str(Path(img_path).resolve())]
    assert "photo" in toolkit._queried_images


def test_vision_query_auto_single(tmp_path: Path):
    """When only one image is loaded, it should be used automatically."""
    img_path = _make_test_image(tmp_path / "only.jpg")
    mock_llm = MockLLMClient(["The only image"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": [img_path], "question": "test"})

    query_fn = next(t.fn for t in toolkit.tools() if t.name == "vision_query")
    result = query_fn(prompt="Describe this")
    assert result == "The only image"


def test_vision_query_multi(tmp_path: Path):
    paths = [
        _make_test_image(tmp_path / "a.jpg", 100, 100),
        _make_test_image(tmp_path / "b.jpg", 200, 200),
    ]
    mock_llm = MockLLMClient(["Both images analyzed"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": paths, "question": "compare"})

    query_fn = next(t.fn for t in toolkit.tools() if t.name == "vision_query")
    result = query_fn(prompt="Compare", image_ids=["a", "b"])

    assert result == "Both images analyzed"
    assert len(mock_llm.calls[0]["frame_paths"]) == 2
    assert toolkit._queried_images == {"a", "b"}


# ── Phase 3: Rich tools tests ───────────────────────────────


def test_crop_region(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg", 800, 600)
    bbox_json = json.dumps({"x": 100, "y": 100, "x2": 400, "y2": 400})
    mock_llm = MockLLMClient([bbox_json])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": [img_path], "question": "test"})

    crop_fn = next(t.fn for t in toolkit.tools() if t.name == "crop_region")
    result = crop_fn(image_id="photo", region_description="the center area")

    assert "crop_path" in result
    assert Path(result["crop_path"]).exists()
    assert result["bbox"] == {"x": 100, "y": 100, "x2": 400, "y2": 400}
    assert result["region"] == "the center area"


def test_crop_region_bad_response(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg", 800, 600)
    # Both attempts return garbage
    mock_llm = MockLLMClient(["I can't determine coordinates", "Still no coordinates"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": [img_path], "question": "test"})

    crop_fn = next(t.fn for t in toolkit.tools() if t.name == "crop_region")
    result = crop_fn(image_id="photo", region_description="something")

    assert "error" in result


def test_search_images_lazy_caption(tmp_path: Path):
    paths = [
        _make_test_image(tmp_path / "sunset.jpg", 100, 100, "orange"),
        _make_test_image(tmp_path / "ocean.jpg", 100, 100, "blue"),
    ]
    mock_llm = MockLLMClient([
        "A beautiful sunset over mountains with orange sky",
        "A calm ocean with blue water and waves",
    ])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": paths, "question": "test"})

    search_fn = next(t.fn for t in toolkit.tools() if t.name == "search_images")

    # First call should trigger captioning (2 vision calls)
    results = search_fn(query="sunset")
    assert len(mock_llm.calls) == 2  # one per image
    assert any(r["image_id"] == "sunset" for r in results)

    # Second call should reuse cache
    search_fn(query="ocean")
    assert len(mock_llm.calls) == 2  # no new calls


def test_compare_images(tmp_path: Path):
    paths = [
        _make_test_image(tmp_path / "a.jpg", 100, 100, "red"),
        _make_test_image(tmp_path / "b.jpg", 100, 100, "blue"),
    ]
    mock_llm = MockLLMClient(["Image A is red, Image B is blue"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": paths, "question": "compare"})

    compare_fn = next(t.fn for t in toolkit.tools() if t.name == "compare_images")
    result = compare_fn(image_id_1="a", image_id_2="b")

    assert result == "Image A is red, Image B is blue"
    assert len(mock_llm.calls[0]["frame_paths"]) == 2
    assert toolkit._queried_images == {"a", "b"}


def test_build_evidence(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "photo.jpg", 200, 150)
    mock_llm = MockLLMClient(["description"])

    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit._llm_client = mock_llm
    toolkit.setup({"image": [img_path], "question": "test"})

    # Query the image so it appears in evidence
    query_fn = next(t.fn for t in toolkit.tools() if t.name == "vision_query")
    query_fn(prompt="describe", image_id="photo")

    evidence = toolkit.build_evidence()
    assert len(evidence) == 1
    assert evidence[0].source == "image:photo"
    assert "200x150" in evidence[0].rationale


def test_prompt_section(tmp_path: Path):
    img_path = _make_test_image(tmp_path / "test.jpg")
    toolkit = ImageToolkit(workspace_dir=str(tmp_path / "ws"))
    toolkit.setup({"image": [img_path], "question": "test"})

    section = toolkit.prompt_section()
    assert section is not None
    assert "vision_query" in section
    assert "list_images" in section
    assert "crop_region" in section
    assert "compare_images" in section
    assert "1 image(s) loaded" in section


def test_prompt_section_empty():
    toolkit = ImageToolkit()
    toolkit.setup({"question": "test"})
    assert toolkit.prompt_section() is None


# ── _parse_bbox_response tests ───────────────────────────────


def test_parse_bbox_valid():
    response = '{"x": 10, "y": 20, "x2": 100, "y2": 200}'
    result = _parse_bbox_response(response, 500, 500)
    assert result == (10, 20, 100, 200)


def test_parse_bbox_with_markdown():
    response = '```json\n{"x": 10, "y": 20, "x2": 100, "y2": 200}\n```'
    result = _parse_bbox_response(response, 500, 500)
    assert result == (10, 20, 100, 200)


def test_parse_bbox_in_text():
    response = 'The bounding box is {"x": 10, "y": 20, "x2": 100, "y2": 200} as requested.'
    result = _parse_bbox_response(response, 500, 500)
    assert result == (10, 20, 100, 200)


def test_parse_bbox_invalid_json():
    result = _parse_bbox_response("not json at all", 500, 500)
    assert result is None


def test_parse_bbox_inverted():
    result = _parse_bbox_response('{"x": 200, "y": 200, "x2": 10, "y2": 10}', 500, 500)
    assert result is None  # x >= x2
