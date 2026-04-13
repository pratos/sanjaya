"""Image loading, normalization, and manipulation helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".webp", ".gif",
    ".tiff", ".tif", ".bmp",
    ".heic", ".heif",
    ".svg",
})

_HEIC_EXTENSIONS = frozenset({".heic", ".heif"})
_SVG_EXTENSIONS = frozenset({".svg"})


@dataclass
class ImageInfo:
    """Metadata about a loaded image."""

    path: str
    image_id: str
    width: int
    height: int
    format: str
    mode: str
    file_size_bytes: int
    normalized_path: str | None = None


def _open_heic(path: Path) -> Any:
    """Open a HEIC/HEIF file via pillow-heif, return a PIL Image."""
    try:
        import pillow_heif  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "HEIC/HEIF support requires pillow-heif. "
            "Install it with: uv add pillow-heif  (or pip install pillow-heif)"
        ) from None

    from PIL import Image

    heif_file = pillow_heif.open_heif(str(path))
    return Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
    )


def _open_svg(path: Path) -> Any:
    """Rasterize an SVG file via cairosvg, return a PIL Image."""
    try:
        import cairosvg  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "SVG support requires cairosvg. "
            "Install it with: uv add cairosvg  (or pip install cairosvg)"
        ) from None

    from PIL import Image

    png_bytes = cairosvg.svg2png(url=str(path))
    return Image.open(io.BytesIO(png_bytes))


def _open_image(path: Path) -> Any:
    """Open an image file with Pillow, handling HEIC and SVG specially."""
    from PIL import Image

    suffix = path.suffix.lower()

    if suffix in _HEIC_EXTENSIONS:
        return _open_heic(path)
    if suffix in _SVG_EXTENSIONS:
        return _open_svg(path)

    return Image.open(path)


def load_image(path: str) -> ImageInfo:
    """Load an image and return its metadata.

    Supports JPEG, PNG, WebP, GIF, TIFF, BMP natively via Pillow.
    HEIC requires pillow-heif; SVG requires cairosvg.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    img = _open_image(p)
    width, height = img.size
    fmt = img.format or suffix.lstrip(".").upper()
    mode = img.mode

    # Derive image_id from filename stem (caller may override)
    image_id = p.stem

    return ImageInfo(
        path=str(p),
        image_id=image_id,
        width=width,
        height=height,
        format=fmt,
        mode=mode,
        file_size_bytes=p.stat().st_size,
    )


def normalize_for_vision(
    path: str,
    max_dim: int = 1536,
    quality: int = 80,
) -> bytes:
    """Resize and convert an image to JPEG bytes for vision model input.

    Downscales the longest side to *max_dim* (preserving aspect ratio).
    Converts RGBA to RGB by compositing onto a white background.
    """
    p = Path(path).resolve()
    img = _open_image(p)

    # RGBA → RGB (composite on white)
    if img.mode in ("RGBA", "LA", "PA"):
        from PIL import Image

        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])  # alpha channel as mask
        img = background
    elif img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img.thumbnail((max_dim, max_dim))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def normalize_to_file(
    path: str,
    output_path: str,
    max_dim: int = 1536,
    quality: int = 80,
) -> str:
    """Normalize an image and save to disk. Returns the output path."""
    data = normalize_for_vision(path, max_dim=max_dim, quality=quality)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return str(out)


def crop_image(
    path: str,
    bbox: tuple[int, int, int, int],
    output_path: str,
) -> str:
    """Crop an image to a pixel bounding box and save.

    Args:
        path: Source image path.
        bbox: (x, y, x2, y2) — left, top, right, bottom in pixels.
        output_path: Where to save the cropped image.

    Returns:
        The output path.
    """
    p = Path(path).resolve()
    img = _open_image(p)

    x, y, x2, y2 = bbox
    w, h = img.size

    # Clamp to image bounds
    x = max(0, min(x, w))
    y = max(0, min(y, h))
    x2 = max(x + 1, min(x2, w))
    y2 = max(y + 1, min(y2, h))

    cropped = img.crop((x, y, x2, y2))

    # Convert if needed for JPEG save
    if cropped.mode in ("RGBA", "LA", "PA"):
        from PIL import Image

        background = Image.new("RGB", cropped.size, (255, 255, 255))
        background.paste(cropped, mask=cropped.split()[-1])
        cropped = background
    elif cropped.mode not in ("RGB", "L"):
        cropped = cropped.convert("RGB")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(str(out), format="JPEG", quality=85, optimize=True)
    return str(out)
