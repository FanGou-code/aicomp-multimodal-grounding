"""Red-box marked image rendering for the annotation pipeline.

Pure PIL helpers, no QC/state logic. Extracted from ``annotation_state``
(2026-09-01 cleanup); the render-section constants are fingerprint-bearing
metadata values — never change them retroactively.
"""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw

from aicomp_grounding.bbox import validate_bbox

RENDER_PROTOCOL = "single-marked-full-rgb-v8"
MARK_COLOR = (255, 0, 0)


def _resize_preserving_aspect(image: Image.Image, max_side: int) -> Image.Image:
    resized = image.convert("RGB")
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _draw_outward_box(image: Image.Image, pixel_box: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = pixel_box
    padding = max(4, round(min(image.size) * 0.006))
    line_width = max(3, round(min(image.size) * 0.004))
    ImageDraw.Draw(image).rectangle(
        (
            max(0, round(x1) - padding),
            max(0, round(y1) - padding),
            min(image.width - 1, round(x2) + padding),
            min(image.height - 1, round(y2) + padding),
        ),
        outline=MARK_COLOR,
        width=line_width,
    )


def build_marked_annotation_view(image: Image.Image, bbox: list[float]) -> Image.Image:
    """Return one complete RGB scene with an outward target rectangle."""
    normalized = validate_bbox(bbox)
    if normalized is None:
        raise ValueError(f"Invalid annotation bbox: {bbox!r}")
    marked = _resize_preserving_aspect(image, 1536)
    width, height = marked.size
    x1, y1, x2, y2 = normalized
    _draw_outward_box(
        marked,
        (x1 * width, y1 * height, x2 * width, y2 * height),
    )
    return marked


def jpeg_data_url(image: Image.Image, *, quality: int = 90) -> str:
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
