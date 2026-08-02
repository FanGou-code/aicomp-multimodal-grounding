"""Canonical bounding-box conversion, parsing, validation, and scoring."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

BBox = list[float]

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_PAIR_BODY = rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s*,\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"
_STRICT_PATTERNS = (
    re.compile(rf"<\|box_start\|>\s*{_PAIR_BODY}\s*<\|box_end\|>"),
    re.compile(_PAIR_BODY),
    re.compile(rf"\[\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\]"),
    re.compile(rf"({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})"),
)
_EMBEDDED_SPECIAL_TOKEN_PATTERN = re.compile(
    rf"<\|box_start\|>\s*{_PAIR_BODY}\s*<\|box_end\|>"
)


def validate_bbox(box: Sequence[float], *, normalized: bool = True) -> BBox | None:
    """Return a normalized list when ``box`` is finite, ordered, and in range."""
    if box is None or isinstance(box, (str, bytes)):
        return None

    try:
        if len(box) != 4:
            return None
    except TypeError:
        return None

    if any(isinstance(value, bool) for value in box):
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(value) for value in values):
        return None
    if normalized and not all(0.0 <= value <= 1.0 for value in values):
        return None

    x1, y1, x2, y2 = values
    if x1 >= x2 or y1 >= y2:
        return None
    return values


def normalize_pixel_bbox(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    *,
    clip: bool = True,
    precision: int | None = 6,
) -> BBox | None:
    """Convert an absolute ``[x, y, width, height]`` box to normalized XYXY."""
    if any(isinstance(value, bool) for value in (x, y, width, height)):
        return None
    if image_width <= 0 or image_height <= 0:
        return None
    if not all(math.isfinite(float(value)) for value in (x, y, width, height)):
        return None

    values = [
        float(x) / image_width,
        float(y) / image_height,
        (float(x) + float(width)) / image_width,
        (float(y) + float(height)) / image_height,
    ]
    if clip:
        values = [min(1.0, max(0.0, value)) for value in values]
    if precision is not None:
        values = [round(value, precision) for value in values]
    return validate_bbox(values)


def format_qwen_bbox(box: Sequence[float], *, special_tokens: bool = True) -> str:
    """Format normalized XYXY as Qwen's integer 0-1000 coordinate protocol."""
    values = validate_bbox(box)
    if values is None:
        raise ValueError(f"Invalid normalized bbox: {box!r}")

    x1, y1, x2, y2 = (round(value * 1000) for value in values)
    body = f"({x1},{y1}),({x2},{y2})"
    if special_tokens:
        return f"<|box_start|>{body}<|box_end|>"
    return body


def parse_bbox_from_text(text: str, *, coordinate_scale: str = "qwen_1000") -> BBox | None:
    """Parse an explicitly formatted bbox without consuming unrelated prose numbers.

    ``coordinate_scale`` must be ``qwen_1000`` for integer Qwen coordinates or
    ``normalized`` for coordinates already in the official 0-1 range.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if coordinate_scale not in {"qwen_1000", "normalized"}:
        raise ValueError(f"Unsupported coordinate scale: {coordinate_scale}")

    candidate = text.strip().strip("`").strip()
    match = next(
        (matched for pattern in _STRICT_PATTERNS if (matched := pattern.fullmatch(candidate))), None
    )
    if match is None and candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and set(payload) == {"bbox_2d"}:
            values = payload["bbox_2d"]
            if isinstance(values, list) and len(values) == 4:
                text_values = ",".join(str(value) for value in values)
                return parse_bbox_from_text(f"[{text_values}]", coordinate_scale=coordinate_scale)

    if match is None:
        special_matches = list(_EMBEDDED_SPECIAL_TOKEN_PATTERN.finditer(candidate))
        if len(special_matches) == 1:
            match = special_matches[0]
    if match is None:
        return None

    values = [float(value) for value in match.groups()]
    if coordinate_scale == "qwen_1000":
        if not all(0.0 <= value <= 1000.0 for value in values):
            return None
        values = [value / 1000.0 for value in values]
    return validate_bbox(values)


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute standard IoU for two valid normalized XYXY boxes."""
    a = validate_bbox(box_a)
    b = validate_bbox(box_b)
    if a is None or b is None:
        return 0.0

    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    intersection = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


