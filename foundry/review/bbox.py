"""Bounding-box validation in normalized 0-1 XYXY coordinates.

The on-disk annotation format matches the main project's prediction files:
``{"<item_id>": [x1, y1, x2, y2]}`` with all values in [0, 1].


Vendored from gt-annotator (github.com/FanGou-code/gt-annotator,
MIT License, (c) 2026 FanGou-code) - adapted for query-foundry review."""

from __future__ import annotations

import math

# Reject boxes that are empty (or empty after clipping to [0, 1]).
MIN_EXTENT = 1e-6


def normalize_bbox(values: object) -> list[float]:
    """Validate and clip ``[x1, y1, x2, y2]``; return canonical floats.

    Raises ValueError on wrong shape, non-numeric or non-finite entries, and
    boxes that are empty after clipping.
    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise ValueError(f"bbox must be a sequence of 4 numbers, got {values!r}")
    if len(values) != 4:
        raise ValueError(f"bbox must have exactly 4 numbers, got {len(values)}")
    try:
        x1, y1, x2, y2 = (float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bbox entries must be numbers: {values!r}") from exc
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"bbox entries must be finite: {values!r}")
    x1 = min(max(x1, 0.0), 1.0)
    y1 = min(max(y1, 0.0), 1.0)
    x2 = min(max(x2, 0.0), 1.0)
    y2 = min(max(y2, 0.0), 1.0)
    if x2 - x1 <= MIN_EXTENT or y2 - y1 <= MIN_EXTENT:
        raise ValueError(f"bbox is empty after clipping: {values!r}")
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]
