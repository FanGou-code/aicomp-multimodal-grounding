"""Axis values and the ordering rule shared by the serving and annotation sides.

``ordinal.resolve.rank_instances`` (serving) and ``annotation.reverse.pick_kth``
(annotation) order candidates identically: sort by the primary axis value, with
ties keeping ``(x1, y1)`` ascending whatever the direction.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Closed ordinal axis set: every axis is computable from a sample's boxes,
#: the 16-bit millimetre depth map, or the infrared image.
AXES = ("x", "y", "depth", "ir", "area")
DIRECTIONS = ("asc", "desc")


def _box_patch(array, bbox: Sequence[float], image_size: tuple[int, int]):
    """Slice ``array`` (height, width) or (height, width, channels) to the box."""
    if image_size is None:
        return None
    width, height = image_size
    shape = getattr(array, "shape", None)
    if shape is None or len(shape) < 2 or shape[0] != height or shape[1] != width:
        return None
    x1 = min(width - 1, max(0, int(math.floor(bbox[0] * width))))
    y1 = min(height - 1, max(0, int(math.floor(bbox[1] * height))))
    x2 = min(width, max(x1 + 1, int(math.ceil(bbox[2] * width))))
    y2 = min(height, max(y1 + 1, int(math.ceil(bbox[3] * height))))
    return array[y1:y2, x1:x2]


def axis_value(
    axis: str,
    bbox: Sequence[float],
    *,
    depth_mm=None,
    ir=None,
    image_size: tuple[int, int] | None = None,
) -> float | None:
    """Return the sort value of one box on one axis, or None if unsupported."""
    if axis == "x":
        return float(bbox[0])
    if axis == "y":
        return float(bbox[1])
    if axis == "area":
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if axis == "depth":
        patch = None if depth_mm is None else _box_patch(depth_mm, bbox, image_size)
        if patch is None:
            return None
        valid = patch[patch > 0]
        if getattr(valid, "size", 0) == 0:
            return None
        import numpy

        return float(numpy.median(valid))
    if axis == "ir":
        patch = None if ir is None else _box_patch(ir, bbox, image_size)
        if patch is None:
            return None
        if getattr(patch, "ndim", 0) == 3:
            patch = patch[:, :, 0]
        if getattr(patch, "size", 0) == 0:
            return None
        return float(patch.mean())
    return None


def order_indices(
    values: Sequence[float],
    boxes: Sequence[Sequence[float]],
    *,
    direction: str,
) -> list[int]:
    """Indices of ``boxes`` in ordinal order.

    The primary key is ``values``; ties keep ``(x1, y1)`` ascending whatever the
    direction, so the order never depends on the sort's stability.
    """
    order = sorted(range(len(boxes)), key=lambda index: (boxes[index][0], boxes[index][1]))
    order.sort(key=lambda index: values[index], reverse=(direction == "desc"))
    return order
