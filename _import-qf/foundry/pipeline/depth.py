"""Local depth facts from the raw uint16 millimeter depth maps.

The dataset stores per-sample raw depth PNGs (``Train/<seq>/depth/<frame>.png``,
single-channel uint16, millimeters, 0 = invalid sensor reading). Facts are
computed in-process and travel as JSON numbers inside the census merged.json —
no image files are ever produced. Millimeter semantics are exact: smaller
value = nearer to the camera; no colormap decoding, no cross-frame
normalization concerns, no fallback ladder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

NEAREST_MARGIN_MM = 200
BAND_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)
DEPTH_SOURCE = "raw-uint16-mm"


def raw_depth_path(data_root: Path, visible_reference: str) -> Path:
    """Derive the raw depth path from the visible modality reference.

    ``Train/<seq>/color/<frame>.png`` → ``Train/<seq>/depth/<frame>.png``.
    """
    parts = visible_reference.split("/")
    if len(parts) >= 2 and parts[-2] == "color":
        parts[-2] = "depth"
    return data_root.joinpath(*parts)


def load_depth_millimeters(path: Path) -> np.ndarray:
    """Load a raw depth PNG as a uint16 millimeter array (0 = invalid)."""
    with Image.open(path) as img:
        return np.asarray(img, dtype=np.uint16)


def object_depth_medians(
    depth_mm: np.ndarray, objects: list[dict]
) -> dict[int, int | None]:
    """Median millimeters inside each object's bbox; ``None`` when fully invalid."""
    height, width = depth_mm.shape[:2]
    medians: dict[int, int | None] = {}
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        left = max(0, int(x1 * width))
        right = min(width, int(round(x2 * width)))
        top = max(0, int(y1 * height))
        bottom = min(height, int(round(y2 * height)))
        if right <= left or bottom <= top:
            medians[obj["i"]] = None
            continue
        pixels = depth_mm[top:bottom, left:right]
        valid = pixels[pixels > 0]
        medians[obj["i"]] = int(np.median(valid)) if valid.size else None
    return medians


def depth_ranks(medians: dict[int, int | None]) -> dict[int, int]:
    """Rank objects by median depth, 1 = nearest. Objects without depth excluded."""
    ranked = sorted(
        ((index, mm) for index, mm in medians.items() if mm is not None),
        key=lambda pair: (pair[1], pair[0]),
    )
    return {index: rank for rank, (index, _) in enumerate(ranked, start=1)}


def frame_depth_facts(depth_mm: np.ndarray, objects: list[dict]) -> dict:
    """Per-object depth facts for one frame, plus the frame's valid depth span."""
    valid = depth_mm[depth_mm > 0]
    medians = object_depth_medians(depth_mm, objects)
    return {
        "source": DEPTH_SOURCE,
        "frame_min_mm": int(valid.min()) if valid.size else None,
        "frame_max_mm": int(valid.max()) if valid.size else None,
        "objects": {str(index): {"median_mm": mm} for index, mm in medians.items()},
        "ranks": {str(index): rank for index, rank in depth_ranks(medians).items()},
    }


def is_nearest(median_mm: int | None, others_mm: list[int], margin_mm: int = NEAREST_MARGIN_MM) -> bool:
    """True when the object leads every other valid object by ``margin_mm``."""
    if median_mm is None or not others_mm:
        return False
    return median_mm <= min(others_mm) - margin_mm


def is_farthest(median_mm: int | None, others_mm: list[int], margin_mm: int = NEAREST_MARGIN_MM) -> bool:
    """True when the object trails every other valid object by ``margin_mm``."""
    if median_mm is None or not others_mm:
        return False
    return median_mm >= max(others_mm) + margin_mm


def is_foreground(median_mm: int | None, frame_min_mm: int | None, frame_max_mm: int | None) -> bool:
    """Near-band membership: median within the nearest third of the frame span."""
    if median_mm is None or frame_min_mm is None or frame_max_mm is None:
        return False
    span = frame_max_mm - frame_min_mm
    if span <= 0:
        return False
    return median_mm <= frame_min_mm + span * BAND_FRACTIONS[0]


def is_background(median_mm: int | None, frame_min_mm: int | None, frame_max_mm: int | None) -> bool:
    """Far-band membership: median within the farthest third of the frame span."""
    if median_mm is None or frame_min_mm is None or frame_max_mm is None:
        return False
    span = frame_max_mm - frame_min_mm
    if span <= 0:
        return False
    return median_mm >= frame_min_mm + span * BAND_FRACTIONS[1]
