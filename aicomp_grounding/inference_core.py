"""Platform-agnostic inference core shared by cloud and offline entrypoints.

Pure, torch-free helpers for: loading inference items from the supported
JSON shapes (approved annotation artifact, flat query index, or plain list),
computing ACC@0.5 / mean-IoU metrics against ground truth, and merging the
prediction payloads of parallel shards.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from aicomp_grounding.bbox import compute_iou, validate_bbox
from aicomp_grounding.io import load_json


def load_inference_items(
    path: str | Path,
    *,
    limit: int = 0,
) -> tuple[list[dict], dict | None]:
    """Load a normalized item list from the JSON object at ``path``.

    Accepts an approved annotation artifact (``{metadata, data}``), a flat
    ``{query_id: item}`` index, or the official Test template directly.
    Official template paths are mapped in memory to the processed worker layout
    (``Test/Images/...`` and ``Processed/Test/depth_jet/...``), so a separate
    ``data/test.json`` file is not required.
    Every item gains a ``"key"`` entry. Returns ``(items, approved_metadata_or_None)``.
    """
    raw = load_json(path)

    approved_metadata = None
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("data"), dict)
        and isinstance(raw.get("metadata"), dict)
    ):
        approved_metadata = raw["metadata"]
        raw = raw["data"]

    if isinstance(raw, dict) and raw:
        first = next(iter(raw.values()))
        if (
            isinstance(first, dict)
            and isinstance(first.get("visible"), str)
            and first["visible"].startswith("Images/visible/")
        ):
            raw = {
                key: {
                    **item,
                    "visible": f"Test/{item['visible']}",
                    "infrared": f"Test/{item['infrared']}",
                    "depth": (
                        "Processed/Test/depth_jet/"
                        f"{PurePosixPath(item['depth']).name}"
                    ),
                }
                for key, item in raw.items()
                if isinstance(item, dict)
            }

    items: list[dict] = []
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError(
                f"Dataset entry {key!r} is not a JSON object; refusing to skip it silently"
            )
        entry = dict(value)
        entry["key"] = key
        items.append(entry)

    if limit > 0:
        items = items[:limit]
    return items, approved_metadata


def evaluate_predictions(
    items: list[dict],
    predictions: dict[str, list[float] | None],
) -> dict | None:
    """Compute ACC@0.5 / mean-IoU metrics, or None when items carry no GT.

    Driven by the dataset: every evaluated item counts toward the denominator,
    so a partial prediction set is scored as failures instead of silently
    shrinking the denominator. Unparsable (None/invalid) predictions count as
    failures; mean IoU only accumulates for items with a valid GT bbox.
    """
    if not items or "bbox" not in items[0]:
        return None

    hits = 0
    total_iou = 0.0
    failures = 0
    total = len(items)
    for item in items:
        gt_bbox = validate_bbox(item.get("bbox"))
        if gt_bbox is None:
            raise ValueError(
                f"Sample {item.get('key')!r} has an invalid ground-truth bbox"
            )
        valid_pred = validate_bbox(predictions.get(item["key"]))

        if valid_pred is None:
            failures += 1
            continue
        iou = compute_iou(valid_pred, gt_bbox)
        total_iou += iou
        if iou >= 0.5:
            hits += 1

    return {
        "hits": hits,
        "total": total,
        "acc_at_0_5": hits / total if total > 0 else 0.0,
        "mean_iou": total_iou / total if total > 0 else 0.0,
        "failures": failures,
    }
