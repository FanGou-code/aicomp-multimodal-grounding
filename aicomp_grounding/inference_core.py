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
    (``raw/Test/Images/...`` and ``derived/Processed/Test/depth_jet/...``), so
    a separate processed test index file is not required.
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
                    "visible": f"raw/Test/{item['visible']}",
                    "infrared": f"raw/Test/{item['infrared']}",
                    "depth": (
                        "derived/Processed/Test/depth_jet/"
                        f"{PurePosixPath(item['depth']).name}"
                    ),
                }
                for key, item in raw.items()
                if isinstance(item, dict)
            }

    items: list[dict] = []
    for key, value in raw.items():
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

    Mirrors the historical offline evaluation semantics: every prediction key
    counts toward the denominator; unparsable (None/invalid) predictions count
    as failures; mean IoU only accumulates for items with a valid GT bbox.
    """
    if not items or "bbox" not in items[0]:
        return None
    item_map = {item["key"]: item for item in items}

    hits = 0
    total_iou = 0.0
    failures = 0
    total = len(predictions)
    for key, pred_bbox in predictions.items():
        gt_item = item_map.get(key, {})
        gt_bbox = validate_bbox(gt_item.get("bbox"))
        valid_pred = validate_bbox(pred_bbox)

        if valid_pred is None:
            failures += 1
            continue
        if gt_bbox is not None:
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


def merge_shard_results(shard_results: list[dict]) -> dict:
    """Merge parallel shard payloads into a single prediction result.

    Effective runtime of a parallel run is the slowest shard (wall clock),
    not the sum of shard durations.
    """
    merged: dict[str, list[float] | None] = {}
    for result in shard_results:
        merged.update(result["predictions"])
    elapsed = (
        max(result["elapsed_seconds"] for result in shard_results)
        if shard_results
        else 0.0
    )
    return {"predictions": merged, "elapsed_seconds": elapsed}
