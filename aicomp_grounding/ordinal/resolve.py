"""Ordinal resolve entry point: gates, axis ranking, the k-th pick, and the file hand-off.

Pure code, no model, no I/O: callers hand in the base box, the parsed intent,
the enumeration payload, and (for depth/ir axes) the pixel arrays.  Every
outcome is either "keep the base box" or "replace it with the k-th instance".
"""

from __future__ import annotations

import argparse
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from aicomp_grounding.bbox import compute_iou, validate_bbox
from aicomp_grounding.inference_core import load_inference_items
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.ordinal import run

#: Closed axis set.  Every axis is computable from what a sample carries:
#: boxes, the 16-bit millimetre depth map, the infrared image.
AXES = ("x", "y", "depth", "ir", "area")
DIRECTIONS = ("asc", "desc")

MATCH_IOU = 0.5


@dataclass(frozen=True)
class Instance:
    """One enumerated instance of the queried category."""

    bbox: list[float]
    confidence: float | None = None


@dataclass(frozen=True)
class Selection:
    """Parsed intent: which category, and how the target is picked out."""

    category: str
    mode: str  # "unique" | "rank"
    k: int | None = None
    axis: str | None = None
    direction: str | None = None


@dataclass(frozen=True)
class Decision:
    """What to do with one query.

    ``bbox is None`` means "leave the prediction exactly as it is"; only a
    ``replace`` carries coordinates, so a missing box is a no-op for the caller.
    """

    action: str  # "keep" | "replace"
    bbox: list[float] | None
    reason: str


def coerce_instance(raw: object) -> Instance | None:
    """Accept a mapping or an Instance and return a valid Instance, else None."""
    if isinstance(raw, Instance):
        bbox = validate_bbox(raw.bbox)
        return None if bbox is None else Instance(bbox, raw.confidence)
    if isinstance(raw, dict):
        bbox = validate_bbox(raw.get("bbox"))
        if bbox is None:
            return None
        confidence = raw.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            confidence = None
        return Instance(bbox, None if confidence is None else float(confidence))
    return None


def coerce_instances(raw_run: object) -> list[Instance] | None:
    """Return a valid instance list, or None if any entry is malformed."""
    if not isinstance(raw_run, (list, tuple)):
        return None
    out: list[Instance] = []
    for raw in raw_run:
        instance = coerce_instance(raw)
        if instance is None:
            return None
        out.append(instance)
    return out


def parse_selection(payload: object) -> Selection | None:
    """Validate a parsed-intent payload; None when it does not meet the schema."""
    if not isinstance(payload, dict):
        return None
    category = payload.get("category")
    if not isinstance(category, str) or not category.strip():
        return None
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        return None
    mode = selection.get("mode")
    if mode == "unique":
        return Selection(category.strip(), "unique")
    if mode != "rank":
        return None
    k = selection.get("k")
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        return None
    axis = selection.get("axis")
    if axis not in AXES:
        return None
    direction = selection.get("direction")
    if direction not in DIRECTIONS:
        return None
    return Selection(category.strip(), "rank", k, axis, direction)


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
    """Return the sort value of one instance on one axis, or None if unsupported."""
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


def rank_instances(
    instances: Sequence[Instance],
    axis: str,
    direction: str,
    *,
    depth_mm=None,
    ir=None,
    image_size: tuple[int, int] | None = None,
) -> list[Instance] | None:
    """Order instances along ``axis``; None when any instance lacks that axis.

    Ties are broken by the box tuple ascending regardless of ``direction`` so the
    order never depends on the sort's stability.
    """
    values: list[float] = []
    for instance in instances:
        value = axis_value(
            axis, instance.bbox, depth_mm=depth_mm, ir=ir, image_size=image_size
        )
        if value is None:
            return None
        values.append(value)
    order = sorted(range(len(instances)), key=lambda i: (instances[i].bbox[0], instances[i].bbox[1]))
    order.sort(key=lambda i: values[i], reverse=(direction == "desc"))
    return [instances[i] for i in order]


def resolve_query(
    base_bbox: Sequence[float] | None,
    parse_payload: object,
    run_payload: object,
    *,
    depth_mm=None,
    ir=None,
    image_size: tuple[int, int] | None = None,
    iou_threshold: float = MATCH_IOU,
) -> Decision:
    """Run every gate and return the final box for one query.

    ``run_payload`` is one enumeration's decoded ``{"count", "instances"}``.
    """
    selection = parse_selection(parse_payload)
    if selection is None:
        return Decision("keep", None, "parse-invalid")
    if selection.mode != "rank":
        return Decision("keep", None, "not-rank")

    if not isinstance(run_payload, dict):
        return Decision("keep", None, "run-malformed")
    instances = coerce_instances(run_payload.get("instances"))
    if instances is None:
        return Decision("keep", None, "run-malformed")
    count = run_payload.get("count")
    if isinstance(count, bool) or not isinstance(count, int):
        return Decision("keep", None, "count-missing")
    if count != len(instances):
        return Decision("keep", None, "truncated")
    if count == 0:
        return Decision("keep", None, "count-zero")

    if selection.k > len(instances):
        return Decision("keep", None, "k-out-of-range")

    ordered = rank_instances(
        instances, selection.axis, selection.direction,
        depth_mm=depth_mm, ir=ir, image_size=image_size,
    )
    if ordered is None:
        return Decision("keep", None, "axis-unsupported")

    target = ordered[selection.k - 1]
    base = validate_bbox(base_bbox) if base_bbox is not None else None
    if base is not None and compute_iou(base, target.bbox) >= iou_threshold:
        return Decision("keep", None, "already-correct")
    return Decision("replace", list(target.bbox), "replaced")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _axis_arrays(item: dict, data_dir: Path, cache: dict):
    key = str(item.get("visible", ""))
    if key not in cache:
        cache[key] = run.load_axis_arrays(item, data_dir)
    return cache[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enum-run", type=Path, required=True, help="the enumeration run directory")
    parser.add_argument("--predictions", type=Path, required=True, help="the predictions.json to repair")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--iou-threshold", type=float, default=MATCH_IOU)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ordinal"))
    parser.add_argument("--resume", action="store_true", help="write into an existing run directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    items, _approved = load_inference_items(args.test_json, limit=args.limit)
    dataset = {item["key"]: item for item in items}
    keys = [item["key"] for item in items]
    enum_metadata, parse_map, instances_map = run.load_enum_artifacts(args.enum_run)
    predictions = load_json(args.predictions)
    prediction_fingerprint = _file_sha256(args.predictions)

    def metadata(stats: dict | None = None) -> dict:
        return run.build_resolve_metadata(
            enum_run_id=enum_metadata["run_id"],
            prediction_fingerprint=prediction_fingerprint,
            iou_threshold=args.iou_threshold,
            run_tag=args.run_tag,
            selected_keys=keys,
            stats=stats,
        )

    out_dir = args.output_dir / metadata()["run_id"]
    if out_dir.exists() and not args.resume:
        raise SystemExit(f"run directory already exists: {out_dir} (pass --resume or a new --run-tag)")
    out_dir.mkdir(parents=True, exist_ok=True)

    final_predictions = dict(predictions)
    stats: dict = {
        "queries": len(keys),
        "rank_queries": 0,
        "adopted": 0,
        "reasons": {},
    }
    axis_cache: dict = {}

    for key in keys:
        item = dataset.get(key)
        intent = (parse_map.get(key) or {}).get("intent")
        selection = parse_selection(intent)
        is_rank = selection is not None and selection.mode == "rank"
        if is_rank:
            stats["rank_queries"] += 1

        depth = infrared = size = None
        if item is not None and is_rank and selection.axis in ("depth", "ir"):
            depth, infrared, size = _axis_arrays(item, args.data_dir, axis_cache)

        payload = (instances_map.get(key) or {}).get("payload")
        decision = resolve_query(
            predictions.get(key),
            intent,
            payload,
            depth_mm=depth,
            ir=infrared,
            image_size=size,
            iou_threshold=args.iou_threshold,
        )
        stats["reasons"][decision.reason] = stats["reasons"].get(decision.reason, 0) + 1

        if decision.action == "replace" and decision.bbox is not None:
            final_predictions[key] = list(decision.bbox)
            stats["adopted"] += 1

    name = f"predictions_{enum_metadata['model'].replace('/', '-')}.json"
    atomic_write_json(out_dir / name, final_predictions)
    outputs = [{"model": enum_metadata["model"], "file": name}]

    atomic_write_json(out_dir / "metadata.json", metadata({**stats, "outputs": outputs}))
    print(f"[ordinal-resolve] run_id={out_dir.name}")
    print(
        f"[ordinal-resolve] rank_queries={stats['rank_queries']}/{stats['queries']} "
        f"adopted={stats['adopted']}"
    )
    print(f"[ordinal-resolve] reasons: {stats['reasons']}")
    print(f"[ordinal-resolve] wrote {out_dir}")


if __name__ == "__main__":
    main()
