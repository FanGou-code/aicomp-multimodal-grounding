"""Weighted Boxes Fusion for heterogeneous grounding models.

Design notes
------------
- Input units are per-model ``predictions.json`` files (``{key: bbox|None}``),
  i.e. exactly what the inference entrypoint emits, so contributors can
  exchange prediction files without any new contract.
- Weights are **model-level** (per input file). VLMs emit no calibrated
  confidence, so equal weights are the honest default; calibrate on val if
  budget allows. DINO's native per-box scores can be supplied via optional
  ``--scores`` files and multiply into the effective weight.
- Fusion per query: greedy IoU clustering of the valid boxes, cluster score
  = sum of member weights, winner = heaviest cluster, fused box = weighted
  average of the cluster's coordinates (classic WBF, single-class reduced).
- Fusion only. This module never builds a submission package; packaging is the
  separate, explicit ``aicomp_grounding.submission`` step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aicomp_grounding.bbox import compute_iou, validate_bbox
from aicomp_grounding.io import atomic_write_json, load_json

ALGORITHM = "wbf-v1"


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fuse_boxes(boxes: list[list[float]], weights: list[float]) -> list[float]:
    """Weighted coordinate average of a cluster of valid XYXY boxes."""
    total = sum(weights)
    return [
        round(sum(w * box[k] for w, box in zip(weights, boxes, strict=True)) / total, 6)
        for k in range(4)
    ]


def wbf_fuse_key(
    box_weight_pairs: list[tuple[list[float], float]],
    *,
    iou_threshold: float,
) -> list[float] | None:
    """Fuse one query's (box, weight) pairs; None when nothing is valid."""
    valid = [
        (box, weight)
        for box, weight in box_weight_pairs
        if weight > 0.0 and validate_bbox(box) is not None
    ]
    if not valid:
        return None

    # Greedy clustering: heaviest box first, attach to the best-matching
    # existing cluster, otherwise open a new one.
    clusters: list[dict] = []
    for box, weight in sorted(valid, key=lambda pair: -pair[1]):
        target = None
        best_iou = iou_threshold
        for cluster in clusters:
            for member_box in cluster["boxes"]:
                iou = compute_iou(box, member_box)
                if iou > best_iou:
                    best_iou = iou
                    target = cluster
        if target is None:
            clusters.append({"boxes": [box], "weights": [weight]})
        else:
            target["boxes"].append(box)
            target["weights"].append(weight)

    winner = max(clusters, key=lambda cluster: sum(cluster["weights"]))
    return fuse_boxes(winner["boxes"], winner["weights"])


def fuse_predictions(
    prediction_files: list[Path | str],
    *,
    weights: list[float],
    iou_threshold: float = 0.55,
    score_files: list[Path | str | None] | None = None,
) -> dict[str, list[float] | None]:
    """Fuse N prediction files into one prediction dict keyed by query id."""
    if len(weights) != len(prediction_files):
        raise ValueError(
            f"weights ({len(weights)}) must align with prediction files "
            f"({len(prediction_files)})"
        )
    if score_files is not None and len(score_files) != len(prediction_files):
        raise ValueError("score files must align with prediction files")

    loaded = [load_json(path) for path in prediction_files]
    loaded_scores = [
        None if path is None else load_json(path)
        for path in (score_files or [None] * len(prediction_files))
    ]
    all_keys: set[str] = set()
    for predictions in loaded:
        all_keys.update(predictions)

    fused: dict[str, list[float] | None] = {}
    for key in sorted(all_keys):
        pairs: list[tuple[list[float], float]] = []
        for index, predictions in enumerate(loaded):
            box = predictions.get(key)
            if box is None:
                continue
            weight = float(weights[index])
            if loaded_scores[index] is not None:
                score = loaded_scores[index].get(key)
                if score is not None:
                    weight *= float(score)
            pairs.append((box, weight))
        fused[key] = wbf_fuse_key(pairs, iou_threshold=iou_threshold)
    return fused


def build_fusion_metadata(
    prediction_files: list[Path | str],
    *,
    weights: list[float],
    iou_threshold: float,
    score_files: list[Path | str | None] | None = None,
) -> dict:
    """Deterministic fusion identity from input file bytes + parameters."""
    payload = {
        "algorithm": ALGORITHM,
        "inputs": [
            {"path": Path(path).name, "sha256": _file_sha256(path)}
            for path in prediction_files
        ],
        "weights": [float(weight) for weight in weights],
        "iou_threshold": float(iou_threshold),
        "scores": [
            None
            if score is None
            else {"path": Path(score).name, "sha256": _file_sha256(score)}
            for score in (score_files or [None] * len(prediction_files))
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    run_id = f"fusion_wbf_{hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:16]}"
    return {"algorithm": ALGORITHM, "run_id": run_id, "inputs": payload}


def normalize_score_files(values: list[str] | None) -> list[Path | None] | None:
    """Convert CLI score arguments; empty strings mean no score file for that model."""
    if values is None:
        return None
    return [None if str(value) == "" else Path(value) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more per-model predictions.json files.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Model-level weights aligned with --predictions (default: all 1.0).",
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        default=None,
        help="Optional per-model score files (e.g. DINO confidences); use '' to skip a model.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.55)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/fusion"))
    args = parser.parse_args()

    if len(args.predictions) < 2:
        parser.error("WBF needs at least two prediction files")
    weights = args.weights or [1.0] * len(args.predictions)
    score_files = normalize_score_files(args.scores)

    metadata = build_fusion_metadata(
        args.predictions,
        weights=weights,
        iou_threshold=args.iou_threshold,
        score_files=score_files,
    )
    fused = fuse_predictions(
        args.predictions,
        weights=weights,
        iou_threshold=args.iou_threshold,
        score_files=score_files,
    )

    run_dir = args.output_dir / metadata["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.json"
    atomic_write_json(predictions_path, fused)
    atomic_write_json(run_dir / "metadata.json", metadata)

    valid = sum(1 for box in fused.values() if box is not None)
    print(f"[wbf] run_id: {metadata['run_id']}")
    print(f"[wbf] queries: {len(fused)} | fused valid boxes: {valid}")
    print(f"[wbf] predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
