"""Pure state management for resumable sharded inference.

Data structures
---------------
Run metadata dict (from build_run_metadata):
    {version, run_id, mode, split, annotation_run_id, model, model_revision,
     lora_path, adapter_fingerprint, prompt_hash, generation_config, compute_dtype,
     min_pixels, max_pixels, run_tag, limit, selected_key_hash, input_fingerprint,
     image_fingerprint, num_shards, base_run_id, base_prediction_fingerprint}

Shard payload / checkpoint dict:
    {metadata: RunMetadata + {shard_id, assigned_key_hash},
     predictions: dict[query_id, [x1,y1,x2,y2] | None]}

Inference plan dict (from prepare_inference_plan):
    {metadata, keys, shards, result_keys, all_key_hash, result_is_full_split,
     base_predictions, completed_payloads, pending_shard_ids}

Summary dict (summary.json):
    {metadata, metrics: {hits, total, acc_at_0_5, mean_iou, failures} | None,
     total_predictions, valid_predictions, submission_ready}
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path

from aicomp_grounding.artifacts import (
    key_hash,
    require_exact_metadata,
    stable_json_hash,
)
from aicomp_grounding.bbox import compute_iou, validate_bbox
from aicomp_grounding.config import (
    CHECKPOINT_VERSION,
    INFERENCE_COMPUTE_DTYPE,
    RUNTIME_PYTHON_VERSION,
)
from aicomp_grounding.io import load_json
from aicomp_grounding.models.qwen3vl import MAX_PIXELS, MIN_PIXELS
from aicomp_grounding.training_state import validate_adapter_manifest, adapter_weight_path

RUN_METADATA_FIELDS = (
    "version",
    "python_version",
    "run_id",
    "mode",
    "split",
    "annotation_run_id",
    "model",
    "model_revision",
    "lora_path",
    "adapter_fingerprint",
    "prompt_hash",
    "generation_config",
    "compute_dtype",
    "min_pixels",
    "max_pixels",
    "run_tag",
    "limit",
    "selected_key_hash",
    "input_fingerprint",
    "image_fingerprint",
    "num_shards",
    "base_run_id",
    "base_prediction_fingerprint",
)


def resolve_lora_path(raw_path: str | Path | None, data_root: str | Path) -> Path | None:
    """Resolve a LoRA directory inside the mounted dataset root and validate its files."""
    if raw_path is None or str(raw_path).strip() == "":
        return None
    root = Path(data_root).resolve()
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"LoRA adapter must be inside {root}: {candidate}") from exc
    if not candidate.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {candidate}")
    config_path = candidate / "adapter_config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise FileNotFoundError(f"LoRA adapter is missing adapter_config.json: {candidate}")
    load_json(config_path)
    manifest_path = candidate / "adapter_manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError(f"LoRA adapter is missing adapter_manifest.json: {candidate}")
    validate_adapter_manifest(load_json(manifest_path))
    if adapter_weight_path(candidate) is None:
        raise FileNotFoundError(
            f"LoRA adapter has no adapter_model.safetensors or adapter_model.bin: {candidate}"
        )
    return candidate


def fingerprint_lora(adapter_dir: Path | None) -> str:
    """Hash complete adapter configuration and weight files; never fall back to path identity."""
    if adapter_dir is None:
        return "base"
    weight_file = adapter_weight_path(adapter_dir)
    files = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_manifest.json"]
    if weight_file:
        files.append(weight_file)
    if len(files) < 3 or any(not path.is_file() for path in files):
        raise FileNotFoundError(f"Incomplete LoRA adapter directory: {adapter_dir}")

    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(adapter_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return f"lora_{digest.hexdigest()[:16]}"


def fingerprint_inputs(dataset: dict, keys: list[str]) -> str:
    identity: dict[str, dict] = {}
    for key in sorted(keys):
        if key not in dataset:
            raise KeyError(f"Dataset does not contain selected key {key!r}")
        item = dataset[key]
        identity[key] = {
            field: item.get(field)
            for field in ("visible", "infrared", "depth", "query", "bbox", "width", "height")
            if field in item
        }
    return stable_json_hash(identity)


def build_run_metadata(
    *,
    mode: str,
    split: str,
    annotation_run_id: str,
    model: str,
    model_revision: str | None,
    lora_path: Path | None,
    adapter_fingerprint: str,
    prompt_hash: str,
    generation_config: dict,
    run_tag: str,
    limit: int | None,
    selected_keys: list[str],
    input_fingerprint: str,
    image_fingerprint: str,
    num_shards: int,
    base_run_id: str = "",
    base_prediction_fingerprint: str = "",
    min_pixels: int | None = MIN_PIXELS,
    max_pixels: int | None = MAX_PIXELS,
) -> dict:
    if mode not in {"base", "retry"}:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if not isinstance(annotation_run_id, str):
        raise ValueError("annotation_run_id must be a string")
    if split in {"train", "val"}:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotation_run_id):
            raise ValueError(f"Invalid annotation_run_id {annotation_run_id!r}")
    elif split == "test" and annotation_run_id:
        raise ValueError("Test inference forbids annotation_run_id")
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    identity = {
        "version": CHECKPOINT_VERSION,
        "python_version": RUNTIME_PYTHON_VERSION,
        "mode": mode,
        "split": split,
        "annotation_run_id": annotation_run_id,
        "model": model,
        "model_revision": model_revision,
        "lora_path": str(lora_path) if lora_path is not None else None,
        "adapter_fingerprint": adapter_fingerprint,
        "prompt_hash": prompt_hash,
        "generation_config": generation_config,
        "compute_dtype": INFERENCE_COMPUTE_DTYPE,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "run_tag": run_tag,
        "limit": limit,
        "selected_key_hash": key_hash(selected_keys),
        "input_fingerprint": input_fingerprint,
        "image_fingerprint": image_fingerprint,
        "num_shards": num_shards,
        "base_run_id": base_run_id,
        "base_prediction_fingerprint": base_prediction_fingerprint,
    }
    identity_hash = stable_json_hash(identity, length=16)
    run_id = f"infer_{split}_{mode}_{identity_hash}"
    metadata = {
        "version": CHECKPOINT_VERSION,
        "python_version": RUNTIME_PYTHON_VERSION,
        "run_id": run_id,
        **identity,
    }
    if tuple(metadata) != RUN_METADATA_FIELDS:
        raise AssertionError("Inference metadata schema is out of sync")
    return metadata


def build_shard_metadata(run_metadata: dict, shard_id: int, assigned_keys: list[str]) -> dict:
    if not 0 <= shard_id < run_metadata["num_shards"]:
        raise ValueError(f"Invalid shard ID {shard_id}")
    return {
        **run_metadata,
        "shard_id": shard_id,
        "assigned_key_hash": key_hash(assigned_keys),
    }


def validate_checkpoint_payload(
    payload: object,
    expected_metadata: dict,
    assigned_keys: list[str],
    *,
    require_complete: bool,
    label: str = "checkpoint",
) -> dict[str, list[float] | None]:
    if not isinstance(payload, dict) or set(payload) != {"metadata", "predictions"}:
        raise ValueError(f"{label} must contain exactly metadata and predictions")
    require_exact_metadata(payload["metadata"], expected_metadata, label=label)
    predictions = payload["predictions"]
    if not isinstance(predictions, dict):
        raise ValueError(f"{label} predictions must be an object")

    assigned = set(assigned_keys)
    actual = set(predictions)
    extra = actual - assigned
    if extra:
        raise ValueError(f"{label} contains unexpected query IDs: {sorted(extra)[:5]}")
    if require_complete and actual != assigned:
        missing = assigned - actual
        raise ValueError(f"{label} is missing query IDs: {sorted(missing)[:5]}")

    normalized: dict[str, list[float] | None] = {}
    for key, value in predictions.items():
        if value is None:
            normalized[key] = None
            continue
        bbox = validate_bbox(value)
        if bbox is None:
            raise ValueError(f"{label} contains invalid bbox for {key!r}: {value!r}")
        normalized[key] = bbox
    return normalized


def pending_keys(assigned_keys: list[str], predictions: Mapping[str, object]) -> list[str]:
    """Missing keys are interrupted work; explicit None is an attempted failure."""
    return [key for key in assigned_keys if key not in predictions]


def assign_pending_shards(
    items: list[dict],
    *,
    num_shards: int,
    existing_predictions: Mapping[str, object] | None = None,
) -> list[list[dict]]:
    """Split only unfinished items into non-empty local shard batches."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    if existing_predictions is None:
        pending = list(items)
    else:
        pending = [item for item in items if item.get("key") not in existing_predictions]
    shards = [pending[index::num_shards] for index in range(num_shards)]
    return [shard for shard in shards if shard]


def merge_shard_payloads(
    run_metadata: dict,
    shard_assignments: list[list[str]],
    payloads: list[dict],
) -> dict[str, list[float] | None]:
    if len(shard_assignments) != run_metadata["num_shards"]:
        raise ValueError("Shard assignments do not match metadata num_shards")
    if any(not assignment for assignment in shard_assignments):
        raise ValueError("Inference shard assignments must be non-empty")
    flattened = [key for assignment in shard_assignments for key in assignment]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Inference shard assignments contain duplicate query IDs")
    if key_hash(flattened) != run_metadata["selected_key_hash"]:
        raise ValueError("Inference shard assignments do not cover the selected query IDs")
    if len(payloads) != len(shard_assignments):
        raise ValueError(
            f"Expected {len(shard_assignments)} shard payloads, received {len(payloads)}"
        )
    by_shard: dict[int, dict] = {}
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError("Invalid shard payload structure")
        shard_id = payload["metadata"].get("shard_id")
        if not isinstance(shard_id, int) or shard_id in by_shard:
            raise ValueError(f"Duplicate or invalid shard ID: {shard_id!r}")
        by_shard[shard_id] = payload

    merged: dict[str, list[float] | None] = {}
    for shard_id, assigned_keys in enumerate(shard_assignments):
        if shard_id not in by_shard:
            raise ValueError(f"Missing shard payload {shard_id}")
        expected = build_shard_metadata(run_metadata, shard_id, assigned_keys)
        predictions = validate_checkpoint_payload(
            by_shard[shard_id],
            expected,
            assigned_keys,
            require_complete=True,
            label=f"shard {shard_id}",
        )
        duplicate = set(merged) & set(predictions)
        if duplicate:
            raise ValueError(f"Duplicate query IDs across shards: {sorted(duplicate)[:5]}")
        merged.update(predictions)
    return merged


def merge_retry_predictions(
    base_predictions: dict[str, list[float] | None],
    overlay: dict[str, list[float] | None],
    retry_keys: list[str],
) -> dict[str, list[float] | None]:
    allowed = set(retry_keys)
    extra = set(overlay) - allowed
    if extra:
        raise ValueError(f"Retry overlay contains non-retry IDs: {sorted(extra)[:5]}")
    result = dict(base_predictions)
    for key, value in overlay.items():
        bbox = validate_bbox(value) if value is not None else None
        if bbox is not None:
            result[key] = bbox
    return result


def predictions_are_submission_ready(
    expected_keys: list[str],
    predictions: Mapping[str, object],
) -> bool:
    """Return whether predictions exactly cover a split with valid normalized boxes."""
    if len(expected_keys) != len(set(expected_keys)):
        return False
    if set(predictions) != set(expected_keys):
        return False
    return all(validate_bbox(predictions[key]) is not None for key in expected_keys)


def evaluate_predictions(dataset: dict, keys: list[str], predictions: dict) -> dict:
    hits = 0
    total_iou = 0.0
    failures = 0
    for key in keys:
        ground_truth = validate_bbox(dataset[key].get("bbox"))
        prediction = validate_bbox(predictions.get(key))
        if ground_truth is None:
            raise ValueError(f"Evaluation sample {key!r} has invalid ground-truth bbox")
        if prediction is None:
            failures += 1
            continue
        iou = compute_iou(prediction, ground_truth)
        total_iou += iou
        hits += int(iou >= 0.5)
    total = len(keys)
    return {
        "hits": hits,
        "total": total,
        "acc_at_0_5": hits / total if total else 0.0,
        "mean_iou": total_iou / total if total else 0.0,
        "failures": failures,
    }
