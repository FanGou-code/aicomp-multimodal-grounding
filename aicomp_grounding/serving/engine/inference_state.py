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
     total_predictions, valid_predictions}
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
from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.config import (
    CHECKPOINT_VERSION,
    INFERENCE_COMPUTE_DTYPE,
    INFERENCE_DEFAULT_MAX_PIXELS,
    INFERENCE_DEFAULT_MIN_PIXELS,
    INFERENCE_SPLITS,
    RUNTIME_PYTHON_VERSION,
)
from aicomp_grounding.io import load_json
from aicomp_grounding.serving.engine.training_state import adapter_weight_path

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
    min_pixels: int | None = INFERENCE_DEFAULT_MIN_PIXELS,
    max_pixels: int | None = INFERENCE_DEFAULT_MAX_PIXELS,
    compute_dtype: str | None = None,
) -> dict:
    if mode not in {"base", "retry"}:
        raise ValueError(f"Unsupported inference mode: {mode}")
    if split not in INFERENCE_SPLITS:
        raise ValueError(f"Unsupported inference split: {split!r}")
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
        "compute_dtype": compute_dtype if compute_dtype is not None else INFERENCE_COMPUTE_DTYPE,
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
    # lora_path is recorded for traceability but excluded from the identity
    # hash: it is a host-specific absolute path, so including it would give the
    # same adapter a different run_id on every machine (breaking resume and
    # cross-environment reproducibility). adapter_fingerprint already pins the
    # adapter's content.
    identity_hash = stable_json_hash(
        {key: value for key, value in identity.items() if key != "lora_path"},
        length=16,
    )
    if mode == "retry":
        run_id = f"infer_{identity_hash}_retry"
    else:
        run_id = f"infer_{identity_hash}"
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
    if not isinstance(payload, dict) or not (
        {"metadata", "predictions"} <= set(payload)
        <= {"metadata", "predictions", "assigned_keys"}
    ):
        raise ValueError(f"{label} must contain metadata and predictions")
    actual_metadata = payload["metadata"]
    if not isinstance(actual_metadata, dict):
        raise ValueError(f"{label} metadata must be an object")
    # The adapter content fingerprint, not its machine-specific directory,
    # identifies weights. Keep the original path as trace information only.
    require_exact_metadata(
        {k: v for k, v in actual_metadata.items() if k != "lora_path"},
        {k: v for k, v in expected_metadata.items() if k != "lora_path"},
        label=label,
    )
    if set(actual_metadata) != set(expected_metadata):
        raise ValueError(f"{label} metadata fields do not match")
    if "assigned_keys" in payload:
        recorded_keys = payload["assigned_keys"]
        if not isinstance(recorded_keys, list) or key_hash(recorded_keys) != key_hash(assigned_keys):
            raise ValueError(f"{label} assigned keys do not match")
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


def load_resume_predictions(
    run_dir: Path, metadata: dict, selected_keys: list[str],
) -> dict[str, list[float] | None]:
    """Validate every saved source before reconciling interrupted inference."""
    predictions: dict[str, list[float] | None] = {}

    def merge(payload, expected, assigned, label):
        checked = validate_checkpoint_payload(
            payload, expected, assigned, require_complete=False, label=label,
        )
        for key, value in checked.items():
            if key in predictions and predictions[key] != value:
                raise ValueError(f"Conflicting resumed prediction for {key!r}: {label}")
            predictions[key] = value

    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = load_json(checkpoint_path) if checkpoint_path.is_file() else None
    if checkpoint is not None:
        merge(checkpoint, metadata, selected_keys, str(checkpoint_path))

    metadata_path = run_dir / "metadata.json"
    saved_metadata = load_json(metadata_path) if metadata_path.is_file() else None
    if saved_metadata is not None:
        validate_checkpoint_payload(
            {"metadata": saved_metadata, "predictions": {}}, metadata, selected_keys,
            require_complete=False, label=str(metadata_path),
        )
    elif checkpoint is not None:
        saved_metadata = checkpoint["metadata"]
    predictions_path = run_dir / "predictions.json"
    if predictions_path.is_file():
        if saved_metadata is None:
            raise ValueError(f"Cannot resume {predictions_path} without recorded metadata")
        merge(
            {"metadata": saved_metadata, "predictions": load_json(predictions_path)},
            metadata, selected_keys, str(predictions_path),
        )

    for path in sorted((run_dir / "shard_checkpoints").glob("shard_*.checkpoint.json")):
        payload = load_json(path)
        recorded = payload.get("metadata", {})
        match = re.fullmatch(r"shard_(\d+)\.checkpoint\.json", path.name)
        if match is None or not isinstance(recorded, dict):
            raise ValueError(f"Invalid shard checkpoint: {path}")
        shard_id = int(match[1])
        if not 0 <= shard_id < metadata["num_shards"]:
            raise ValueError(f"Unexpected shard ID in {path}")
        assigned = payload.get("assigned_keys")
        if assigned is None:
            # Legacy files did not store assignments. Validate their full run
            # identity and every result key; do not invent a new assignment hash.
            digest = recorded.get("assigned_key_hash")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid legacy shard assignment hash: {path}")
            expected = {**metadata, "shard_id": shard_id, "assigned_key_hash": digest}
            assigned = selected_keys
        else:
            if not isinstance(assigned, list) or not assigned:
                raise ValueError(f"Invalid shard assignment: {path}")
            key_hash(assigned)  # validates string IDs and uniqueness
            if set(assigned) - set(selected_keys):
                raise ValueError(f"Shard assignment contains unknown query IDs: {path}")
            expected = build_shard_metadata(metadata, shard_id, assigned)
        merge(payload, expected, assigned, str(path))
    return predictions


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
