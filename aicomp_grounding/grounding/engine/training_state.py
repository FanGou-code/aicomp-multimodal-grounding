"""Pure validation and scheduling helpers for compliant QLoRA training.

Data structures
---------------
Training metadata dict (from build_training_metadata):
    {protocol_version, training_run_id, annotation_run_id, train_dataset_fingerprint,
     val_dataset_fingerprint, train_image_fingerprint, val_image_fingerprint,
     annotation_prompt_hash, model_name, model_revision, grounding_prompt_hash,
     hyperparameters, seed, run_tag, train_samples, val_samples}

Epoch adapter manifest:
    {metadata: TrainingMetadata, epoch: int, val_loss: float}

Final adapter manifest:
    {metadata: TrainingMetadata, completed_epochs: int}

Completed training state (completed.json):
    {metadata, status, global_step, best_val_loss, best_path, last_path,
     best_metric, best_metric_value}

Checkpoint state (checkpoints/epoch_N/state.json):
    {metadata, completed_epoch, global_step, best_val_loss, best_path,
     train_loss, val_loss, best_metric, best_metric_value, epoch_metrics}
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from aicomp_grounding.contract import approved_dataset_fingerprint, validate_training_artifacts
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.io import load_json
from aicomp_grounding.config import TRAINING_PROTOCOL_VERSION

TRAINING_METADATA_FIELDS = (
    "protocol_version",
    "training_run_id",
    "annotation_run_id",
    "train_dataset_fingerprint",
    "val_dataset_fingerprint",
    "train_image_fingerprint",
    "val_image_fingerprint",
    "annotation_prompt_hash",
    "model_name",
    "model_revision",
    "grounding_prompt_hash",
    "hyperparameters",
    "seed",
    "run_tag",
    "train_samples",
    "val_samples",
)


def build_training_metadata(
    *,
    annotation_run_id: str,
    train_artifact: dict,
    val_artifact: dict,
    model_name: str,
    model_revision: str,
    grounding_prompt_hash: str,
    hyperparameters: dict,
    seed: int,
    run_tag: str,
) -> dict:
    if not annotation_run_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", annotation_run_id):
        raise ValueError(f"Invalid annotation_run_id {annotation_run_id!r}")
    train, val = validate_training_artifacts(
        train_artifact,
        val_artifact,
        annotation_run_id=annotation_run_id,
    )
    train_meta = train["metadata"]
    val_meta = val["metadata"]
    identity = {
        "annotation_run_id": annotation_run_id,
        "train_dataset_fingerprint": train_meta.get("dataset_fingerprint") or approved_dataset_fingerprint(train["data"]),
        "val_dataset_fingerprint": val_meta.get("dataset_fingerprint") or approved_dataset_fingerprint(val["data"]),
        "train_image_fingerprint": train_meta.get("image_fingerprint", ""),
        "val_image_fingerprint": val_meta.get("image_fingerprint", ""),
        "annotation_prompt_hash": train_meta.get("prompt_hash", ""),
        "model_name": model_name,
        "model_revision": model_revision,
        "grounding_prompt_hash": grounding_prompt_hash,
        "hyperparameters": hyperparameters,
        "seed": seed,
        "run_tag": run_tag,
    }

    training_run_id = f"train_{stable_json_hash(identity, length=16)}"
    metadata = {
        "protocol_version": TRAINING_PROTOCOL_VERSION,
        "training_run_id": training_run_id,
        **identity,
        "train_samples": len(train["data"]),
        "val_samples": len(val["data"]),
    }
    if tuple(metadata) != TRAINING_METADATA_FIELDS:
        raise AssertionError("Training metadata schema is out of sync")
    return metadata


def validate_adapter_manifest(manifest: object) -> dict:
    """Validate a publishable epoch or final adapter and its training identity."""
    if not isinstance(manifest, dict):
        raise ValueError("Adapter manifest must be a JSON object")
    fields = set(manifest)
    if fields == {"metadata", "epoch", "val_loss"}:
        epoch = manifest["epoch"]
        val_loss = manifest["val_loss"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0:
            raise ValueError("Epoch adapter manifest has an invalid epoch")
        if (
            isinstance(val_loss, bool)
            or not isinstance(val_loss, (int, float))
            or not math.isfinite(val_loss)
        ):
            raise ValueError("Epoch adapter manifest has an invalid val_loss")
    elif fields == {"metadata", "completed_epochs"}:
        completed_epochs = manifest["completed_epochs"]
        if (
            isinstance(completed_epochs, bool)
            or not isinstance(completed_epochs, int)
            or completed_epochs <= 0
        ):
            raise ValueError("Last adapter manifest has invalid completed_epochs")
    else:
        raise ValueError("Adapter manifest is not a publishable epoch/final artifact")

    metadata = manifest["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(TRAINING_METADATA_FIELDS):
        raise ValueError("Adapter training metadata schema is invalid")
    if metadata["protocol_version"] != TRAINING_PROTOCOL_VERSION:
        raise ValueError("Adapter training protocol version is not supported")
    if not re.fullmatch(r"train_[0-9a-f]{16}", metadata["training_run_id"] or ""):
        raise ValueError("Adapter training_run_id is invalid")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", metadata["annotation_run_id"] or ""):
        raise ValueError("Adapter annotation_run_id is invalid")
    for field in (
        "train_dataset_fingerprint",
        "val_dataset_fingerprint",
        "train_image_fingerprint",
        "val_image_fingerprint",
        "annotation_prompt_hash",
        "model_name",
        "model_revision",
        "grounding_prompt_hash",
    ):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"Adapter training metadata has invalid {field}")
    if not isinstance(metadata["hyperparameters"], dict) or not metadata["hyperparameters"]:
        raise ValueError("Adapter training hyperparameters are missing")
    if isinstance(metadata["seed"], bool) or not isinstance(metadata["seed"], int):
        raise ValueError("Adapter training seed is invalid")
    if not isinstance(metadata["run_tag"], str):
        raise ValueError("Adapter training run_tag is invalid")
    for field in ("train_samples", "val_samples"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Adapter training metadata has invalid {field}")
    return manifest


def build_epoch_adapter_manifest(
    metadata: dict,
    *,
    epoch: int,
    val_loss: float,
) -> dict:
    """Build the shared manifest used by best and per-epoch inference candidates."""
    manifest = {
        "metadata": metadata,
        "epoch": epoch,
        "val_loss": val_loss,
    }
    return validate_adapter_manifest(manifest)


def validate_epoch_metrics(metrics: object) -> dict:
    """Validate grounding metrics produced by an epoch validation pass."""
    if not isinstance(metrics, dict):
        raise ValueError("Epoch metrics must be a JSON object")
    required = {"hits", "total", "acc_at_0_5", "mean_iou", "failures"}
    if set(metrics) != required:
        raise ValueError("Epoch metrics schema is invalid")
    hits = metrics["hits"]
    total = metrics["total"]
    failures = metrics["failures"]
    if (
        isinstance(hits, bool)
        or not isinstance(hits, int)
        or hits < 0
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or isinstance(failures, bool)
        or not isinstance(failures, int)
        or failures < 0
        or failures > total
    ):
        raise ValueError("Epoch metrics contain invalid counts")
    for field in ("acc_at_0_5", "mean_iou"):
        value = metrics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
            or value > 1.0
        ):
            raise ValueError(f"Epoch metrics has invalid {field}")
    return metrics


def optimizer_steps_per_epoch(num_batches: int, grad_accum_steps: int) -> int:
    if num_batches <= 0 or grad_accum_steps <= 0:
        raise ValueError("num_batches and grad_accum_steps must be positive")
    return math.ceil(num_batches / grad_accum_steps)


def should_optimizer_step(batch_index: int, num_batches: int, grad_accum_steps: int) -> bool:
    if not 0 <= batch_index < num_batches:
        raise ValueError("batch_index is outside the epoch")
    return (batch_index + 1) % grad_accum_steps == 0 or batch_index + 1 == num_batches


def accumulation_window_size(
    batch_index: int,
    num_batches: int,
    grad_accum_steps: int,
) -> int:
    """Return the true divisor for this batch's accumulation window, including the tail."""
    if not 0 <= batch_index < num_batches or grad_accum_steps <= 0:
        raise ValueError("Invalid accumulation window arguments")
    window_start = (batch_index // grad_accum_steps) * grad_accum_steps
    return min(grad_accum_steps, num_batches - window_start)


def move_batch_to_device(batch: dict, device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def assert_single_cuda_device_map(device_map: dict | None) -> None:
    if not device_map:
        return
    invalid = {
        str(device)
        for device in device_map.values()
        if str(device).lower() in {"cpu", "disk"}
    }
    if invalid:
        raise RuntimeError(
            f"Model was offloaded outside the primary CUDA GPU: {sorted(invalid)}"
        )


# ---------------------------------------------------------------------------
# Adapter and checkpoint validation (shared by training_core and inference_state)
# ---------------------------------------------------------------------------

def adapter_weight_path(adapter_dir: "Path") -> "Path | None":
    """Return the first non-empty adapter weight file, or None."""
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_dir / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def expected_global_steps(
    metadata: dict,
    completed_epochs: int,
    *,
    batch_size: int,
    grad_accum_steps: int,
) -> int:
    train_samples = metadata.get("train_samples")
    if (
        isinstance(train_samples, bool)
        or not isinstance(train_samples, int)
        or train_samples <= 0
    ):
        raise ValueError("Training metadata has an invalid train_samples count")
    batches = math.ceil(train_samples / batch_size)
    return completed_epochs * optimizer_steps_per_epoch(batches, grad_accum_steps)


def validated_prompt_length(full_input_ids, prompt_input_ids, *, sample_id: str) -> int:
    if len(full_input_ids.shape) != 2 or len(prompt_input_ids.shape) != 2:
        raise ValueError(f"Training token IDs must be rank 2 for {sample_id}")
    prompt_length = int(prompt_input_ids.shape[1])
    if (
        full_input_ids.shape[0] != prompt_input_ids.shape[0]
        or prompt_length <= 0
        or prompt_length >= full_input_ids.shape[1]
    ):
        raise ValueError(f"Training label boundary is invalid for {sample_id}")
    if not bool((full_input_ids[:, :prompt_length] == prompt_input_ids).all().item()):
        raise ValueError(f"Training prompt tokens are not an exact prefix for {sample_id}")
    return prompt_length


def validate_adapter_directory(
    adapter_path: "str | Path",
    *,
    run_dir: "Path",
    metadata: dict,
    expected_kind: str,
    num_epochs: int,
) -> dict:
    """Validate an adapter directory structure and manifest within a training run."""

    if not isinstance(adapter_path, (str, Path)) or not str(adapter_path):
        raise ValueError(f"Completed training state has no {expected_kind} adapter path")
    adapter_dir = Path(adapter_path).resolve()
    resolved_run_dir = run_dir.resolve()
    try:
        relative = adapter_dir.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError(f"{expected_kind} adapter escapes the training run directory") from exc
    if expected_kind == "last":
        if relative.parts != ("last",):
            raise ValueError("Last adapter path does not point to this run's last directory")
    else:
        expected_parent = "best" if expected_kind == "best" else "checkpoints"
        if (
            expected_kind not in {"best", "checkpoint"}
            or len(relative.parts) != 2
            or relative.parts[0] != expected_parent
            or not re.fullmatch(r"epoch_\d+", relative.parts[1])
        ):
            raise ValueError(
                f"{expected_kind.capitalize()} adapter path does not point to "
                f"a {expected_parent}/epoch_N directory"
            )
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"{expected_kind.capitalize()} adapter directory is missing: {adapter_dir}")
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        raise FileNotFoundError(f"{expected_kind.capitalize()} adapter config is missing: {config_path}")
    if adapter_weight_path(adapter_dir) is None:
        raise FileNotFoundError(f"{expected_kind.capitalize()} adapter weights are missing: {adapter_dir}")
    manifest_path = adapter_dir / "adapter_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{expected_kind.capitalize()} adapter manifest is missing: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("metadata") != metadata:
        raise ValueError(f"{expected_kind.capitalize()} adapter manifest metadata mismatch")
    if expected_kind in {"best", "checkpoint"}:
        if set(manifest) != {"metadata", "epoch", "val_loss"}:
            raise ValueError(f"{expected_kind.capitalize()} adapter manifest has an invalid schema")
        epoch = manifest["epoch"]
        val_loss = manifest["val_loss"]
        if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= num_epochs:
            raise ValueError(f"{expected_kind.capitalize()} adapter manifest has an invalid epoch")
        if relative.parts[1] != f"epoch_{epoch:02d}":
            raise ValueError(f"{expected_kind.capitalize()} adapter directory and manifest epoch disagree")
        if (
            isinstance(val_loss, bool)
            or not isinstance(val_loss, (int, float))
            or not math.isfinite(val_loss)
        ):
            raise ValueError(f"{expected_kind.capitalize()} adapter manifest has an invalid val_loss")
    else:
        if set(manifest) != {"metadata", "completed_epochs"}:
            raise ValueError("Last adapter manifest has an invalid schema")
        if manifest["completed_epochs"] != num_epochs:
            raise ValueError("Last adapter manifest does not cover every training epoch")
    return manifest


def validate_completed_training_state(
    completed: object,
    metadata: dict,
    run_dir: "Path",
    *,
    batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
) -> dict:
    required = {
        "metadata",
        "status",
        "global_step",
        "best_val_loss",
        "best_path",
        "last_path",
        "best_metric",
        "best_metric_value",
    }
    if not isinstance(completed, dict) or set(completed) != required:
        raise ValueError("Completed training state has an invalid schema")
    if completed["metadata"] != metadata or completed["status"] != "completed":
        raise ValueError("Completed training metadata or status does not match")
    expected = expected_global_steps(
        metadata, num_epochs, batch_size=batch_size, grad_accum_steps=grad_accum_steps
    )
    if completed["global_step"] != expected:
        raise ValueError(
            f"Completed training global_step={completed['global_step']} does not match expected {expected}"
        )
    if (
        isinstance(completed["best_val_loss"], bool)
        or not isinstance(completed["best_val_loss"], (int, float))
        or not math.isfinite(completed["best_val_loss"])
    ):
        raise ValueError("Completed training state has an invalid best_val_loss")
    best_manifest = validate_adapter_directory(
        completed["best_path"], run_dir=run_dir, metadata=metadata,
        expected_kind="best", num_epochs=num_epochs,
    )
    validate_adapter_directory(
        completed["last_path"], run_dir=run_dir, metadata=metadata,
        expected_kind="last", num_epochs=num_epochs,
    )
    if best_manifest["val_loss"] != completed["best_val_loss"]:
        raise ValueError("Completed training state and best adapter val_loss disagree")
    if not isinstance(completed["best_metric"], str) or not completed["best_metric"]:
        raise ValueError("Completed training state has an invalid best_metric")
    best_metric_value = completed["best_metric_value"]
    if (
        isinstance(best_metric_value, bool)
        or not isinstance(best_metric_value, (int, float))
        or not math.isfinite(best_metric_value)
    ):
        raise ValueError("Completed training state has an invalid best_metric_value")
    return completed


def validate_resume_checkpoint(
    checkpoint_path: "Path",
    *,
    run_dir: "Path",
    metadata: dict,
    batch_size: int,
    grad_accum_steps: int,
    num_epochs: int,
) -> dict:

    checkpoint = Path(checkpoint_path).resolve()
    checkpoint_root = (run_dir / "checkpoints").resolve()
    try:
        relative = checkpoint.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ValueError("Training checkpoint escapes the current run directory") from exc
    match_epoch = re.fullmatch(r"epoch_(\d+)", relative.as_posix())
    match_step = re.fullmatch(r"step_(\d+)", relative.as_posix())
    if match_epoch is None and match_step is None:
        raise ValueError(f"Invalid training checkpoint path: {checkpoint}")

    if not (checkpoint / "adapter_config.json").is_file() or adapter_weight_path(checkpoint) is None:
        raise FileNotFoundError(f"Training checkpoint adapter is incomplete: {checkpoint}")
    binary_path = checkpoint / "training_state.pt"
    if not binary_path.is_file() or binary_path.stat().st_size == 0:
        raise FileNotFoundError(f"Training optimizer/RNG state is missing: {binary_path}")
    state = load_json(checkpoint / "state.json")
    if state.get("metadata") != metadata:
        raise ValueError("Training checkpoint state metadata does not match")

    if match_epoch:
        epoch = int(match_epoch.group(1))
        if not 1 <= epoch <= num_epochs:
            raise ValueError(f"Training checkpoint epoch is outside 1..{num_epochs}: {epoch}")
        checkpoint_manifest = validate_adapter_directory(
            checkpoint, run_dir=run_dir, metadata=metadata,
            expected_kind="checkpoint", num_epochs=num_epochs,
        )
        required = {
            "metadata",
            "completed_epoch",
            "global_step",
            "best_val_loss",
            "best_path",
            "train_loss",
            "val_loss",
            "best_metric",
            "best_metric_value",
            "epoch_metrics",
        }
        if set(state) != required:
            raise ValueError("Training checkpoint state schema does not match")
        if state["completed_epoch"] != epoch:
            raise ValueError("Training checkpoint path and completed_epoch disagree")
        expected_steps = expected_global_steps(
            metadata, epoch, batch_size=batch_size, grad_accum_steps=grad_accum_steps
        )
        if state["global_step"] != expected_steps:
            raise ValueError(
                f"Training checkpoint global_step={state['global_step']} does not match expected {expected_steps}"
            )
        for field in ("best_val_loss", "train_loss", "val_loss"):
            value = state[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Training checkpoint has invalid {field}")
        validate_epoch_metrics(state["epoch_metrics"])
        if not isinstance(state["best_metric"], str) or not state["best_metric"]:
            raise ValueError("Training checkpoint has an invalid best_metric")
        if (
            isinstance(state["best_metric_value"], bool)
            or not isinstance(state["best_metric_value"], (int, float))
            or not math.isfinite(state["best_metric_value"])
        ):
            raise ValueError("Training checkpoint has an invalid best_metric_value")
        if checkpoint_manifest["val_loss"] != state["val_loss"]:
            raise ValueError("Training checkpoint state and adapter manifest val_loss disagree")
        best_manifest = validate_adapter_directory(
            state["best_path"], run_dir=run_dir, metadata=metadata,
            expected_kind="best", num_epochs=num_epochs,
        )
        if best_manifest["epoch"] > epoch or best_manifest["val_loss"] != state["best_val_loss"]:
            raise ValueError("Training checkpoint and best adapter manifest disagree")
    else:
        step = int(match_step.group(1))
        if state.get("global_step") != step:
            raise ValueError(f"Training checkpoint path step={step} and global_step={state.get('global_step')} disagree")
        # Step-level checkpoints are valid resume sources.  The training
        # loop uses batch_index to skip deterministic batches and
        # restore RNG / optimizer state so double-training is avoided.
        step_epoch = state.get("completed_epoch")
        batch_index = state.get("batch_index", 0)
        if not isinstance(step_epoch, int) or isinstance(step_epoch, bool):
            raise ValueError("Step checkpoint state has an invalid completed_epoch")
        if not isinstance(batch_index, int) or isinstance(batch_index, bool) or batch_index < 0:
            raise ValueError("Step checkpoint state has an invalid batch_index")
        steps_per_epoch = optimizer_steps_per_epoch(
            math.ceil(metadata["train_samples"] / batch_size), grad_accum_steps
        )
        if not (step_epoch * steps_per_epoch < step <= (step_epoch + 1) * steps_per_epoch):
            raise ValueError(
                f"Step checkpoint step={step} is outside epoch {step_epoch} "
                f"(steps {step_epoch * steps_per_epoch}..{(step_epoch + 1) * steps_per_epoch})"
            )
    return state


def validate_loaded_training_state(
    binary_state: object,
    state: dict,
    metadata: dict,
) -> dict:
    """Validate that a loaded checkpoint binary state belongs to this run.

    Only checks training_run_id and global_step alignment. Deeper optimizer/scheduler
    structure validation is delegated to PyTorch's load_state_dict.
    """
    required = {"optimizer", "scheduler", "torch_rng_state", "cuda_rng_state",
                "completed_epoch", "global_step", "training_run_id"}
    if not isinstance(binary_state, dict) or set(binary_state) != required:
        raise ValueError("Training optimizer/RNG state has an invalid schema")
    if binary_state["completed_epoch"] != state["completed_epoch"]:
        raise ValueError("Binary training state completed_epoch does not match state.json")
    if binary_state["global_step"] != state["global_step"]:
        raise ValueError("Binary training state global_step does not match state.json")
    if binary_state["training_run_id"] != metadata["training_run_id"]:
        raise ValueError("Binary training state belongs to a different training run")
    if not isinstance(binary_state["optimizer"], dict) or not isinstance(binary_state["scheduler"], dict):
        raise ValueError("Binary optimizer or scheduler state is invalid")
    return binary_state
