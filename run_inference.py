"""Modal Qwen3-VL inference with strict preflight, sharding, Resume, and Retry."""

from __future__ import annotations

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import modal

from aicomp_grounding.annotation_state import validate_approved_artifact
from aicomp_grounding.artifacts import key_hash, require_exact_metadata, stable_json_hash
from aicomp_grounding.bbox import parse_bbox_from_text, validate_bbox
from aicomp_grounding.config import (
    CHECKPOINT_VERSION,
    DATA_ROOT,
    INFERENCE_COMPUTE_DTYPE,
    INFERENCE_SPLITS,
    MAX_MODAL_CONTAINERS,
    MAX_PIXELS,
    MIN_PIXELS,
    MODAL_GPU_PACKAGES,
    MODEL_NAME,
    MODEL_REVISION,
)
from aicomp_grounding.inference_state import (
    RUN_METADATA_FIELDS,
    build_run_metadata,
    build_shard_metadata,
    evaluate_predictions,
    fingerprint_inputs,
    fingerprint_lora,
    merge_retry_predictions,
    merge_shard_payloads,
    pending_keys,
    predictions_are_submission_ready,
    resolve_lora_path,
    validate_checkpoint_payload,
)
from aicomp_grounding.images import (
    is_trusted_image_fingerprint,
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    RETRY_GROUNDING_SYSTEM_PROMPT,
    build_grounding_messages,
    build_retry_grounding_messages,
    grounding_prompt_hash,
)
from aicomp_grounding.sharding import shard_keys_by_scene
from aicomp_grounding.test_data import (
    validate_processed_test_index,
    validate_test_preparation_manifest,
)
from aicomp_grounding.training_state import (
    build_training_metadata,
    validate_adapter_manifest,
    validate_training_artifacts,
)

dataset_volume = modal.Volume.from_name("rgbdt-dataset")
model_volume = modal.Volume.from_name("hf-model-cache")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*MODAL_GPU_PACKAGES)
    .add_local_python_source("aicomp_grounding")
)

app = modal.App("rgbdt-vg-inference", image=image)


def _validate_options(
    split: str,
    annotation_run_id: str,
    limit: int | None,
    num_shards: int,
    retry_failed: bool,
    base_run_id: str,
) -> str:
    normalized = split.lower().strip()
    if normalized not in INFERENCE_SPLITS:
        raise ValueError(f"Unsupported inference split {split!r}")
    if not isinstance(annotation_run_id, str):
        raise ValueError("annotation_run_id must be a string")
    if normalized in {"train", "val"}:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotation_run_id):
            raise ValueError(
                "Train/val inference requires a safe, non-empty annotation_run_id"
            )
    elif annotation_run_id:
        raise ValueError("Test inference forbids annotation_run_id")
    if isinstance(limit, bool) or (
        limit is not None and (not isinstance(limit, int) or limit <= 0)
    ):
        raise ValueError(f"limit must be a positive integer or None, got {limit!r}")
    if not 1 <= num_shards <= MAX_MODAL_CONTAINERS:
        raise ValueError(
            f"num_shards must be between 1 and {MAX_MODAL_CONTAINERS}, got {num_shards}"
        )
    if retry_failed and not base_run_id.strip():
        raise ValueError("Retry requires the exact base_run_id from a completed inference run")
    if base_run_id and not re.fullmatch(r"[A-Za-z0-9_.-]+", base_run_id):
        raise ValueError(f"Invalid base_run_id {base_run_id!r}")
    if retry_failed and num_shards != 1:
        raise ValueError("Retry is deliberately single-shard; set num_shards=1")
    return normalized


def _prompt_hash(retry_failed: bool) -> str:
    prompt = RETRY_GROUNDING_SYSTEM_PROMPT if retry_failed else GROUNDING_SYSTEM_PROMPT
    return grounding_prompt_hash(prompt)


def _generation_config(retry_failed: bool, retry_seed: int) -> dict:
    if retry_failed:
        return {
            "max_new_tokens": 32,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": retry_seed,
            "seed_strategy": "sha256-base-seed-and-query-id-v1",
        }
    return {"max_new_tokens": 32, "do_sample": False}


def _retry_query_seed(base_seed: int, query_id: str) -> int:
    seed_bytes = hashlib.sha256(f"{base_seed}:{query_id}".encode("utf-8")).digest()
    return int.from_bytes(seed_bytes[:8], "big") & ((1 << 63) - 1)


def _load_dataset(data_root: Path, split: str, annotation_run_id: str) -> dict:
    if split in {"train", "val"}:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", annotation_run_id or ""):
            raise ValueError(
                "Train/val inference requires a safe, non-empty annotation_run_id"
            )
        path = (
            data_root
            / "outputs"
            / "annotations"
            / annotation_run_id
            / split
            / "approved.json"
        )
    elif split == "test":
        if annotation_run_id:
            raise ValueError("Test inference forbids annotation_run_id")
        path = data_root / "test.json"
    else:
        raise ValueError(f"Unsupported inference split {split!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Dataset index not found: {path}")
    loaded = load_json(path)
    if split in {"train", "val"}:
        artifact = validate_approved_artifact(
            loaded,
            expected_split=split,
            expected_run_id=annotation_run_id,
        )
        data = artifact["data"]
    else:
        data = loaded
    if not data:
        raise ValueError(f"Dataset index is empty: {path}")
    for query_id, item in data.items():
        if not isinstance(query_id, str) or not isinstance(item, dict):
            raise ValueError(f"Invalid dataset entry {query_id!r}")
        for field in ("visible", "infrared", "depth", "query"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"Dataset entry {query_id!r} has invalid {field!r}")
    return data


def _verify_selected_images(data_root: Path, dataset: dict, keys: list[str]) -> str:
    """Decode each unique image once on CPU before any GPU is requested."""
    return verify_dataset_images(
        data_root,
        dataset,
        keys,
        require_recorded_size=False,
    )


def _selected_image_fingerprint(
    data_root: Path,
    dataset: dict,
    keys: list[str],
    *,
    deep_verify: bool,
    expected_fingerprint: str | None = None,
) -> str:
    trusted = trusted_dataset_image_fingerprint(
        dataset,
        keys,
        require_recorded_size=False,
    )
    if expected_fingerprint is not None and is_trusted_image_fingerprint(
        expected_fingerprint
    ):
        if deep_verify:
            _verify_selected_images(data_root, dataset, keys)
        return trusted
    if deep_verify or expected_fingerprint is not None:
        return _verify_selected_images(data_root, dataset, keys)
    return trusted


def _validate_base_checkpoint(
    payload: dict,
    *,
    base_run_id: str,
    split: str,
    annotation_run_id: str,
    dataset: dict,
    current_image_fingerprint: str,
) -> dict[str, list[float] | None]:
    if not isinstance(payload, dict) or set(payload) != {"metadata", "predictions"}:
        raise ValueError("Base checkpoint must contain exactly metadata and predictions")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict) or set(metadata) != set(RUN_METADATA_FIELDS):
        raise ValueError("Base checkpoint has an incompatible metadata schema")
    if metadata["run_id"] != base_run_id or metadata["mode"] != "base":
        raise ValueError("Retry base checkpoint identity does not match base_run_id")
    if metadata["split"] != split:
        raise ValueError(
            f"Retry split {split!r} does not match base split {metadata['split']!r}"
        )
    if metadata["annotation_run_id"] != annotation_run_id:
        raise ValueError("Retry annotation run does not match the base checkpoint")
    if metadata["model"] != MODEL_NAME or metadata["model_revision"] != MODEL_REVISION:
        raise ValueError("Retry base checkpoint uses a different model revision")
    if metadata["version"] != CHECKPOINT_VERSION:
        raise ValueError("Retry base checkpoint uses a different protocol version")
    if metadata["min_pixels"] != MIN_PIXELS or metadata["max_pixels"] != MAX_PIXELS:
        raise ValueError("Retry base checkpoint uses different image pixel limits")
    if metadata["prompt_hash"] != _prompt_hash(False):
        raise ValueError("Retry base checkpoint uses a different grounding prompt")
    if metadata["generation_config"] != _generation_config(False, 0):
        raise ValueError("Retry base checkpoint uses a different generation configuration")

    predictions = payload["predictions"]
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("Retry base checkpoint has no predictions")
    keys = sorted(predictions)
    declared_limit = metadata["limit"]
    if isinstance(declared_limit, bool) or (
        declared_limit is not None
        and (not isinstance(declared_limit, int) or declared_limit <= 0)
    ):
        raise ValueError("Retry base checkpoint has an invalid limit")
    if not set(keys) <= set(dataset):
        raise ValueError("Retry base checkpoint contains IDs outside the current dataset")
    all_keys = sorted(dataset, key=lambda query_id: (dataset[query_id]["visible"], query_id))
    expected_keys = all_keys if declared_limit is None else all_keys[:declared_limit]
    if set(keys) != set(expected_keys):
        raise ValueError("Retry base checkpoint does not cover its declared dataset selection")
    if metadata["selected_key_hash"] != key_hash(keys):
        raise ValueError("Retry base checkpoint selected-key hash does not match its predictions")
    current_fingerprint = fingerprint_inputs(dataset, keys)
    if metadata["input_fingerprint"] != current_fingerprint:
        raise ValueError("Dataset inputs changed since the base inference run")
    if metadata["image_fingerprint"] != current_image_fingerprint:
        raise ValueError("Dataset image bytes changed since the base inference run")
    expected_metadata = build_run_metadata(
        mode="base",
        split=split,
        annotation_run_id=metadata["annotation_run_id"],
        model=MODEL_NAME,
        model_revision=MODEL_REVISION,
        lora_path=Path(metadata["lora_path"]) if metadata["lora_path"] else None,
        adapter_fingerprint=metadata["adapter_fingerprint"],
        prompt_hash=_prompt_hash(False),
        generation_config=_generation_config(False, 0),
        run_tag=metadata["run_tag"],
        limit=metadata["limit"],
        selected_keys=keys,
        input_fingerprint=current_fingerprint,
        image_fingerprint=current_image_fingerprint,
        num_shards=metadata["num_shards"],
        base_run_id="",
        base_prediction_fingerprint="",
    )
    if metadata != expected_metadata:
        raise ValueError("Retry base checkpoint metadata identity is not self-consistent")

    normalized: dict[str, list[float] | None] = {}
    for key, value in predictions.items():
        if value is None:
            normalized[key] = None
        else:
            bbox = validate_bbox(value)
            if bbox is None:
                raise ValueError(f"Base checkpoint contains invalid bbox for {key!r}")
            normalized[key] = bbox
    return normalized


def _validate_adapter_provenance(
    data_root: Path,
    adapter_dir: Path | None,
    *,
    split: str,
    annotation_run_id: str,
) -> dict | None:
    """Bind every LoRA inference run to this pipeline's approved self-hosted data."""
    if adapter_dir is None:
        return None
    manifest = validate_adapter_manifest(load_json(adapter_dir / "adapter_manifest.json"))
    metadata = manifest["metadata"]
    adapter_annotation_run_id = metadata["annotation_run_id"]
    if split in {"train", "val"} and adapter_annotation_run_id != annotation_run_id:
        raise ValueError(
            "LoRA adapter annotation run does not match the evaluation annotation run"
        )

    annotation_root = data_root / "outputs" / "annotations" / adapter_annotation_run_id
    train_path = annotation_root / "train" / "approved.json"
    val_path = annotation_root / "val" / "approved.json"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError(
            "LoRA provenance requires its approved train and val artifacts: "
            f"{train_path}, {val_path}"
        )
    train_artifact, val_artifact = validate_training_artifacts(
        load_json(train_path),
        load_json(val_path),
        annotation_run_id=adapter_annotation_run_id,
    )
    rebuilt = build_training_metadata(
        annotation_run_id=adapter_annotation_run_id,
        train_artifact=train_artifact,
        val_artifact=val_artifact,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
        grounding_prompt_hash=_prompt_hash(False),
        hyperparameters=metadata["hyperparameters"],
        seed=metadata["seed"],
        run_tag=metadata["run_tag"],
    )
    require_exact_metadata(metadata, rebuilt, label="LoRA adapter provenance")
    return metadata


def prepare_inference_plan(
    *,
    data_root: str | Path,
    split: str = "val",
    annotation_run_id: str = "",
    lora_path: str | None = None,
    limit: int | None = None,
    run_tag: str = "",
    retry_failed: bool = False,
    base_run_id: str = "",
    num_shards: int = 1,
    retry_seed: int = 42,
    resume: bool = True,
    verify_images: bool = True,
    validate_committed_volume: bool = False,
) -> dict:
    """Filesystem-only preflight used by the Modal CPU wrapper and offline tests."""
    split = split.lower().strip()
    if split not in INFERENCE_SPLITS:
        raise ValueError(f"Unsupported inference split {split!r}")
    root = Path(data_root).resolve()
    dataset = _load_dataset(root, split, annotation_run_id)
    if split == "test" and (verify_images or validate_committed_volume):
        official_path = root / "Test" / "queries" / "queries.json"
        if not official_path.is_file():
            raise FileNotFoundError(f"Official Test template not found: {official_path}")
        official = load_json(official_path)
        validate_processed_test_index(dataset, official)
        manifest_path = root / "split_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Test inference requires split_manifest.json from a completed formal preprocessing run"
            )
        validate_test_preparation_manifest(
            root,
            load_json(manifest_path),
            dataset,
            official,
            verify_file_bytes=verify_images,
        )
    all_keys = sorted(dataset, key=lambda query_id: (dataset[query_id]["visible"], query_id))
    base_predictions: dict[str, list[float] | None] = {}

    if retry_failed:
        base_path = root / "outputs" / "inference" / base_run_id / "checkpoint.json"
        if not base_path.is_file():
            raise FileNotFoundError(f"Retry base checkpoint not found: {base_path}")
        base_payload = load_json(base_path)
        raw_base_predictions = base_payload.get("predictions")
        if not isinstance(raw_base_predictions, dict):
            raise ValueError("Base checkpoint predictions must be an object")
        base_keys = sorted(raw_base_predictions)
        recorded_base_fingerprint = base_payload.get("metadata", {}).get(
            "image_fingerprint"
        )
        base_image_fingerprint = _selected_image_fingerprint(
            root,
            dataset,
            base_keys,
            deep_verify=verify_images,
            expected_fingerprint=recorded_base_fingerprint,
        )
        base_predictions = _validate_base_checkpoint(
            base_payload,
            base_run_id=base_run_id,
            split=split,
            annotation_run_id=annotation_run_id,
            dataset=dataset,
            current_image_fingerprint=base_image_fingerprint,
        )
        base_metadata = base_payload["metadata"]
        effective_lora = lora_path or base_metadata["lora_path"]
        adapter_dir = resolve_lora_path(effective_lora, root)
        if lora_path and str(adapter_dir) != base_metadata["lora_path"]:
            raise ValueError("Explicit Retry LoRA path does not match the base checkpoint")
        _validate_adapter_provenance(
            root,
            adapter_dir,
            split=split,
            annotation_run_id=annotation_run_id,
        )
        adapter_fingerprint = fingerprint_lora(adapter_dir)
        if adapter_fingerprint != base_metadata["adapter_fingerprint"]:
            raise ValueError("LoRA adapter content changed since the base inference run")
        keys = sorted(key for key, value in base_predictions.items() if value is None)
        if limit is not None:
            keys = keys[:limit]
    else:
        adapter_dir = resolve_lora_path(lora_path, root)
        _validate_adapter_provenance(
            root,
            adapter_dir,
            split=split,
            annotation_run_id=annotation_run_id,
        )
        adapter_fingerprint = fingerprint_lora(adapter_dir)
        keys = all_keys if limit is None else all_keys[:limit]

    if not keys:
        shards: list[list[str]] = []
        effective_shards = 1
    else:
        shards = shard_keys_by_scene(keys, dataset, num_shards)
        effective_shards = len(shards)
    image_fingerprint = _selected_image_fingerprint(
        root,
        dataset,
        keys,
        deep_verify=verify_images,
    )

    metadata = build_run_metadata(
        mode="retry" if retry_failed else "base",
        split=split,
        annotation_run_id=annotation_run_id,
        model=MODEL_NAME,
        model_revision=MODEL_REVISION,
        lora_path=adapter_dir,
        adapter_fingerprint=adapter_fingerprint,
        prompt_hash=_prompt_hash(retry_failed),
        generation_config=_generation_config(retry_failed, retry_seed),
        run_tag=run_tag,
        limit=limit,
        selected_keys=keys,
        input_fingerprint=fingerprint_inputs(dataset, keys),
        image_fingerprint=image_fingerprint,
        num_shards=effective_shards,
        base_run_id=base_run_id,
        base_prediction_fingerprint=(
            stable_json_hash(base_predictions) if retry_failed else ""
        ),
    )
    result_keys = sorted(base_predictions) if retry_failed else list(keys)
    result_is_full_split = set(result_keys) == set(all_keys)
    completed_payloads: list[dict] = []
    pending_shard_ids = list(range(len(shards)))
    if resume:
        pending_shard_ids = []
        for shard_id, assigned_keys in enumerate(shards):
            expected_metadata = build_shard_metadata(metadata, shard_id, assigned_keys)
            checkpoint_path = (
                root
                / "outputs"
                / "inference"
                / metadata["run_id"]
                / "shards"
                / f"shard_{shard_id:02d}.json"
            )
            if not checkpoint_path.is_file():
                pending_shard_ids.append(shard_id)
                continue
            payload = load_json(checkpoint_path)
            predictions = validate_checkpoint_payload(
                payload,
                expected_metadata,
                assigned_keys,
                require_complete=False,
                label=f"shard {shard_id} checkpoint",
            )
            if set(predictions) == set(assigned_keys):
                completed_payloads.append(payload)
            else:
                pending_shard_ids.append(shard_id)
    return {
        "metadata": metadata,
        "keys": keys,
        "shards": shards,
        "result_keys": result_keys,
        "all_key_hash": key_hash(all_keys),
        "result_is_full_split": result_is_full_split,
        "base_predictions": base_predictions,
        "completed_payloads": completed_payloads,
        "pending_shard_ids": pending_shard_ids,
    }


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes={"/data": dataset_volume})
def preflight_inference_environment(
    split: str = "val",
    annotation_run_id: str = "",
    lora_path: str | None = None,
    limit: int | None = None,
    run_tag: str = "",
    retry_failed: bool = False,
    base_run_id: str = "",
    num_shards: int = 1,
    retry_seed: int = 42,
    resume: bool = True,
    deep_verify_images: bool = False,
) -> dict:
    dataset_volume.reload()
    plan = prepare_inference_plan(
        data_root=DATA_ROOT,
        split=split,
        annotation_run_id=annotation_run_id,
        lora_path=lora_path,
        limit=limit,
        run_tag=run_tag,
        retry_failed=retry_failed,
        base_run_id=base_run_id,
        num_shards=num_shards,
        retry_seed=retry_seed,
        resume=resume,
        verify_images=deep_verify_images,
        validate_committed_volume=True,
    )
    plan_path = (
        Path(DATA_ROOT)
        / "outputs"
        / "inference"
        / plan["metadata"]["run_id"]
        / "plan.json"
    )
    transient_fields = {"base_predictions", "completed_payloads", "pending_shard_ids"}
    persisted_plan = {key: value for key, value in plan.items() if key not in transient_fields}
    if plan_path.is_file():
        if load_json(plan_path) != persisted_plan:
            raise ValueError(f"Inference plan mismatch at {plan_path}")
    else:
        atomic_write_json(plan_path, persisted_plan)
        dataset_volume.commit()
    return plan


def _load_model_and_processor(lora_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if INFERENCE_COMPUTE_DTYPE != "bfloat16":
        raise ValueError(f"Unsupported inference compute dtype: {INFERENCE_COMPUTE_DTYPE}")
    compute_dtype = torch.bfloat16
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        quantization_config=None,
        device_map="auto",
        dtype=compute_dtype,
        attn_implementation="sdpa",
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model.eval()
    return model, processor


def _execute_inference_batch(
    *,
    model,
    processor,
    dataset: dict,
    assigned_keys: list[str],
    predictions: dict[str, list[float] | None],
    metadata: dict,
    checkpoint_path: Path,
    checkpoint_every: int = 25,
) -> dict[str, list[float] | None]:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    retry_failed = metadata["mode"] == "retry"
    generation_config = dict(metadata["generation_config"])
    retry_seed = generation_config.pop("seed", None)
    seed_strategy = generation_config.pop("seed_strategy", None)
    scene_cache: OrderedDict[tuple[str, str, str], tuple] = OrderedDict()

    def save_checkpoint() -> None:
        atomic_write_json(
            checkpoint_path,
            {"metadata": metadata, "predictions": predictions},
        )
        dataset_volume.commit()

    def load_images(item: dict):
        paths = tuple(str(Path(DATA_ROOT) / item[field]) for field in ("visible", "infrared", "depth"))
        cached = scene_cache.pop(paths, None)
        if cached is not None:
            scene_cache[paths] = cached
            return cached
        images = []
        for path in paths:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        result = tuple(images)
        scene_cache[paths] = result
        if len(scene_cache) > 32:
            scene_cache.popitem(last=False)
        return result

    todo = pending_keys(assigned_keys, predictions)
    try:
        for offset, query_id in enumerate(todo, start=1):
            item = dataset[query_id]
            try:
                visible, infrared, depth = load_images(item)
            except OSError as exc:
                print(f"[{offset}/{len(todo)}] {query_id}: image error: {exc}", flush=True)
                predictions[query_id] = None
                continue

            messages = (
                build_retry_grounding_messages(visible, infrared, depth, item["query"])
                if retry_failed
                else build_grounding_messages(visible, infrared, depth, item["query"])
            )
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            if retry_seed is not None:
                if seed_strategy != "sha256-base-seed-and-query-id-v1":
                    raise ValueError(f"Unsupported Retry seed strategy: {seed_strategy!r}")
                query_seed = _retry_query_seed(retry_seed, query_id)
                torch.manual_seed(query_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(query_seed)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **generation_config)
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            response = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            predictions[query_id] = parse_bbox_from_text(response)
            print(
                f"[{offset}/{len(todo)}] {query_id}: "
                f"{predictions[query_id] if predictions[query_id] else 'parse failed'}",
                flush=True,
            )
            if offset % checkpoint_every == 0:
                save_checkpoint()
    except Exception:
        save_checkpoint()
        raise

    save_checkpoint()
    return predictions


@app.function(
    gpu="L40S",
    cpu=4.0,
    memory=32768,
    max_containers=MAX_MODAL_CONTAINERS,
    timeout=43200,
    volumes={
        "/data": dataset_volume,
        "/root/.cache/huggingface": model_volume,
    },
)
def run_inference_shard(
    shard_id: int,
    assigned_keys: list[str],
    selected_keys: list[str],
    run_metadata: dict,
    resume: bool = True,
) -> dict:
    dataset_volume.reload()
    expected_metadata = build_shard_metadata(run_metadata, shard_id, assigned_keys)
    if key_hash(selected_keys) != run_metadata["selected_key_hash"]:
        raise ValueError("Worker selected keys do not match the authoritative run plan")

    dataset = _load_dataset(
        Path(DATA_ROOT),
        run_metadata["split"],
        run_metadata["annotation_run_id"],
    )
    if fingerprint_inputs(dataset, selected_keys) != run_metadata["input_fingerprint"]:
        raise ValueError("Dataset inputs changed after inference preflight")
    adapter_dir = resolve_lora_path(run_metadata["lora_path"], DATA_ROOT)
    if fingerprint_lora(adapter_dir) != run_metadata["adapter_fingerprint"]:
        raise ValueError("LoRA adapter content changed after inference preflight")

    checkpoint_path = (
        Path(DATA_ROOT)
        / "outputs"
        / "inference"
        / run_metadata["run_id"]
        / "shards"
        / f"shard_{shard_id:02d}.json"
    )
    predictions: dict[str, list[float] | None] = {}
    if resume and checkpoint_path.is_file():
        predictions = validate_checkpoint_payload(
            load_json(checkpoint_path),
            expected_metadata,
            assigned_keys,
            require_complete=False,
            label=f"shard {shard_id} checkpoint",
        )
    elif not resume and checkpoint_path.is_file():
        checkpoint_path.unlink()
        dataset_volume.commit()

    if pending_keys(assigned_keys, predictions):
        model, processor = _load_model_and_processor(run_metadata["lora_path"])
        predictions = _execute_inference_batch(
            model=model,
            processor=processor,
            dataset=dataset,
            assigned_keys=assigned_keys,
            predictions=predictions,
            metadata=expected_metadata,
            checkpoint_path=checkpoint_path,
        )

    payload = {"metadata": expected_metadata, "predictions": predictions}
    validate_checkpoint_payload(
        payload,
        expected_metadata,
        assigned_keys,
        require_complete=True,
        label=f"shard {shard_id} result",
    )
    return payload


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes={"/data": dataset_volume})
def finalize_inference_run(plan: dict, payloads: list[dict]) -> dict:
    dataset_volume.reload()
    metadata = plan["metadata"]
    dataset = _load_dataset(
        Path(DATA_ROOT),
        metadata["split"],
        metadata["annotation_run_id"],
    )
    all_keys = sorted(dataset, key=lambda query_id: (dataset[query_id]["visible"], query_id))
    if key_hash(all_keys) != plan.get("all_key_hash"):
        raise ValueError("Dataset key set changed after inference preflight")
    if len(plan["result_keys"]) != len(set(plan["result_keys"])):
        raise ValueError("Inference result plan contains duplicate query IDs")
    expected_full_split = set(plan["result_keys"]) == set(all_keys)
    if plan.get("result_is_full_split") is not expected_full_split:
        raise ValueError("Inference full-split marker does not match the result keys")
    if key_hash(plan["keys"]) != metadata["selected_key_hash"]:
        raise ValueError("Inference plan keys do not match run metadata")
    if fingerprint_inputs(dataset, plan["keys"]) != metadata["input_fingerprint"]:
        raise ValueError("Dataset inputs changed after inference preflight")
    current_image_fingerprint = _selected_image_fingerprint(
        Path(DATA_ROOT),
        dataset,
        plan["keys"],
        deep_verify=False,
        expected_fingerprint=metadata["image_fingerprint"],
    )
    if current_image_fingerprint != metadata["image_fingerprint"]:
        raise ValueError("Dataset image references changed after inference preflight")
    merged = merge_shard_payloads(metadata, plan["shards"], payloads) if plan["keys"] else {}
    run_dir = Path(DATA_ROOT) / "outputs" / "inference" / metadata["run_id"]

    if metadata["mode"] == "retry":
        base_path = (
            Path(DATA_ROOT)
            / "outputs"
            / "inference"
            / metadata["base_run_id"]
            / "checkpoint.json"
        )
        base_payload = load_json(base_path)
        base_keys = sorted(base_payload.get("predictions", {}))
        base_image_fingerprint = _selected_image_fingerprint(
            Path(DATA_ROOT),
            dataset,
            base_keys,
            deep_verify=False,
            expected_fingerprint=base_payload.get("metadata", {}).get("image_fingerprint"),
        )
        authoritative_base = _validate_base_checkpoint(
            base_payload,
            base_run_id=metadata["base_run_id"],
            split=metadata["split"],
            annotation_run_id=metadata["annotation_run_id"],
            dataset=dataset,
            current_image_fingerprint=base_image_fingerprint,
        )
        if authoritative_base != plan["base_predictions"]:
            raise ValueError("Retry base checkpoint changed after inference preflight")
        if stable_json_hash(authoritative_base) != metadata["base_prediction_fingerprint"]:
            raise ValueError("Retry base prediction fingerprint does not match metadata")
        overlay_path = run_dir / "retry_overlay.json"
        atomic_write_json(overlay_path, {"metadata": metadata, "predictions": merged})
        resolved = merge_retry_predictions(authoritative_base, merged, plan["keys"])
    else:
        checkpoint_path = run_dir / "checkpoint.json"
        atomic_write_json(checkpoint_path, {"metadata": metadata, "predictions": merged})
        resolved = merged

    if set(resolved) != set(plan["result_keys"]):
        raise ValueError("Final inference results do not cover the authoritative result keys")
    submission_ready = (
        metadata["split"] == "test"
        and expected_full_split
        and predictions_are_submission_ready(all_keys, resolved)
    )
    metrics = None
    if metadata["split"] in {"train", "val"}:
        metrics = evaluate_predictions(dataset, plan["result_keys"], resolved)
    atomic_write_json(
        run_dir / "summary.json",
        {
            "metadata": metadata,
            "metrics": metrics,
            "total_predictions": len(resolved),
            "valid_predictions": sum(validate_bbox(value) is not None for value in resolved.values()),
            "submission_ready": submission_ready,
        },
    )
    dataset_volume.commit()
    return {
        "metadata": metadata,
        "predictions": resolved,
        "metrics": metrics,
        "result_is_full_split": expected_full_split,
        "submission_ready": submission_ready,
    }


@app.local_entrypoint()
def main(
    split: str = "val",
    annotation_run_id: str = "",
    limit: int | None = None,
    lora_path: str = "",
    resume: bool = True,
    run_tag: str = "",
    retry_failed: bool = False,
    base_run_id: str = "",
    num_shards: int = 1,
    retry_seed: int = 42,
    preflight_only: bool = False,
    deep_verify_images: bool = False,
) -> dict:
    split = _validate_options(
        split,
        annotation_run_id,
        limit,
        num_shards,
        retry_failed,
        base_run_id,
    )
    plan = preflight_inference_environment.remote(
        split=split,
        annotation_run_id=annotation_run_id,
        lora_path=lora_path or None,
        limit=limit,
        run_tag=run_tag,
        retry_failed=retry_failed,
        base_run_id=base_run_id,
        num_shards=num_shards,
        retry_seed=retry_seed,
        resume=resume,
        deep_verify_images=deep_verify_images,
    )
    if preflight_only:
        print(
            f"Inference preflight passed: {plan['metadata']['run_id']} | "
            f"pending GPU shards: {len(plan['pending_shard_ids'])}",
            flush=True,
        )
        return plan

    pending_shard_ids = plan.get("pending_shard_ids", list(range(len(plan["shards"]))))
    args = [
        (shard_id, assigned, plan["keys"], plan["metadata"], resume)
        for shard_id, assigned in enumerate(plan["shards"])
        if shard_id in pending_shard_ids
    ]
    payloads = list(plan.get("completed_payloads", []))
    if args:
        payloads.extend(list(run_inference_shard.starmap(args)))
    result = finalize_inference_run.remote(plan, payloads)

    metrics = result["metrics"]
    if metrics:
        print(
            f"ACC@0.5: {metrics['hits']}/{metrics['total']} = "
            f"{metrics['acc_at_0_5'] * 100:.2f}% | "
            f"Mean IoU: {metrics['mean_iou']:.4f} | Failures: {metrics['failures']}"
        )
    print(f"Inference run_id: {result['metadata']['run_id']}")

    if split == "test" and result["submission_ready"]:
        if set(result["predictions"]) != set(plan["result_keys"]):
            raise ValueError("Final test predictions do not cover the authoritative run keys")
        if not predictions_are_submission_ready(plan["result_keys"], result["predictions"]):
            raise ValueError("Final test predictions are marked ready but contain invalid boxes")
        run_id = result["metadata"]["run_id"]
        output = Path("outputs") / "inference" / run_id / "predictions.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, result["predictions"])
        print(f"Predictions saved to: {output.resolve()}")
    elif split == "test" and result["result_is_full_split"]:
        invalid = sum(
            validate_bbox(value) is None for value in result["predictions"].values()
        )
        print(
            f"Full test run has {invalid} invalid predictions; "
            "no predictions file was written.",
            flush=True,
        )
    return result
