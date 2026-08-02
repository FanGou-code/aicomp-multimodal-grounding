"""Pure planning, checkpoint, QC, and publication rules for self-annotation.

Data structures
---------------
Plan dict (from build_annotation_plan):
    metadata: {protocol_version, run_id, run_tag, split, source_fingerprint,
               preparation_fingerprint, image_fingerprint, provider, api_base_url,
               model_name, model_revision, model_weights_url, model_license,
               prompt_hash, render_protocol, generation_config, seed, mode,
               assignment_policy, limit_sequences, requested_num_shards,
               effective_num_shards, selected_sequence_ids, selected_sequence_hash,
               selected_sample_hash}
    shards: list[list[str]]           # sequence IDs grouped per shard
    selected_sample_ids: list[str]    # all sample IDs covered by this run

Shard payload dict:
    metadata: {...}  (plan metadata + shard_id + assigned_sequence_ids)
    results: dict[sequence_id, SequenceResult]

SequenceResult dict:
    status: "in_progress" | "completed" | "failed"
    frames: dict[sample_id, FrameResult]

FrameResult dict:
    status: "completed" | "failed"
    query: str | None
    uncertain: bool
    attempts: int
    error: str
    api_calls: list[ApiCallRecord]

Approved artifact dict:
    metadata: {status, protocol_version, run_id, split, source_fingerprint,
               preparation_fingerprint, image_fingerprint, dataset_fingerprint,
               sample_count, sequence_count, prompt_hash, provenance, qc}
    data: dict[sample_id, {visible, infrared, depth, query, bbox, width, height}]
"""

from __future__ import annotations

from collections.abc import Mapping

from aicomp_grounding.artifacts import key_hash, require_exact_metadata, stable_json_hash
from aicomp_grounding.bbox import validate_bbox
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.query import preflight_check_dataset
from aicomp_grounding.sequence import (
    source_fingerprint,
    validate_annotation_query,
)
from aicomp_grounding.sharding import (
    group_keys_by_scene,
    select_scene_ids,
    shard_scene_ids,
)

ANNOTATION_MODE = "single_marked_frame_generate"
ASSIGNMENT_POLICY = "single_marked_rgb_query_generate"
APPROVED_FIELDS = ("visible", "infrared", "depth", "query", "bbox", "width", "height")

ANNOTATION_METADATA_FIELDS = (
    "protocol_version",
    "run_id",
    "run_tag",
    "split",
    "source_fingerprint",
    "preparation_fingerprint",
    "image_fingerprint",
    "provider",
    "api_base_url",
    "model_name",
    "model_revision",
    "model_weights_url",
    "model_license",
    "prompt_hash",
    "render_protocol",
    "generation_config",
    "seed",
    "mode",
    "assignment_policy",
    "limit_sequences",
    "requested_num_shards",
    "effective_num_shards",
    "selected_sequence_ids",
    "selected_sequence_hash",
    "selected_sample_hash",
)


def build_annotation_plan(
    dataset: dict,
    *,
    split: str,
    model_name: str,
    model_revision: str,
    prompt_hash: str,
    preparation_fingerprint: str,
    image_fingerprint: str,
    provider: str,
    api_base_url: str,
    seed: int,
    limit_sequences: int | None,
    requested_num_shards: int,
    run_tag: str,
    model_weights_url: str,
    model_license: str,
    render_protocol: str,
    generation_config: dict,
) -> dict:
    if not dataset:
        raise ValueError("Annotation source dataset is empty")
    if requested_num_shards <= 0:
        raise ValueError("requested_num_shards must be positive")
    for label, value in (
        ("preparation_fingerprint", preparation_fingerprint),
        ("image_fingerprint", image_fingerprint),
        ("provider", provider),
        ("api_base_url", api_base_url),
        ("model_name", model_name),
        ("model_revision", model_revision),
        ("model_weights_url", model_weights_url),
        ("model_license", model_license),
        ("prompt_hash", prompt_hash),
        ("render_protocol", render_protocol),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} must be a non-empty string")
    if not api_base_url.startswith("https://") or not model_weights_url.startswith("https://"):
        raise ValueError("Annotation API and model weights URLs must use HTTPS")
    if not isinstance(generation_config, dict) or not generation_config:
        raise ValueError("generation_config must be a non-empty object")
    for sample_id, item in dataset.items():
        if not isinstance(item, dict):
            raise ValueError(f"Annotation source item {sample_id!r} must be an object")
        for field in ("visible", "infrared", "depth", "bbox", "width", "height"):
            if field not in item:
                raise ValueError(f"Annotation source item {sample_id!r} is missing {field!r}")
        for field in ("visible", "infrared", "depth"):
            if not isinstance(item[field], str) or not item[field]:
                raise ValueError(
                    f"Annotation source item {sample_id!r} has invalid {field!r}"
                )
        for field in ("width", "height"):
            if (
                isinstance(item[field], bool)
                or not isinstance(item[field], int)
                or item[field] <= 0
            ):
                raise ValueError(
                    f"Annotation source item {sample_id!r} has invalid {field!r}"
                )
        if validate_bbox(item["bbox"]) is None:
            raise ValueError(f"Annotation source item {sample_id!r} has invalid bbox")

    selected_sequences = select_scene_ids(dataset, limit=limit_sequences, seed=seed)
    shards = shard_scene_ids(selected_sequences, dataset, requested_num_shards)
    groups = group_keys_by_scene(list(dataset), dataset)
    selected_samples = sorted(
        sample_id for sequence_id in selected_sequences for sample_id in groups[sequence_id]
    )
    campaign_identity = {
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "run_tag": run_tag,
        "preparation_fingerprint": preparation_fingerprint,
        "provider": provider,
        "api_base_url": api_base_url,
        "model_name": model_name,
        "model_revision": model_revision,
        "model_weights_url": model_weights_url,
        "model_license": model_license,
        "prompt_hash": prompt_hash,
        "render_protocol": render_protocol,
        "generation_config": generation_config,
        "seed": seed,
        "mode": ANNOTATION_MODE,
        "assignment_policy": ASSIGNMENT_POLICY,
        "limit_sequences": limit_sequences,
    }
    run_id = f"annot_{stable_json_hash(campaign_identity, length=16)}"
    metadata = {
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "run_id": run_id,
        "run_tag": run_tag,
        "split": split,
        "source_fingerprint": source_fingerprint(dataset),
        "preparation_fingerprint": preparation_fingerprint,
        "image_fingerprint": image_fingerprint,
        "provider": provider,
        "api_base_url": api_base_url,
        "model_name": model_name,
        "model_revision": model_revision,
        "model_weights_url": model_weights_url,
        "model_license": model_license,
        "prompt_hash": prompt_hash,
        "render_protocol": render_protocol,
        "generation_config": generation_config,
        "seed": seed,
        "mode": ANNOTATION_MODE,
        "assignment_policy": ASSIGNMENT_POLICY,
        "limit_sequences": limit_sequences,
        "requested_num_shards": requested_num_shards,
        "effective_num_shards": len(shards),
        "selected_sequence_ids": selected_sequences,
        "selected_sequence_hash": key_hash(selected_sequences),
        "selected_sample_hash": key_hash(selected_samples),
    }
    if tuple(metadata) != ANNOTATION_METADATA_FIELDS:
        raise AssertionError("Annotation metadata schema is out of sync")
    return {
        "metadata": metadata,
        "shards": shards,
        "selected_sample_ids": selected_samples,
    }


def build_annotation_shard_metadata(
    run_metadata: dict,
    shard_id: int,
    assigned_sequence_ids: list[str],
    dataset: dict,
) -> dict:
    if not 0 <= shard_id < run_metadata["effective_num_shards"]:
        raise ValueError(f"Invalid annotation shard ID {shard_id}")
    groups = group_keys_by_scene(list(dataset), dataset)
    assigned_samples = sorted(
        sample_id
        for sequence_id in assigned_sequence_ids
        for sample_id in groups.get(sequence_id, [])
    )
    return {
        **run_metadata,
        "shard_id": shard_id,
        "assigned_sequence_ids": list(assigned_sequence_ids),
        "assigned_sequence_hash": key_hash(assigned_sequence_ids),
        "assigned_sample_hash": key_hash(assigned_samples),
    }


def _validate_api_calls(calls: object, label: str) -> list[dict]:
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{label} has invalid API call records")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {
            "response_id",
            "trace_id",
            "model",
            "finish_reason",
            "usage",
        }:
            raise ValueError(f"{label} has invalid API call provenance")
        if not isinstance(call["model"], str) or not call["model"]:
            raise ValueError(f"{label} API call has no model identity")
        usage = call["usage"]
        if not isinstance(usage, dict) or set(usage) != {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }:
            raise ValueError(f"{label} API call has invalid usage")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in usage.values()
        ):
            raise ValueError(f"{label} API call usage is invalid")
    return calls


def _validate_attempts(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} has an invalid attempt count")
    return value


def _validate_frame_result(
    frame: object,
    *,
    sample_id: str,
    generator_model: str,
) -> dict:
    expected = {
        "status",
        "query",
        "uncertain",
        "attempts",
        "error",
        "api_calls",
    }
    if not isinstance(frame, dict) or set(frame) != expected:
        raise ValueError(f"Frame {sample_id!r} result schema is invalid")
    if frame["status"] not in {"completed", "failed"}:
        raise ValueError(f"Frame {sample_id!r} has an invalid status")
    _validate_attempts(frame["attempts"], label=f"Frame {sample_id!r}")
    calls = _validate_api_calls(frame["api_calls"], f"Frame {sample_id!r}")
    call_models = {call["model"] for call in calls}
    if not call_models <= {generator_model}:
        raise ValueError(f"Frame {sample_id!r} used an unexpected annotation model")
    if len(frame["api_calls"]) < frame["attempts"]:
        raise ValueError(f"Frame {sample_id!r} attempt history is inconsistent")
    if not isinstance(frame["uncertain"], bool):
        raise ValueError(f"Frame {sample_id!r} uncertain flag is invalid")
    if frame["status"] == "failed":
        if frame["query"] is not None:
            raise ValueError(f"Failed frame {sample_id!r} cannot contain an approved result")
        if not isinstance(frame["error"], str) or not frame["error"]:
            raise ValueError(f"Failed frame {sample_id!r} has invalid failure details")
        return frame

    valid, reason = validate_annotation_query(frame["query"])
    if not valid:
        raise ValueError(f"Completed frame {sample_id!r} query is invalid: {reason}")
    if frame["error"] != "":
        raise ValueError(f"Completed frame {sample_id!r} contains a failure reason")
    return frame


def _validate_sequence_result(
    sequence_id: str,
    result: object,
    sequence_samples: list[str],
    *,
    generator_model: str,
) -> dict:
    if not isinstance(result, dict) or set(result) != {"status", "frames"}:
        raise ValueError(f"Sequence result {sequence_id!r} has an invalid structure")
    if result["status"] not in {"in_progress", "completed", "failed"}:
        raise ValueError(f"Sequence {sequence_id!r} has an unknown status")
    frames = result["frames"]
    if not isinstance(frames, dict) or set(frames) - set(sequence_samples):
        raise ValueError(f"Sequence {sequence_id!r} contains unexpected frame results")
    for sample_id, frame in frames.items():
        _validate_frame_result(
            frame,
            sample_id=sample_id,
            generator_model=generator_model,
        )
    all_frames_present = set(frames) == set(sequence_samples)
    all_frames_completed = all_frames_present and all(
        frame["status"] == "completed" for frame in frames.values()
    )
    if result["status"] == "completed" and not all_frames_completed:
        raise ValueError(f"Completed sequence {sequence_id!r} is missing passed frames")
    if result["status"] == "failed" and not all_frames_present:
        raise ValueError(f"Failed sequence {sequence_id!r} is missing frame outcomes")
    if result["status"] == "failed" and all_frames_completed:
        raise ValueError(f"Failed sequence {sequence_id!r} contains no failed frame")
    return result


def validate_annotation_checkpoint(
    payload: object,
    expected_metadata: dict,
    assigned_sequence_ids: list[str],
    dataset: dict,
    *,
    require_complete: bool,
    label: str = "annotation checkpoint",
) -> dict[str, dict]:
    if not isinstance(payload, dict) or set(payload) != {"metadata", "results"}:
        raise ValueError(f"{label} must contain exactly metadata and results")
    require_exact_metadata(payload["metadata"], expected_metadata, label=label)
    results = payload["results"]
    if not isinstance(results, dict):
        raise ValueError(f"{label} results must be an object")
    assigned = set(assigned_sequence_ids)
    if set(results) - assigned:
        raise ValueError(f"{label} contains unexpected sequence IDs")
    if require_complete and set(results) != assigned:
        raise ValueError(f"{label} does not cover every assigned sequence")

    groups = group_keys_by_scene(list(dataset), dataset)
    generator_model = expected_metadata["model_name"]
    normalized: dict[str, dict] = {}
    for sequence_id, result in results.items():
        normalized[sequence_id] = _validate_sequence_result(
            sequence_id,
            result,
            groups[sequence_id],
            generator_model=generator_model,
        )
        if require_complete and normalized[sequence_id]["status"] == "in_progress":
            raise ValueError(f"{label} contains in-progress sequence {sequence_id!r}")
    return normalized


def pending_sequences(
    assigned_sequence_ids: list[str],
    results: Mapping[str, dict],
    *,
    retry_failed: bool,
) -> list[str]:
    return [
        sequence_id
        for sequence_id in assigned_sequence_ids
        if sequence_id not in results
        or results[sequence_id].get("status") == "in_progress"
        or (retry_failed and results[sequence_id].get("status") == "failed")
    ]


def pending_frames(
    sequence_sample_ids: list[str],
    frame_results: Mapping[str, dict],
    *,
    retry_failed: bool,
) -> list[str]:
    return [
        sample_id
        for sample_id in sequence_sample_ids
        if sample_id not in frame_results
        or (retry_failed and frame_results[sample_id].get("status") == "failed")
    ]


def annotation_attempt_numbers(
    previous_result: Mapping[str, object] | None,
    *,
    retry_failed: bool,
    attempts_per_run: int = 3,
) -> list[int]:
    """Continue failed frame attempts instead of replaying attempts 1..N."""
    if attempts_per_run <= 0:
        raise ValueError("attempts_per_run must be positive")
    prior_attempts = 0
    if retry_failed and previous_result and previous_result.get("status") == "failed":
        value = previous_result.get("attempts")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Failed annotation result has an invalid attempt count")
        prior_attempts = value
    return list(range(prior_attempts + 1, prior_attempts + attempts_per_run + 1))


def merge_annotation_payloads(plan: dict, payloads: list[dict], dataset: dict) -> dict[str, dict]:
    shards = plan["shards"]
    if len(payloads) != len(shards):
        raise ValueError(f"Expected {len(shards)} annotation shards, received {len(payloads)}")
    by_shard: dict[int, dict] = {}
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
            raise ValueError("Invalid annotation shard payload")
        shard_id = payload["metadata"].get("shard_id")
        if not isinstance(shard_id, int) or shard_id in by_shard:
            raise ValueError(f"Duplicate or invalid annotation shard ID {shard_id!r}")
        by_shard[shard_id] = payload

    merged: dict[str, dict] = {}
    for shard_id, assigned_sequences in enumerate(shards):
        if shard_id not in by_shard:
            raise ValueError(f"Missing annotation shard {shard_id}")
        expected = build_annotation_shard_metadata(
            plan["metadata"], shard_id, assigned_sequences, dataset
        )
        results = validate_annotation_checkpoint(
            by_shard[shard_id],
            expected,
            assigned_sequences,
            dataset,
            require_complete=True,
            label=f"annotation shard {shard_id}",
        )
        if set(merged) & set(results):
            raise ValueError("Duplicate sequences across annotation shards")
        merged.update(results)
    return merged


def approved_dataset_fingerprint(data: dict) -> str:
    return stable_json_hash(data)


def build_approved_artifact(dataset: dict, plan: dict, results: dict[str, dict]) -> dict:
    if not isinstance(dataset, dict) or not dataset:
        raise ValueError("Cannot approve an empty annotation dataset")
    metadata = plan["metadata"]
    if metadata["limit_sequences"] is not None:
        raise ValueError("Limited annotation runs cannot be approved")
    groups = group_keys_by_scene(list(dataset), dataset)
    if set(results) != set(groups):
        raise ValueError("Annotation results do not cover every source sequence")

    annotations: dict[str, str] = {}
    for sequence_id, result in results.items():
        if result.get("status") != "completed":
            raise ValueError(f"Sequence {sequence_id!r} did not pass annotation")
        for sample_id, frame in result["frames"].items():
            if sample_id in annotations:
                raise ValueError(f"Duplicate approved sample {sample_id!r}")
            query = frame["query"]
            valid, reason = validate_annotation_query(query)
            if not valid:
                raise ValueError(f"Approved query {sample_id!r} failed QC: {reason}")
            annotations[sample_id] = query
    if set(annotations) != set(dataset):
        raise ValueError("Approved queries do not cover every source sample")

    approved_data: dict[str, dict] = {}
    for sample_id, source_item in dataset.items():
        item = {
            field: source_item[field]
            for field in APPROVED_FIELDS
            if field != "query" and field in source_item
        }
        item["query"] = annotations[sample_id]
        approved_data[sample_id] = item
    errors = preflight_check_dataset(approved_data, split_name=metadata["split"])
    if errors:
        raise ValueError("Approved dataset failed training preflight: " + "; ".join(errors))

    artifact_metadata = {
        "status": "approved",
        "protocol_version": ANNOTATION_PROTOCOL_VERSION,
        "run_id": metadata["run_id"],
        "split": metadata["split"],
        "source_fingerprint": metadata["source_fingerprint"],
        "preparation_fingerprint": metadata["preparation_fingerprint"],
        "image_fingerprint": metadata["image_fingerprint"],
        "dataset_fingerprint": approved_dataset_fingerprint(approved_data),
        "sample_count": len(approved_data),
        "sequence_count": len(groups),
        "prompt_hash": metadata["prompt_hash"],
        "provenance": {
            "source_type": "hosted_open_weights",
            "provider": metadata["provider"],
            "api_base_url": metadata["api_base_url"],
            "annotator_model": metadata["model_name"],
            "annotator_revision": metadata["model_revision"],
            "model_weights_url": metadata["model_weights_url"],
            "model_license": metadata["model_license"],
            "mode": metadata["mode"],
            "assignment_policy": metadata["assignment_policy"],
            "render_protocol": metadata["render_protocol"],
            "generation_config": metadata["generation_config"],
        },
        "qc": {
            "complete": True,
            "failed_sequences": 0,
            "failed_frames": 0,
            "invalid_queries": 0,
            "generated_samples": len(approved_data),
        },
    }
    return {"metadata": artifact_metadata, "data": approved_data}


def validate_approved_artifact(
    artifact: object,
    *,
    expected_split: str,
    expected_run_id: str,
) -> dict:
    if not isinstance(artifact, dict) or set(artifact) != {"metadata", "data"}:
        raise ValueError("Approved annotation artifact must contain metadata and data")
    metadata = artifact["metadata"]
    data = artifact["data"]
    required = {
        "status",
        "protocol_version",
        "run_id",
        "split",
        "source_fingerprint",
        "preparation_fingerprint",
        "image_fingerprint",
        "dataset_fingerprint",
        "sample_count",
        "sequence_count",
        "prompt_hash",
        "provenance",
        "qc",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Approved annotation metadata schema is invalid")
    if (
        metadata["status"] != "approved"
        or metadata["protocol_version"] != ANNOTATION_PROTOCOL_VERSION
    ):
        raise ValueError("Annotation artifact is not approved under the current protocol")
    if metadata["split"] != expected_split or metadata["run_id"] != expected_run_id:
        raise ValueError("Approved annotation split or run ID does not match")
    for field in (
        "source_fingerprint",
        "preparation_fingerprint",
        "image_fingerprint",
        "dataset_fingerprint",
        "prompt_hash",
    ):
        if not isinstance(metadata[field], str) or not metadata[field]:
            raise ValueError(f"Approved annotation {field} is missing")
    provenance = metadata["provenance"]
    provenance_fields = {
        "source_type",
        "provider",
        "api_base_url",
        "annotator_model",
        "annotator_revision",
        "model_weights_url",
        "model_license",
        "mode",
        "assignment_policy",
        "render_protocol",
        "generation_config",
    }
    if not isinstance(provenance, dict) or set(provenance) != provenance_fields:
        raise ValueError("Approved annotation provenance schema is invalid")
    if provenance["source_type"] != "hosted_open_weights":
        raise ValueError("Approved annotation provenance is not an open-weights API")
    for field in (
        "provider",
        "api_base_url",
        "annotator_model",
        "annotator_revision",
        "model_weights_url",
        "model_license",
        "render_protocol",
    ):
        if not isinstance(provenance[field], str) or not provenance[field]:
            raise ValueError(f"Approved annotation provenance {field} is missing")
    if not provenance["api_base_url"].startswith("https://"):
        raise ValueError("Approved annotation API provenance must use HTTPS")
    if not provenance["model_weights_url"].startswith("https://"):
        raise ValueError("Approved annotation weights provenance must use HTTPS")
    if not isinstance(provenance["generation_config"], dict) or not provenance["generation_config"]:
        raise ValueError("Approved annotation generation config is missing")
    if provenance["mode"] != ANNOTATION_MODE:
        raise ValueError("Approved annotation mode is invalid")
    if provenance["assignment_policy"] != ASSIGNMENT_POLICY:
        raise ValueError("Approved assignment policy is invalid")
    qc = metadata["qc"]
    if not isinstance(qc, dict) or qc != {
        "complete": True,
        "failed_sequences": 0,
        "failed_frames": 0,
        "invalid_queries": 0,
        "generated_samples": metadata["sample_count"],
    }:
        raise ValueError("Approved annotation QC status is invalid")
    if not isinstance(data, dict) or not data:
        raise ValueError("Approved annotation data must be a non-empty object")
    for count_field in ("sample_count", "sequence_count"):
        count = metadata[count_field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Approved annotation {count_field} must be a positive integer")
    if len(data) != metadata["sample_count"]:
        raise ValueError("Approved annotation sample count does not match")
    for sample_id, item in data.items():
        if not isinstance(item, dict) or set(item) != set(APPROVED_FIELDS):
            raise ValueError(f"Approved sample {sample_id!r} has an invalid field schema")
        valid, reason = validate_annotation_query(item["query"])
        if not valid:
            raise ValueError(f"Approved sample {sample_id!r} failed query QC: {reason}")
    if len(group_keys_by_scene(list(data), data)) != metadata["sequence_count"]:
        raise ValueError("Approved annotation sequence count does not match")
    if source_fingerprint(data) != metadata["source_fingerprint"]:
        raise ValueError("Approved annotation source fingerprint does not match")
    if approved_dataset_fingerprint(data) != metadata["dataset_fingerprint"]:
        raise ValueError("Approved annotation dataset fingerprint does not match")
    if preflight_check_dataset(data, split_name=expected_split):
        raise ValueError("Approved annotation data failed structural QC")
    return artifact


# ---------------------------------------------------------------------------
# Red-box marked image rendering (formerly annotation_views.py)
# ---------------------------------------------------------------------------

import base64
import io
from PIL import Image, ImageDraw

RENDER_PROTOCOL = "single-marked-full-rgb-v8"
MARK_COLOR = (255, 0, 0)


def _resize_preserving_aspect(image: Image.Image, max_side: int) -> Image.Image:
    resized = image.convert("RGB")
    resized.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return resized


def _draw_outward_box(image: Image.Image, pixel_box: tuple[float, float, float, float]) -> None:
    x1, y1, x2, y2 = pixel_box
    padding = max(4, round(min(image.size) * 0.006))
    line_width = max(3, round(min(image.size) * 0.004))
    ImageDraw.Draw(image).rectangle(
        (
            max(0, round(x1) - padding),
            max(0, round(y1) - padding),
            min(image.width - 1, round(x2) + padding),
            min(image.height - 1, round(y2) + padding),
        ),
        outline=MARK_COLOR,
        width=line_width,
    )


def build_marked_annotation_view(image: Image.Image, bbox: list[float]) -> Image.Image:
    """Return one complete RGB scene with an outward target rectangle."""
    normalized = validate_bbox(bbox)
    if normalized is None:
        raise ValueError(f"Invalid annotation bbox: {bbox!r}")
    marked = _resize_preserving_aspect(image, 1536)
    width, height = marked.size
    x1, y1, x2, y2 = normalized
    _draw_outward_box(
        marked,
        (x1 * width, y1 * height, x2 * width, y2 * height),
    )
    return marked


def jpeg_data_url(image: Image.Image, *, quality: int = 90) -> str:
    if not 1 <= quality <= 95:
        raise ValueError("JPEG quality must be between 1 and 95")
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
