"""Contract validation and fingerprinting for approved annotation artifacts.

Ensures generated artifacts conform to the downstream multimodal grounding training contract
(protocol_version=12). All fingerprints are pure, deterministic SHA-256 content hashes.
Zero pip dependencies required.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath
from typing import Any

from foundry.bbox import validate_bbox
from foundry.pipeline.sharding import group_keys_by_scene
from foundry.utils import stable_json_hash

ANNOTATION_PROTOCOL_VERSION = 12
ANNOTATION_MODE = "single_marked_frame_generate"
ASSIGNMENT_POLICY = "single_marked_rgb_query_generate"
RENDER_PROTOCOL = "single-marked-full-rgb-v8"
APPROVED_FIELDS = ("visible", "infrared", "depth", "query", "bbox", "width", "height")
MODALITY_FIELDS = ("visible", "infrared", "depth")
TRUSTED_IMAGE_FINGERPRINT_PREFIX = "manifest_"

_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)?")
_COORDINATES = re.compile(r"[\[(]\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?")
_CONTROL_TOKEN = re.compile(r"<\|[^<>\r\n]*\|>")
_FORBIDDEN = (
    "red rectangle",
    "red outline",
    "bounding box",
    "image 1",
    "image 2",
    "first image",
    "second image",
)

_ANNOTATION_SCAFFOLD = (
    "highlighted",
    "marked target",
    "marked object",
    "outlined target",
    "red rectangle",
    "red outline",
    "annotated image",
    "annotation view",
    "this frame",
    "current frame",
    "in the frame",
    "within the frame",
    "this image",
    "current image",
    "in the image",
    "within the image",
    "crop",
    "same target",
    "visible image",
    "thermal image",
    "infrared image",
    "depth image",
    "depth map",
)
# NOTE: "image" is deliberately absent here, mirroring the main repository's
# aicomp_grounding/sequence.py. The official test dialect uses "of the image"
# as a frame anchor and v5/v6 targets that dialect, so a blanket ban would
# reject legitimate queries. Real scaffolding ("this image", "in the image",
# "annotated image", ...) is still caught by _ANNOTATION_SCAFFOLD above.
_ANNOTATION_TERM = re.compile(
    r"\b(?:target|crop|annotation|annotated|bbox|coordinate|"
    r"infrared|thermal|depth|rgb)\b",
    flags=re.IGNORECASE,
)
_GENERIC_CATEGORY = re.compile(
    r"\b(?:thing|item|entity)\b",
    flags=re.IGNORECASE,
)


def clean_query_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = text.strip().strip("`").strip().strip('"').strip("'").strip()
    return cleaned.rstrip(".?!;").strip()


def validate_generated_query(
    text: str, *, min_words: int = 1, max_words: int = 55
) -> tuple[bool, str]:
    """Validate format and obvious annotation failures without judging semantics."""
    cleaned = clean_query_text(text)
    if not cleaned:
        return False, "empty response"
    if "\n" in cleaned or "\r" in cleaned:
        return False, "query must be one line"
    if any(marker in cleaned.lower() for marker in _FORBIDDEN):
        return False, "query mentions annotation scaffolding"
    if _COORDINATES.search(cleaned):
        return False, "query contains coordinate-like output"
    if _CONTROL_TOKEN.search(cleaned):
        return False, "query contains a model control token"
    if cleaned.startswith(("#", "- ", "* ")):
        return False, "query contains markdown formatting"

    words = _WORD.findall(cleaned)
    if not min_words <= len(words) <= max_words:
        return False, f"expected {min_words}-{max_words} English words, got {len(words)}"
    if sum(character.isascii() for character in cleaned) / len(cleaned) < 0.9:
        return False, "query is not predominantly English/ASCII"
    return True, ""


def validate_annotation_query(query: str) -> tuple[bool, str]:
    valid, reason = validate_generated_query(query, min_words=1, max_words=55)
    if not valid:
        return valid, reason
    lowered = clean_query_text(query).lower()
    marker = next(
        (
            phrase
            for phrase in _ANNOTATION_SCAFFOLD
            if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lowered)
        ),
        None,
    )
    if marker:
        return False, f"query mentions annotation scaffolding {marker!r}"
    marker_match = _ANNOTATION_TERM.search(clean_query_text(query))
    if marker_match:
        return False, f"query mentions annotation term {marker_match.group(0)!r}"
    generic_match = _GENERIC_CATEGORY.search(clean_query_text(query))
    if generic_match:
        return False, f"query uses generic category {generic_match.group(0)!r}"
    return True, ""


def source_fingerprint(dataset: dict) -> str:
    """Fingerprint immutable annotation inputs while deliberately ignoring Query."""
    identity = {}
    for key in sorted(dataset):
        item = dataset[key]
        identity[key] = {
            field: item.get(field)
            for field in ("visible", "infrared", "depth", "bbox", "width", "height")
        }
    return stable_json_hash(identity)


def approved_dataset_fingerprint(data: dict) -> str:
    return stable_json_hash(data)


def trusted_dataset_image_fingerprint(
    dataset: dict,
    sample_ids: Iterable[str],
    *,
    require_recorded_size: bool,
) -> str:
    """Bind a committed index's image references without reading image bytes."""
    selected = list(sample_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("Image verification sample IDs must be unique")

    references: dict[str, dict[str, object]] = {}
    for sample_id in selected:
        if sample_id not in dataset or not isinstance(dataset[sample_id], dict):
            raise ValueError(f"Image verification dataset has no valid sample {sample_id!r}")
        item = dataset[sample_id]
        record: dict[str, object] = {}
        relative_paths: list[str] = []
        for field in MODALITY_FIELDS:
            relative = item.get(field)
            if not isinstance(relative, str) or not relative or "\\" in relative:
                raise ValueError(f"Sample {sample_id!r} has invalid {field} path")
            posix = PurePosixPath(relative)
            if (
                posix.is_absolute()
                or ".." in posix.parts
                or "." in posix.parts
                or posix.as_posix() != relative
            ):
                raise ValueError(f"Sample {sample_id!r} has non-canonical {field} path")
            record[field] = relative
            relative_paths.append(relative)

        if len(set(relative_paths)) != len(MODALITY_FIELDS):
            raise ValueError(f"Sample {sample_id!r} modality paths must be distinct")
        if require_recorded_size:
            for field in ("width", "height"):
                value = item.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Sample {sample_id!r} has invalid recorded {field}")
                record[field] = value
        references[sample_id] = record

    digest = stable_json_hash(
        {
            "policy": "committed-volume-manifest-v1",
            "references": references,
        }
    )
    return f"{TRUSTED_IMAGE_FINGERPRINT_PREFIX}{digest}"


def is_trusted_image_fingerprint(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith(TRUSTED_IMAGE_FINGERPRINT_PREFIX):
        return False
    digest = value[len(TRUSTED_IMAGE_FINGERPRINT_PREFIX) :]
    return len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)


def preflight_check_dataset(data: dict, *, split_name: str = "dataset") -> list[str]:
    """Validate a dataset dict structurally before training."""
    errors: list[str] = []
    if not isinstance(data, dict) or not data:
        return [f"{split_name}: dataset is empty or is not a JSON object."]

    empty_ids: list[str] = []
    bad_bbox_ids: list[str] = []
    missing_field_ids: list[str] = []

    required_fields = ("visible", "infrared", "depth", "query", "bbox", "width", "height")

    for key, item in data.items():
        for field in required_fields:
            if field not in item:
                missing_field_ids.append(key)
                break
        else:
            query = item.get("query", "")
            if not isinstance(query, str) or not query.strip():
                empty_ids.append(key)
            if validate_bbox(item.get("bbox")) is None:
                bad_bbox_ids.append(key)

    if empty_ids:
        sample = ", ".join(empty_ids[:5])
        errors.append(f"{split_name}: {len(empty_ids)} samples have empty query (e.g. {sample}).")
    if bad_bbox_ids:
        sample = ", ".join(bad_bbox_ids[:5])
        errors.append(f"{split_name}: {len(bad_bbox_ids)} samples have invalid bbox (e.g. {sample}).")
    if missing_field_ids:
        sample = ", ".join(missing_field_ids[:5])
        errors.append(f"{split_name}: {len(missing_field_ids)} samples missing required fields (e.g. {sample}).")
    return errors


def validate_approved_artifact(
    artifact: object,
    *,
    expected_split: str,
    expected_run_id: str,
    strict_query_qc: bool = True,
) -> dict:
    """Strictly validate an approved annotation artifact against the downstream contract."""
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
        raise ValueError(f"Approved annotation metadata schema is invalid")
    if (
        metadata["status"] != "approved"
        or metadata["protocol_version"] != ANNOTATION_PROTOCOL_VERSION
    ):
        raise ValueError("Annotation artifact is not approved under the current protocol")
    if metadata["split"] != expected_split or metadata["run_id"] != expected_run_id:
        raise ValueError(
            f"Approved annotation split or run ID does not match: "
            f"split {metadata['split']} vs {expected_split}, run_id {metadata['run_id']} vs {expected_run_id}"
        )
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
        raise ValueError(f"Approved annotation provenance schema is invalid")
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
        raise ValueError(f"Approved annotation sample count mismatch: {len(data)} != {metadata['sample_count']}")

    invalid_queries = []
    for sample_id, item in data.items():
        if not isinstance(item, dict) or set(item) != set(APPROVED_FIELDS):
            raise ValueError(f"Approved sample {sample_id!r} has an invalid field schema")
        valid, reason = validate_annotation_query(item["query"])
        if not valid:
            invalid_queries.append((sample_id, item["query"], reason))

    if invalid_queries and strict_query_qc:
        sample_fail = invalid_queries[0]
        raise ValueError(f"Approved sample {sample_fail[0]!r} failed query QC: {sample_fail[2]} ({len(invalid_queries)} failed total)")

    if len(group_keys_by_scene(list(data), data)) != metadata["sequence_count"]:
        raise ValueError("Approved annotation sequence count does not match")
    if source_fingerprint(data) != metadata["source_fingerprint"]:
        raise ValueError("Approved annotation source fingerprint does not match")
    if approved_dataset_fingerprint(data) != metadata["dataset_fingerprint"]:
        raise ValueError("Approved annotation dataset fingerprint does not match")
    errors = preflight_check_dataset(data, split_name=expected_split)
    if errors:
        raise ValueError("Approved annotation data failed structural QC: " + "; ".join(errors))
    return artifact


def validate_training_artifacts(
    train_artifact: dict,
    val_artifact: dict,
    *,
    annotation_run_id: str,
    strict_query_qc: bool = True,
) -> tuple[dict, dict]:
    """Validate mutual consistency between train and val approved artifacts."""
    train = validate_approved_artifact(
        train_artifact,
        expected_split="train",
        expected_run_id=annotation_run_id,
        strict_query_qc=strict_query_qc,
    )
    val = validate_approved_artifact(
        val_artifact,
        expected_split="val",
        expected_run_id=annotation_run_id,
        strict_query_qc=strict_query_qc,
    )
    train_meta = train["metadata"]
    val_meta = val["metadata"]
    for field in ("prompt_hash", "provenance"):
        if train_meta[field] != val_meta[field]:
            raise ValueError(f"Train/val approved artifacts disagree on {field}")

    train_data = train["data"]
    val_data = val["data"]
    overlap = set(train_data) & set(val_data)
    if overlap:
        raise ValueError(f"Train/val sample IDs overlap: {sorted(overlap)[:5]}")
    train_scenes = set(group_keys_by_scene(list(train_data), train_data))
    val_scenes = set(group_keys_by_scene(list(val_data), val_data))
    scene_overlap = train_scenes & val_scenes
    if scene_overlap:
        raise ValueError(f"Train/val sequence IDs overlap: {sorted(scene_overlap)[:5]}")
    return train, val
