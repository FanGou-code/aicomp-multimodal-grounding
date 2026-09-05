"""Annotation product contract: approved-artifact schema validation.

The annotation generation pipeline (prompt contract, API client, run state
machine, shard checkpoints, style audit) lives in the external private
repository ``query-foundry``; products are delivered manually into
``outputs/annotations/<run_id>/<split>/approved.json``. This module retains
only the product contract that the training side enforces when loading an
approved artifact (2026-09-05 separation; historical runs unaffected).

Approved artifact dict:
    metadata: {status, protocol_version, run_id, split, source_fingerprint,
               preparation_fingerprint, image_fingerprint, dataset_fingerprint,
               sample_count, sequence_count, prompt_hash, provenance, qc}
    data: dict[sample_id, {visible, infrared, depth, query, bbox, width, height}]
"""

from __future__ import annotations

from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.query import preflight_check_dataset
from aicomp_grounding.sequence import (
    source_fingerprint,
    validate_annotation_query,
)
from aicomp_grounding.sharding import group_keys_by_scene

ANNOTATION_MODE = "single_marked_frame_generate"
ASSIGNMENT_POLICY = "single_marked_rgb_query_generate"
APPROVED_FIELDS = ("visible", "infrared", "depth", "query", "bbox", "width", "height")


def approved_dataset_fingerprint(data: dict) -> str:
    return stable_json_hash(data)


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
