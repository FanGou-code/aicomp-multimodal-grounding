"""Contract validation and fingerprinting for approved annotation artifacts.

Ensures generated artifacts conform to the downstream multimodal grounding training contract
(protocol_version=13). All fingerprints are pure, deterministic SHA-256 content hashes.
Zero pip dependencies required.
"""

from __future__ import annotations

from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.query import preflight_check_dataset, validate_annotation_query
from aicomp_grounding.sharding import group_keys_by_scene

APPROVED_FIELDS = ("visible", "infrared", "depth", "query", "bbox", "width", "height")


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


REQUIRED_METADATA_FIELDS = {
    "status",
    "protocol_version",
    "split",
    "sample_count",
    "sequence_count",
    "provenance",
}
ALLOWED_METADATA_FIELDS = REQUIRED_METADATA_FIELDS | {
    "run_id",
    "run_tag",
    "source_fingerprint",
    "preparation_fingerprint",
    "image_fingerprint",
    "dataset_fingerprint",
    "prompt_hash",
    "qc",
}


def validate_approved_artifact(
    artifact: object,
    *,
    expected_split: str,
    expected_run_id: str | None = None,
    strict_query_qc: bool = True,
    allow_pending: bool = False,
) -> dict:
    """Strictly validate an approved annotation artifact against the downstream contract."""
    if not isinstance(artifact, dict) or set(artifact) != {"metadata", "data"}:
        raise ValueError("Approved annotation artifact must contain metadata and data")
    metadata = artifact["metadata"]
    data = artifact["data"]

    if not isinstance(metadata, dict):
        raise ValueError("Approved annotation metadata schema is invalid")
    if not REQUIRED_METADATA_FIELDS.issubset(set(metadata)) or not set(metadata).issubset(
        ALLOWED_METADATA_FIELDS
    ):
        raise ValueError("Approved annotation metadata schema is invalid")

    if (
        metadata["status"] != "approved"
        or metadata["protocol_version"] != ANNOTATION_PROTOCOL_VERSION
    ):
        raise ValueError("Annotation artifact is not approved under the current protocol")

    if metadata["split"] != expected_split:
        raise ValueError(
            f"Approved annotation split does not match: split {metadata['split']} vs {expected_split}"
        )

    if expected_run_id and metadata.get("run_id") and metadata["run_id"] != expected_run_id:
        raise ValueError(
            f"Approved annotation run ID does not match: run_id {metadata['run_id']} vs {expected_run_id}"
        )

    for field in (
        "source_fingerprint",
        "preparation_fingerprint",
        "image_fingerprint",
        "dataset_fingerprint",
        "prompt_hash",
    ):
        if field in metadata:
            if not isinstance(metadata[field], str) or not metadata[field]:
                raise ValueError(f"Approved annotation {field} is missing")

    provenance = metadata["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {"source_type"}:
        raise ValueError("Approved annotation provenance schema is invalid")
    if provenance["source_type"] != "human_annotated":
        raise ValueError("Approved annotation provenance is not human-annotated")

    if "qc" in metadata:
        qc = metadata["qc"]
        if not isinstance(qc, dict) or qc.get("complete") is not True:
            raise ValueError("Approved annotation QC status is invalid")

    if not isinstance(data, dict) or not data:
        raise ValueError("Approved annotation data must be a non-empty object")
    for count_field in ("sample_count", "sequence_count"):
        count = metadata[count_field]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"Approved annotation {count_field} must be a positive integer")
    if len(data) != metadata["sample_count"]:
        raise ValueError(
            f"Approved annotation sample count mismatch: {len(data)} != {metadata['sample_count']}"
        )

    if len(group_keys_by_scene(list(data), data)) != metadata["sequence_count"]:
        raise ValueError("Approved annotation sequence count does not match")

    if "source_fingerprint" in metadata:
        if source_fingerprint(data) != metadata["source_fingerprint"]:
            raise ValueError("Approved annotation source fingerprint does not match")
    if "dataset_fingerprint" in metadata:
        if approved_dataset_fingerprint(data) != metadata["dataset_fingerprint"]:
            raise ValueError("Approved annotation dataset fingerprint does not match")

    invalid_queries = []
    for sample_id, item in data.items():
        if not isinstance(item, dict) or set(item) != set(APPROVED_FIELDS):
            raise ValueError(f"Approved sample {sample_id!r} has an invalid field schema")
        query = item.get("query", "")
        if query:
            valid, reason = validate_annotation_query(query)
            if not valid:
                invalid_queries.append((sample_id, query, reason))
        elif not allow_pending:
            invalid_queries.append((sample_id, "", "empty query"))

    if invalid_queries and strict_query_qc:
        sample_fail = invalid_queries[0]
        raise ValueError(
            f"Approved sample {sample_fail[0]!r} failed query QC: {sample_fail[2]} ({len(invalid_queries)} failed total)"
        )

    errors = preflight_check_dataset(data, split_name=expected_split, allow_pending=allow_pending)
    if errors:
        raise ValueError("Approved annotation data failed structural QC: " + "; ".join(errors))
    return artifact


def validate_training_artifacts(
    train_artifact: dict,
    val_artifact: dict,
    *,
    annotation_run_id: str | None = None,
    strict_query_qc: bool = True,
    allow_pending: bool = False,
) -> tuple[dict, dict]:
    """Validate mutual consistency between train and val approved artifacts."""
    train = validate_approved_artifact(
        train_artifact,
        expected_split="train",
        expected_run_id=annotation_run_id,
        strict_query_qc=strict_query_qc,
        allow_pending=allow_pending,
    )
    val = validate_approved_artifact(
        val_artifact,
        expected_split="val",
        expected_run_id=annotation_run_id,
        strict_query_qc=strict_query_qc,
        allow_pending=allow_pending,
    )
    train_meta = train["metadata"]
    val_meta = val["metadata"]
    for field in ("prompt_hash", "provenance"):
        if field in train_meta and field in val_meta and train_meta[field] != val_meta[field]:
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

