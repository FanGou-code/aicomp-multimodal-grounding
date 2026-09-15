"""Bind artifacts to committed index image references without reading bytes."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath

from aicomp_grounding.artifacts import stable_json_hash

MODALITY_FIELDS = ("visible", "infrared", "depth")
TRUSTED_IMAGE_FINGERPRINT_PREFIX = "manifest_"


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
