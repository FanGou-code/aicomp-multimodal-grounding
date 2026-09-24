"""Bind and verify committed index image references.

``trusted_dataset_image_fingerprint`` binds references without reading bytes;
``verify_dataset_images`` decodes every unique modality file and returns a
byte-level fingerprint. Both live here so the annotation contract can bind
references at import time without pulling in an imaging library.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from aicomp_grounding.artifacts import file_set_fingerprint, stable_json_hash

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


def verify_dataset_images(
    data_root: str | Path,
    dataset: dict,
    sample_ids: Iterable[str],
    *,
    require_recorded_size: bool,
) -> str:
    """Decode each unique modality, check alignment, and return a byte fingerprint."""
    from PIL import Image

    root = Path(data_root).resolve()
    selected = list(sample_ids)
    if len(selected) != len(set(selected)):
        raise ValueError("Image verification sample IDs must be unique")

    verified: dict[Path, tuple[int, int]] = {}
    for sample_id in selected:
        if sample_id not in dataset or not isinstance(dataset[sample_id], dict):
            raise ValueError(f"Image verification dataset has no valid sample {sample_id!r}")
        item = dataset[sample_id]
        sizes: list[tuple[int, int]] = []
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
            path = (root / Path(*posix.parts)).resolve()
            try:
                path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Sample {sample_id!r} {field} escapes DATA_ROOT") from exc
            if path not in verified:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Sample {sample_id!r} is missing {field} image: {path}"
                    )
                with Image.open(path) as opened:
                    # verify() only validates the bitstream; header attributes
                    # such as .size stay valid afterwards, so one open is enough.
                    opened.verify()
                    verified[path] = opened.size
            sizes.append(verified[path])
            relative_paths.append(relative)

        if len(set(relative_paths)) != len(MODALITY_FIELDS):
            raise ValueError(f"Sample {sample_id!r} modality paths must be distinct")
        if len(set(sizes)) != 1:
            raise ValueError(f"Sample {sample_id!r} modalities are misaligned: {sizes}")
        if require_recorded_size:
            width = item.get("width")
            height = item.get("height")
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, int)
                or not isinstance(height, int)
                or (width, height) != sizes[0]
            ):
                raise ValueError(
                    f"Sample {sample_id!r} recorded size {(width, height)} "
                    f"does not match image size {sizes[0]}"
                )

    return file_set_fingerprint(root, verified)
