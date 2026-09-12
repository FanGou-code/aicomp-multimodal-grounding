"""Annotation-source index loading and fingerprinting.

The census reads the main repository's prepared dataset through the split
index (`data/indexes/<split>.json`) and pins every run to the exact index
bytes via the split manifest. These helpers were absorbed from the retired
v4 generation entry so the census depends only on foundry modules.
"""

from __future__ import annotations

from pathlib import Path

from foundry.utils import resolve_index_dir, split_index_fingerprint, stable_json_hash
from foundry.utils import PREPARATION_PROTOCOL_VERSION
from foundry.pipeline.views import (
    trusted_dataset_image_fingerprint,
    verify_dataset_images,
)
from foundry.utils import load_json


def load_annotation_source(data_root: Path, split: str, *, index_dir: Path | None = None) -> dict:
    """Load the split index and verify it against split_manifest.json."""
    indexes = resolve_index_dir(data_root, index_dir)
    path = indexes / f"{split}.json"
    manifest_path = indexes / "split_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Annotation source index not found: {path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Annotation requires split_manifest.json from prepare_split.py: {manifest_path}"
        )
    data = load_json(path)
    manifest = load_json(manifest_path)
    if (
        manifest.get("status") != "complete"
        or manifest.get("preparation_protocol_version") != PREPARATION_PROTOCOL_VERSION
    ):
        raise ValueError("Dataset split manifest is not a completed supported preparation")
    fingerprints = manifest.get("index_fingerprints")
    counts = manifest.get("index_sample_counts")
    # Both serializations occurred under protocol 2. Check their exact content
    # hashes without rewriting either existing indexes or frozen artifacts.
    valid_hashes = {split_index_fingerprint(data), stable_json_hash(data)}
    if not isinstance(fingerprints, dict) or fingerprints.get(split) not in valid_hashes:
        raise ValueError(f"{split} index does not match split_manifest.json")
    if not isinstance(counts, dict) or counts.get(split) != len(data):
        raise ValueError(f"{split} sample count does not match split_manifest.json")
    return data


def preparation_fingerprint(data_root: Path, *, index_dir: Path | None = None) -> str:
    """Content hash of the split manifest — the preparation half of the run identity."""
    return stable_json_hash(load_json(resolve_index_dir(data_root, index_dir) / "split_manifest.json"))


def image_fingerprint(
    data_root: Path,
    dataset: dict,
    sample_ids: list[str],
    *,
    deep_verify: bool,
) -> str:
    """Image-reference fingerprint; ``deep_verify`` reads and hashes the bytes."""
    if deep_verify:
        return verify_dataset_images(
            data_root,
            dataset,
            sample_ids,
            require_recorded_size=True,
        )
    return trusted_dataset_image_fingerprint(
        dataset,
        sample_ids,
        require_recorded_size=True,
    )
