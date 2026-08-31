#!/usr/bin/env python3
"""Remove training/validation samples whose visible image byte-identically
matches a Test-set image (SHA-256 overlap detection).

This script is a standalone index operation: it reads the existing
train.json and val.json, computes SHA-256 hashes of every sample's
visible image, and deletes any key whose image matches an image in the
competition Test set.

Raw files and processed Depth-JET images are never touched. The
only outputs are updated index JSON files and an audit log.

Usage::

    python scripts/filter_overlap.py \
        --dataset-root data \
        --overwrite-indexes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.artifacts import stable_json_hash


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _build_test_image_hashes(test_visible_dir: Path) -> Dict[str, List[str]]:
    """Return ``{sha256 -> [test_image_stems]}`` mapping."""
    if not test_visible_dir.is_dir():
        print(f"[filter_overlap] Test images not found at {test_visible_dir}; skipping.")
        return {}
    hashes: Dict[str, List[str]] = {}
    for path in sorted(test_visible_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            hashes.setdefault(_sha256_of_file(path), []).append(path.stem)
    print(f"[filter_overlap] Test set: {len(hashes)} unique image hashes.")
    return hashes


def _filter_samples(
    index: Dict[str, dict],
    test_hashes: Dict[str, List[str]],
    data_root: Path,
    split_name: str,
) -> tuple[Dict[str, dict], List[dict]]:
    """Delete samples whose visible image matches a Test image.

    Returns ``(cleaned_index, exclusion_records)``.
    """
    excluded: List[dict] = []
    keys_to_delete: Set[str] = set()

    for key, item in index.items():
        visible_path = data_root / item["visible"]
        if not visible_path.is_file():
            print(f"[filter_overlap] WARNING: missing image {visible_path}, keeping {key}")
            continue
        image_hash = _sha256_of_file(visible_path)
        if image_hash in test_hashes:
            keys_to_delete.add(key)
            excluded.append(
                {
                    "sample_id": key,
                    "visible": item["visible"],
                    "test_images": sorted(set(test_hashes[image_hash])),
                }
            )

    for key in keys_to_delete:
        del index[key]

    print(
        f"[filter_overlap] {split_name}: {len(keys_to_delete)} samples excluded "
        f"(kept {len(index)})"
    )
    return index, excluded


def _index_scene_ids(index: dict) -> List[str]:
    """Return sorted scene IDs represented by an index, dropping emptied scenes."""
    return sorted({str(key).split("_", 1)[0] for key in index})


def _update_split_manifest(manifest: dict, indexes: Dict[str, dict]) -> dict:
    """Refresh manifest fields so counts, fingerprints, and scene lists stay consistent."""
    fingerprints = manifest.setdefault("index_fingerprints", {})
    counts = manifest.setdefault("index_sample_counts", {})
    stats = manifest.get("stats")

    for split, index in indexes.items():
        fingerprints[split] = stable_json_hash(index)
        counts[split] = len(index)

    if isinstance(stats, dict):
        for split in ("train", "val"):
            sample_key = f"{split}_samples"
            if sample_key in stats:
                stats[sample_key] = counts.get(split, 0)
        if "valid_samples" in stats:
            stats["valid_samples"] = sum(int(counts.get(split, 0)) for split in ("train", "val"))
        if "sequences" in stats:
            scene_ids: Set[str] = set()
            for index in indexes.values():
                scene_ids.update(_index_scene_ids(index))
            stats["sequences"] = len(scene_ids)

    if "train_sequences" in manifest and "train" in indexes:
        manifest["train_sequences"] = _index_scene_ids(indexes["train"])
    if "val_sequences" in manifest and "val" in indexes:
        manifest["val_sequences"] = _index_scene_ids(indexes["val"])

    manifest["index_fingerprints"] = fingerprints
    manifest["index_sample_counts"] = counts
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove train/val samples that overlap with Test images (SHA-256)."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data"),
        help="Dataset root directory (default: data)",
    )
    parser.add_argument(
        "--overwrite-indexes",
        action="store_true",
        dest="overwrite_indexes",
        help="Overwrite existing train.json and val.json in place",
    )
    args = parser.parse_args()

    data_root: Path = args.dataset_root
    test_visible_dir = data_root / "Test" / "Images" / "visible"
    test_hashes = _build_test_image_hashes(test_visible_dir)
    if not test_hashes:
        print("[filter_overlap] No Test images to match against; nothing to filter.")
        return

    all_records: Dict[str, List[dict]] = {}
    cleaned_indexes: Dict[str, dict] = {}
    indexes_dir = data_root / "indexes"
    audits_dir = data_root / "audits"
    for split in ("train", "val"):
        index_path = indexes_dir / f"{split}.json"
        if not index_path.is_file():
            print(f"[filter_overlap] {index_path} not found, skipping {split}.")
            continue
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        index, records = _filter_samples(index, test_hashes, data_root, split)
        all_records[split] = records
        cleaned_indexes[split] = index

        if args.overwrite_indexes:
            atomic_write_json(index_path, index)

    # Keep split_manifest.json consistent with the filtered indexes so the
    # annotation and training pipelines accept the new split.
    manifest_path = indexes_dir / "split_manifest.json"
    if args.overwrite_indexes and cleaned_indexes and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _update_split_manifest(manifest, cleaned_indexes)
        atomic_write_json(manifest_path, manifest)
        print("[filter_overlap] Updated split_manifest.json fingerprints/counts/scenes.")

    audits_dir.mkdir(parents=True, exist_ok=True)
    excluded_path = audits_dir / "excluded_overlap.json"
    summary = {
        "description": "Samples excluded because their visible image matches a Test image (SHA-256).",
        "test_images_hashed": len(test_hashes),
        "train_excluded": len(all_records.get("train", [])),
        "val_excluded": len(all_records.get("val", [])),
        "records": all_records,
    }
    atomic_write_json(excluded_path, summary)
    print(
        f"[filter_overlap] Audit log written to {excluded_path}"
        f" (train: {summary['train_excluded']}, val: {summary['val_excluded']})"
    )

    if not args.overwrite_indexes:
        print(
            "[filter_overlap] Dry run completed (no indexes modified). "
            "Use --overwrite-indexes to apply changes."
        )


if __name__ == "__main__":
    main()
