"""Deterministic scene-aware work selection and sharding."""

from __future__ import annotations

import random
from pathlib import PurePosixPath


def extract_scene_id(query_id: str, item: dict | None = None) -> str:
    """Return the sequence/scene identifier shared by related frames or queries."""
    prefix = query_id.split("_", 1)[0]
    if prefix:
        return prefix

    visible = item.get("visible") if isinstance(item, dict) else None
    if isinstance(visible, str):
        parts = PurePosixPath(visible.replace("\\", "/")).parts
        for part in parts:
            if part.isdigit():
                return part
    raise ValueError(f"Cannot derive scene ID for query {query_id!r}")


def group_keys_by_scene(keys: list[str], dataset: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for key in sorted(keys):
        if key not in dataset:
            raise KeyError(f"Dataset does not contain key {key!r}")
        scene_id = extract_scene_id(key, dataset[key])
        groups.setdefault(scene_id, []).append(key)
    return groups


def select_scene_ids(
    dataset: dict,
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[str]:
    """Select a global number of scenes before any worker sharding occurs."""
    scene_ids = sorted(group_keys_by_scene(list(dataset), dataset))
    if limit is None:
        return scene_ids
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    shuffled = list(scene_ids)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def shard_keys_by_scene(
    keys: list[str],
    dataset: dict,
    num_shards: int,
) -> list[list[str]]:
    """Keep scenes intact and balance shard sizes by query count."""
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    groups = group_keys_by_scene(keys, dataset)
    if not groups:
        return []

    effective_shards = min(num_shards, len(groups))
    shards: list[list[str]] = [[] for _ in range(effective_shards)]
    counts = [0] * effective_shards
    ordered_groups = sorted(groups.items(), key=lambda pair: (-len(pair[1]), pair[0]))

    for _scene_id, scene_keys in ordered_groups:
        shard_id = min(range(effective_shards), key=lambda idx: (counts[idx], idx))
        shards[shard_id].extend(scene_keys)
        counts[shard_id] += len(scene_keys)

    return [sorted(shard) for shard in shards]


def shard_scene_ids(
    scene_ids: list[str],
    dataset: dict,
    num_shards: int,
) -> list[list[str]]:
    """Balance explicit sequence IDs by their number of frames."""
    if not scene_ids:
        return []
    groups = group_keys_by_scene(list(dataset), dataset)
    unknown = set(scene_ids) - set(groups)
    if unknown:
        raise ValueError(f"Unknown scene IDs: {sorted(unknown)[:5]}")
    all_keys = [key for sid in scene_ids for key in groups[sid]]
    key_shards = shard_keys_by_scene(all_keys, dataset, num_shards)
    return [
        sorted(set(extract_scene_id(key) for key in shard))
        for shard in key_shards
    ]




from foundry.utils import stable_json_hash


def source_fingerprint(dataset: dict) -> str:
    
    identity = {}
    for key in sorted(dataset):
        item = dataset[key]
        identity[key] = {
            field: item.get(field)
            for field in ("visible", "infrared", "depth", "bbox", "width", "height")
        }
    return stable_json_hash(identity)
