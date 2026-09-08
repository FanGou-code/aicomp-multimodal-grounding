"""Census review session: teacher boxes pre-seeded for human verification.

One review item per teacher-enumerated object on the selected frames of a
census run. Teacher boxes are written into the crash-safe review store as
AI pre-annotations (``glm-4.6v``); the reviewer's adjustments overwrite them
under their own name. Seeding is idempotent — only missing item ids are
seeded, so a restarted server never clobbers a finished review.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Lazy import to keep tool layer zero-dependency for --manifest mode
def __trusted_objects(frame):
    from foundry.pipeline.census import trusted_objects
    return trusted_objects(frame)
from foundry.utils import ANNOTATION_MODEL_NAME, load_json  # noqa: E402
from foundry.review.store import AnnotationStore  # noqa: E402

#: Single source of truth for the seeding model's annotator label. Must stay
#: in sync with the census/package pipeline (foundry.utils.ANNOTATION_MODEL_NAME)
#: so switching the base model never orphans teacher-verdict detection here.
TEACHER_ANNOTATOR = ANNOTATION_MODEL_NAME


def build_census_session(census_run_dir: Path, data_root: Path, review_root: Path) -> dict:
    """Build review items from a census run and seed missing teacher boxes."""
    census_run_dir = Path(census_run_dir)
    data_root = Path(data_root)
    merged = load_json(census_run_dir / "merged.json")
    metadata = merged.get("metadata", {})
    split = metadata.get("split", "train")
    index = load_json(data_root / "indexes" / f"{split}.json")
    results = merged.get("results", {})
    store = AnnotationStore(Path(review_root) / str(metadata.get("run_id") or census_run_dir.name))
    existing_meta = store.all_meta()
    pending_seeds: list[tuple[str, list[float], str]] = []

    items: list[dict] = []
    stats = {"seeded": 0, "frames": 0, "already_seeded": 0}
    for sequence_id in sorted(results):
        sequence = results[sequence_id]
        if sequence.get("status") != "completed":
            continue
        for sample_id in sorted(sequence.get("selected") or []):
            frame = sequence["frames"].get(sample_id)
            if not frame or frame.get("status") != "completed":
                continue
            entry = index.get(sample_id)
            if entry is None:
                continue
            # Presentation order: left to right within the frame. The
            # cross-pass intersection's greedy order is not positionally
            # stable under IoU ties, so sort explicitly here.
            objects = sorted(__trusted_objects(frame), key=lambda o: o["bbox"][0])
            stats["frames"] += 1
            for obj in objects:
                item_id = f"{sample_id}#{obj['i']:02d}"
                item = {
                    "id": item_id,
                    "image": entry["visible"],
                    "query": f"#{obj['i']} {obj['category']}",
                    "ordinal": obj["i"],
                    "frame_id": sample_id,
                    "gt_bbox": entry["bbox"],
                    "category": obj["category"],
                    "corpus": split,
                }
                if item_id not in existing_meta:
                    pending_seeds.append((item_id, [float(v) for v in obj["bbox"]], TEACHER_ANNOTATOR))
                    stats["seeded"] += 1
                else:
                    stats["already_seeded"] += 1
                items.append(item)
    store.seed_many(pending_seeds)
    return {
        "name": f"census-review:{metadata.get('run_id', census_run_dir.name)}",
        "items": items,
        "stats": stats,
        "census_run_id": metadata.get("run_id"),
        "split": split,
        "store": store,
    }


def build_assembly_session(assembly_path: Path, data_root: Path, review_root: Path) -> dict:
    """Review items from an assembled query manifest (1 sampled frame/sequence).

    Each record becomes an item whose query text is the assembled sentence and
    whose seeded box is the assembled bbox (real records carry the organizer
    GT box). Per sequence exactly one selected frame is sampled (deterministic:
    lowest sample_id with records), matching the agreed sampling plan of one
    reviewed frame per sequence.
    """
    assembly_path = Path(assembly_path)
    data_root = Path(data_root)
    manifest = load_json(assembly_path)
    metadata = manifest.get("metadata", {})
    split = metadata.get("split", "train")
    index = load_json(data_root / "indexes" / f"{split}.json")
    by_sequence: dict[str, dict[str, list[dict]]] = {}
    for record in manifest.get("records", []):
        sequence_id = record["sequence_id"]
        frames = by_sequence.setdefault(sequence_id, {})
        frames.setdefault(record["sample_id"], []).append(record)

    # full mode: every frame with records (admin 2026-09-06); sampled mode:
    # one deterministic frame per sequence. Both group items by sequence.
    full_mode = bool(manifest.get("metadata", {}).get("all_frames"))
    if full_mode:
        chosen_frames = {sid: frames for sid, frames in by_sequence.items()}
    else:
        # Bind each sequence's own frame dict (f) in the comprehension; a bare
        # ``frames`` here would resolve to the outer loop's leftover variable
        # and bind every sequence to the last sequence's frames.
        chosen_frames = {sid: {min(f): f[min(f)]} for sid, f in by_sequence.items()}

    items: list[dict] = []
    stats = {"seeded": 0, "frames": 0, "already_seeded": 0}
    run_tag = str(metadata.get("run_tag") or assembly_path.parent.name)
    store = AnnotationStore(Path(review_root) / run_tag)
    existing_meta = store.all_meta()
    pending_seeds: list[tuple[str, list[float], str]] = []
    for sequence_id in sorted(chosen_frames):
        for sample_id in sorted(chosen_frames[sequence_id]):
            records = sorted(chosen_frames[sequence_id][sample_id], key=lambda r: r["object_index"])
            entry = index.get(sample_id)
            if entry is None:
                continue
            stats["frames"] += 1
            for record in records:
                item_id = f"{sample_id}#{record['object_index']:02d}"
                query = record["query"]
                item = {
                    "id": item_id,
                    "image": entry["visible"],
                    "query": query,
                    "ordinal": record["object_index"],
                    "frame_id": sample_id,
                    # Only real records have a GT reference; teacher records are
                    # judged on their own.
                    "gt_bbox": record["bbox"] if record["source"] == "real" else None,
                    "category": record["category"],
                    "bucket": record.get("bucket", ""),
                    "family": record.get("family", ""),
                    "corpus": split,

                }
                meta = existing_meta.get(item_id)
                if meta is None or meta.get("annotator") == TEACHER_ANNOTATOR:
                    # 新条目播种; 上一轮种子未被动过则同步到最新组装框。
                    pending_seeds.append(
                        (item_id, [float(v) for v in record["bbox"]], TEACHER_ANNOTATOR)
                    )
                    stats["seeded"] += 1
                else:
                    stats["already_seeded"] += 1
                items.append(item)
    store.seed_many(pending_seeds)
    return {
        "name": f"assembly-review:{metadata.get('run_tag', assembly_path.parent.name)}",
        "items": items,
        "stats": stats,
        "census_run_id": metadata.get("census_run_id"),
        "split": split,
        "store": store,
    }
