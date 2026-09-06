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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.census import pass_agreement  # noqa: E402
from foundry.io import load_json  # noqa: E402
from foundry.review.store import AnnotationStore  # noqa: E402

TEACHER_ANNOTATOR = "glm-4.6v"


def _trusted_objects(frame: dict) -> list[dict]:
    if frame.get("single_pass"):
        good = frame["findall_1"] if frame["findall_1"]["status"] == "completed" else frame["findall_2"]
        return good["objects"]
    return pass_agreement(frame["findall_1"]["objects"], frame["findall_2"]["objects"])["agreed_objects"]


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
            objects = sorted(_trusted_objects(frame), key=lambda o: o["bbox"][0])
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
                }
                if store.meta(item_id) is None:
                    store.set(item_id, [float(v) for v in obj["bbox"]], annotator=TEACHER_ANNOTATOR)
                    stats["seeded"] += 1
                else:
                    stats["already_seeded"] += 1
                items.append(item)
    return {
        "name": f"census-review:{metadata.get('run_id', census_run_dir.name)}",
        "items": items,
        "stats": stats,
        "census_run_id": metadata.get("run_id"),
        "split": split,
        "store": store,
    }
