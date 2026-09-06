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

from foundry.assembly import extract_frame_facts  # noqa: E402
from foundry.facts import category_head  # noqa: E402
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


def _claim_summary(record: dict, facts) -> str:
    """Human-readable claim sheet: 对象/颜色/特征/序数/方位/深度/面积."""
    parts = [f"对象:{record['category']}"]
    if facts.color:
        parts.append(f"颜色:{facts.color}")
    if facts.features:
        parts.append(f"特征:{facts.features}")
    if facts.rank_left and facts.rank_right:
        parts.append(
            f"序数:左起第{facts.rank_left}·右起第{facts.rank_right}(共{facts.count_in_head})"
        )
    elif facts.rank_left:
        parts.append(f"序数:左起第{facts.rank_left}/{facts.count_in_head}")
    elif facts.rank_right:
        parts.append(f"序数:右起第{facts.rank_right}/{facts.count_in_head}")
    sp = []
    if facts.is_leftmost:
        sp.append("极左")
    if facts.is_rightmost:
        sp.append("极右")
    if facts.is_topmost:
        sp.append("最顶")
    if facts.is_bottommost:
        sp.append("最底")
    if facts.side_of_image:
        sp.append("画幅左侧" if facts.side_of_image == "left" else "画幅右侧")
    if facts.anchors_left:
        sp.append(f"在{category_head(facts.anchors_left[0][1])}的左边")
    if facts.anchors_right:
        sp.append(f"在{category_head(facts.anchors_right[0][1])}的右边")
    if sp:
        parts.append("方位:" + "/".join(sp))
    dp = []
    if facts.is_closest:
        dp.append("最近")
    if facts.is_farthest:
        dp.append("最远")
    if facts.is_in_foreground:
        dp.append("前景")
    if facts.is_in_background:
        dp.append("背景")
    if facts.median_mm:
        dp.append(f"{facts.median_mm}mm")
    if dp:
        parts.append("深度:" + "/".join(dp))
    if facts.area_ratio_lead and facts.area_ratio_lead >= 1.5:
        parts.append(f"面积:同类{facts.area_ratio_lead:.1f}倍")
    return " · ".join(parts)


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
    census_merged = None
    census_run_id = metadata.get("census_run_id")
    census_path = Path(PROJECT_ROOT) / "outputs" / "census" / str(census_run_id) / "merged.json"
    if census_path.is_file():
        census_merged = load_json(census_path)

    by_sequence: dict[str, dict[str, list[dict]]] = {}
    for record in manifest.get("records", []):
        sequence_id = record["sequence_id"]
        frames = by_sequence.setdefault(sequence_id, {})
        frames.setdefault(record["sample_id"], []).append(record)

    items: list[dict] = []
    stats = {"seeded": 0, "frames": 0, "already_seeded": 0}
    store = AnnotationStore(Path(review_root) / str(metadata.get("run_tag") or assembly_path.parent.name))
    for sequence_id in sorted(by_sequence):
        frames = by_sequence[sequence_id]
        sample_id = min(frames)  # deterministic: first sampled frame with records
        records = sorted(frames[sample_id], key=lambda r: r["object_index"])
        entry = index.get(sample_id)
        if entry is None:
            continue
        facts_by_index = {}
        if census_merged is not None:
            seq = census_merged.get("results", {}).get(sequence_id, {})
            frame = seq.get("frames", {}).get(sample_id, {})
            if frame.get("status") == "completed":
                if frame.get("single_pass"):
                    good = (frame["findall_1"] if frame["findall_1"]["status"] == "completed"
                            else frame["findall_2"])
                    objects = good["objects"]
                else:
                    objects = pass_agreement(
                        frame["findall_1"]["objects"], frame["findall_2"]["objects"]
                    )["agreed_objects"]
                attr = frame.get("attr")
                depth = frame.get("depth")
                if objects:
                    frame_facts = extract_frame_facts(objects, entry["bbox"], attr, depth)
                    facts_by_index = {f.index: f for f in frame_facts}
        stats["frames"] += 1
        for record in records:
            item_id = f"{sample_id}#{record['object_index']:02d}"
            item = {
                "id": item_id,
                "image": entry["visible"],
                "query": record["query"],
                "ordinal": record["object_index"],
                "frame_id": sample_id,
                # Only real records have a GT reference; teacher records are
                # judged on their own.
                "gt_bbox": record["bbox"] if record["source"] == "real" else None,
                "category": record["category"],
                "bucket": record.get("bucket", ""),
                "family": record.get("family", ""),
                "claims": _claim_summary(record, facts_by_index[record["object_index"]])
                if record["object_index"] in facts_by_index else f"对象:{record['category']}",
            }
            if store.meta(item_id) is None:
                store.set(item_id, [float(v) for v in record["bbox"]], annotator=TEACHER_ANNOTATOR)
                stats["seeded"] += 1
            else:
                stats["already_seeded"] += 1
            items.append(item)
    return {
        "name": f"assembly-review:{metadata.get('run_tag', assembly_path.parent.name)}",
        "items": items,
        "stats": stats,
        "census_run_id": metadata.get("census_run_id"),
        "split": split,
        "store": store,
    }
