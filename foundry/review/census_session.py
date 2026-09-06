"""Census review session: teacher boxes pre-seeded for human verification.

One review item per teacher-enumerated object on the selected frames of a
census run. Teacher boxes are written into the crash-safe review store as
AI pre-annotations (``glm-4.6v``); the reviewer's adjustments overwrite them
under their own name. Seeding is idempotent — only missing item ids are
seeded, so a restarted server never clobbers a finished review.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.facts import category_head, extract_frame_facts  # noqa: E402
from foundry.census import trusted_objects  # noqa: E402
from foundry.io import load_json  # noqa: E402
from foundry.review.store import AnnotationStore  # noqa: E402

TEACHER_ANNOTATOR = "glm-4.6v"


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
            objects = sorted(trusted_objects(frame), key=lambda o: o["bbox"][0])
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


SUPERLATIVE_PHRASE = {
    "y2-max": "closest to the camera", "y2-min": "farthest from the camera",
    "x-min": "◀◀ far left", "x-max": "far right ▶▶",
    "y-min": "topmost", "y-max": "bottommost",
}
ORDINAL_WORD_RE = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", re.I
)


def _claim_summary(record: dict, facts) -> str:
    """方向 · 序数 · 主体 — 主体直接摘自 query 本身, 保证与句子同源."""
    f = record.get("facts") or []
    q = record.get("query", "")
    parts = []
    if f and f[0].startswith("rank:"):
        direction = f[1] if len(f) > 1 else ""
        parts.append("▶" if direction in ("from left to right", "from the left") else "◀")
    elif f and f[0] in SUPERLATIVE_PHRASE:
        parts.append(SUPERLATIVE_PHRASE[f[0]])
    elif f and f[0] == "image:left":
        parts.append("◀ side of image")
    elif f and f[0] == "image:right":
        parts.append("▶ side of image")
    elif f and f[0].startswith("anchor-left:"):
        parts.append("◀ of the " + category_head(f[0].split(":")[2]))
    elif f and f[0].startswith("anchor-right:"):
        parts.append("▶ of the " + category_head(f[0].split(":")[2]))
    m = ORDINAL_WORD_RE.search(q)
    if m:
        parts.append(m.group(1).lower())
    # 主体 = query 主语名词短语 (修饰词+头名词), 与句子逐字同源。
    # 先剥掉方向短语尾巴, 再从前往后收词到头名词为止。
    q_body = re.sub(
        r",?\s*(from|to) the (left|right)( to the (left|right))?$",
        "", q, flags=re.I,
    )
    q_body = re.sub(
        r",?\s*from (left|right) to (left|right)$", "", q_body, flags=re.I,
    )
    tokens = q_body.split()
    stop = {"with", "wearing", "holding", "carrying", "in", "on", "from", "to",
            "perched", "of", "by", "near", "the", "a", "an", "and"}
    phrase: list[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in stop and phrase:
            break
        phrase.append(tok)
    subject = " ".join(phrase)
    if subject[:2].lower() in ("a ", "an"):
        subject = subject[3:]
    elif subject[:4].lower() == "the ":
        subject = subject[4:]
    parts.append(subject)
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

    # full mode: every frame with records (admin 2026-09-06); sampled mode:
    # one deterministic frame per sequence. Both group items by sequence.
    full_mode = bool(manifest.get("metadata", {}).get("all_frames"))
    if full_mode:
        chosen_frames = {sid: frames for sid, frames in by_sequence.items()}
    else:
        chosen_frames = {sid: {min(frames): frames[min(frames)]} for sid in by_sequence}

    items: list[dict] = []
    stats = {"seeded": 0, "frames": 0, "already_seeded": 0}
    store = AnnotationStore(Path(review_root) / str(metadata.get("run_tag") or assembly_path.parent.name))
    existing_meta = store.all_meta()
    pending_seeds: list[tuple[str, list[float], str]] = []
    for sequence_id in sorted(chosen_frames):
        for sample_id in sorted(chosen_frames[sequence_id]):
            records = sorted(chosen_frames[sequence_id][sample_id], key=lambda r: r["object_index"])
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
