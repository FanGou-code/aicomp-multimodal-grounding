#!/usr/bin/env python3
"""Review adjustment report — per-sequence view of human modifications.

Read-only over the crash-safe review store: for every review item it compares
the teacher-seeded box against the final human box and groups adjustments by
sequence, so scene-level problems (many/strong adjustments in one sequence)
surface for targeted re-review ("连坐" decisions). Zero API, zero writes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.bbox import compute_iou  # noqa: E402
from foundry.census import trusted_objects  # noqa: E402
from foundry.io import load_json  # noqa: E402
from foundry.review.census_session import TEACHER_ANNOTATOR  # noqa: E402
from foundry.review.store import AnnotationStore  # noqa: E402


def is_human(annotator: object) -> bool:
    return isinstance(annotator, str) and bool(annotator) and annotator != TEACHER_ANNOTATOR


def build_report(census_run_dir: Path, review_root: Path) -> dict:
    merged = load_json(Path(census_run_dir) / "merged.json")
    metadata = merged.get("metadata", {})
    run_id = str(metadata.get("run_id") or Path(census_run_dir).name)
    results = merged.get("results", {})
    store = AnnotationStore(Path(review_root) / run_id)

    sequences: dict[str, dict] = {}
    for sequence_id in sorted(results):
        sequence = results[sequence_id]
        if sequence.get("status") != "completed":
            continue
        entry = sequences.setdefault(
            sequence_id, {"total": 0, "adjusted": 0, "items": []}
        )
        for sample_id in sorted(sequence.get("selected") or []):
            frame = sequence["frames"].get(sample_id)
            if not frame or frame.get("status") != "completed":
                continue
            objects = sorted(trusted_objects(frame), key=lambda o: o["bbox"][0])
            for obj in objects:
                item_id = f"{sample_id}#{obj['i']:02d}"
                teacher_box = [float(v) for v in obj["bbox"]]
                final_box = store.get(item_id)
                meta = store.meta(item_id) or {}
                human = is_human(meta.get("annotator"))
                entry["total"] += 1
                record = {
                    "item_id": item_id,
                    "category": obj["category"],
                    "teacher_box": teacher_box,
                    "final_box": final_box,
                    "annotator": meta.get("annotator"),
                    "human_adjusted": bool(human and final_box is not None),
                }
                if record["human_adjusted"]:
                    record["iou_to_teacher"] = round(compute_iou(teacher_box, final_box), 3)
                    entry["adjusted"] += 1
                entry["items"].append(record)

    totals = {
        "run_id": run_id,
        "total_items": sum(s["total"] for s in sequences.values()),
        "human_adjusted": sum(s["adjusted"] for s in sequences.values()),
        "sequences_affected": sum(1 for s in sequences.values() if s["adjusted"]),
    }
    return {"totals": totals, "sequences": sequences}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-run", required=True, help="census run dir with merged.json")
    parser.add_argument("--review-root", default=PROJECT_ROOT / "outputs" / "review")
    parser.add_argument("--json", action="store_true", help="dump full JSON to stdout")
    parser.add_argument("--min-adjust", type=int, default=1,
                        help="only list sequences with at least this many adjustments")
    args = parser.parse_args(argv)

    report = build_report(Path(args.census_run), Path(args.review_root))
    totals = report["totals"]
    print(f"run={totals['run_id']}  人核/调整 {totals['human_adjusted']}/{totals['total_items']}"
          f"  涉及序列 {totals['sequences_affected']}")
    flagged = 0
    for sequence_id, entry in report["sequences"].items():
        if entry["adjusted"] < args.min_adjust:
            continue
        flagged += 1
        print(f"\n[{sequence_id}] 调整 {entry['adjusted']}/{entry['total']}  ← 连坐候选")
        for record in entry["items"]:
            if not record["human_adjusted"]:
                continue
            print(f"  {record['item_id']}  {record['category']:<14}"
                  f" IoU(师,人)={record['iou_to_teacher']}")
    if not flagged:
        print("\n无调整记录（或低于 --min-adjust 阈值）")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
