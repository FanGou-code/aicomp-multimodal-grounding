#!/usr/bin/env python3
"""Apply human review edits to r5 assembly records.

Produces ``asm-*-r6`` as the packaging baseline for the R6 chain.

Merge: human-edited query text, human-corrected target boxes and human-confirmed
absence verdicts (from the review store snapshots) take priority; un-reviewed
items keep their original assembled values. Re-validates bucket classifier
and text QC for changed queries. Detects same-frame collisions.

Output: ``outputs/assembly/asm-{train,val}-r6/assembly.json``
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.pipeline.buckets import classify_frozen  # noqa: E402
from foundry.utils import (  # noqa: E402
    ANNOTATION_MODEL_NAME,
    atomic_write_json,
    load_json,
)
from foundry.review.store import AnnotationStore  # noqa: E402
from foundry.pipeline.text_qc import apply_text_qc  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly",
        type=Path,
        required=True,
        help="path to r5 assembly.json (outputs/assembly/asm-{train,val}-r5/assembly.json)",
    )
    parser.add_argument(
        "--review-queries",
        type=Path,
        nargs="+",
        default=None,
        help="path(s) to annotations.queries.json (auto-detected from assembly run_tag if omitted)",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="output run tag (default: asm-{split}-r6)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "assembly",
    )
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing output dir")
    parser.add_argument("--show", type=int, default=15,
                        help="sample records to print")
    return parser


def _load_human_queries(queries_paths: list[Path] | Path | None) -> dict[str, str]:
    if queries_paths is None:
        return {}
    if isinstance(queries_paths, (str, Path)):
        queries_paths = [Path(queries_paths)]
    merged: dict[str, str] = {}
    for p in queries_paths:
        p = Path(p)
        if p.is_file():
            merged.update(load_json(p))
    return merged


def _load_json_quiet(path: Path) -> dict:
    return load_json(path) if path.is_file() else {}


def _review_snapshots(queries_paths: list[Path]) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Discover box / absence snapshots next to each review queries file.

    The review store writes ``annotations.predictions.json`` (final box per
    annotated item) and ``annotations.absent.json`` (``item_id -> *:absent``)
    alongside ``annotations.queries.json``. All three must flow into r6:
    box overrides correct the teacher geometry, and only human-confirmed
    absences remove records — a ``{model}:absent`` verdict is still pending
    human review.
    """
    boxes: dict[str, list[float]] = {}
    absent: dict[str, str] = {}
    for path in queries_paths:
        boxes.update(_load_json_quiet(path.parent / "annotations.predictions.json"))
        absent.update(_load_json_quiet(path.parent / "annotations.absent.json"))
    return boxes, absent


def _human_confirmed_absent(absent: dict[str, str]) -> set[str]:
    pending = f"{ANNOTATION_MODEL_NAME}:absent"
    return {item_id for item_id, annotator in absent.items() if annotator != pending}


def _load_todo_items(queries_paths: list[Path]) -> set[str]:
    """Item ids whose final journal annotator carries a ``:todo`` suffix.

    Reviewers mark problematic / ambiguous items as todo — they stay on the
    pending disambiguation list and must not ship as clean ground truth, so
    apply_review excludes and flags them instead of carrying them into r6.
    If a subsequent review pass resolves the item with a clean human
    annotation, it is no longer marked todo.
    """
    todo: set[str] = set()
    seen_dirs: set[Path] = set()
    for path in queries_paths:
        data_dir = path.parent
        if data_dir in seen_dirs or not (data_dir / "annotations.jsonl").is_file():
            continue
        seen_dirs.add(data_dir)
        for item_id, meta in AnnotationStore(data_dir).all_meta().items():
            ann = meta.get("annotator") or ""
            if ann.endswith(":todo"):
                todo.add(item_id)
            elif ann and ann != "glm-4.6v":
                todo.discard(item_id)
    return todo


def _detect_collisions(records: list[dict]) -> set[str]:
    """Detect same-frame duplicate display queries after overlay."""
    by_frame: dict[str, dict[str, list[str]]] = {}
    for rec in records:
        sample_id = rec["sample_id"]
        query = rec["query"].lower()
        item_id = f"{sample_id}#{rec['object_index']:02d}"
        by_frame.setdefault(sample_id, {}).setdefault(query, []).append(item_id)
    collided: set[str] = set()
    for queries in by_frame.values():
        for ids in queries.values():
            if len(ids) > 1:
                collided.update(ids)
    return collided


def apply(assembly_path: Path, queries_path: list[Path] | Path | None,
          run_tag: str, output_root: Path, force: bool) -> dict:
    manifest = load_json(assembly_path)
    records: list[dict] = list(manifest["records"])
    metadata = dict(manifest["metadata"])
    split = metadata.get("split", "train")
    tag = run_tag or f"asm-{split}-r6"

    assembly_tag = metadata.get("run_tag", assembly_path.parent.name)
    if not queries_path:
        queries_paths = [PROJECT_ROOT / "outputs" / "review" / assembly_tag / "annotations.queries.json"]
    elif isinstance(queries_path, (str, Path)):
        queries_paths = [Path(queries_path)]
        for p in queries_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Review queries file not found: {p}")
    else:
        queries_paths = [Path(p) for p in queries_path]
        for p in queries_paths:
            if not p.is_file():
                raise FileNotFoundError(f"Review queries file not found: {p}")

    human_queries = _load_human_queries(queries_paths)
    human_boxes, absent = _review_snapshots(queries_paths)
    removed_absent = _human_confirmed_absent(absent)
    todo_items = _load_todo_items(queries_paths)

    # --- Merge ---
    stats: dict[str, int] = Counter(
        total=len(records), human=0, original=0, collision=0, bucket_changed=0,
        qc_edited=0, box_seen=0, box_changed=0, absent_removed=0, todo_excluded=0,
    )
    flagged: list[dict] = []

    kept: list[dict] = []
    for rec in records:
        item_id = f"{rec['sample_id']}#{rec['object_index']:02d}"
        if item_id in removed_absent:
            # Human-confirmed false positive: do not carry it into r6.
            stats["absent_removed"] += 1
            flagged.append({"item_id": item_id, "reason": "absent",
                            "query": rec.get("query", "")})
            continue
        if item_id in todo_items:
            # Reviewer marked the item for disambiguation (ambiguous / poor
            # wording); do not ship it as clean ground truth in r6.
            stats["todo_excluded"] += 1
            flagged.append({"item_id": item_id, "reason": "todo",
                            "query": rec.get("query", "")})
            continue

        original_query = rec["query"]
        original_bucket = rec.get("bucket", "")
        source = "original"

        # Priority: human > original
        if item_id in human_queries:
            new_query = human_queries[item_id]
            source = "human"
            stats["human"] += 1
        else:
            new_query = original_query
            stats["original"] += 1

        rec["query"] = new_query
        rec["review_source"] = source
        rec["original_query"] = original_query if new_query != original_query else None

        # Re-classify bucket for changed queries
        if new_query != original_query:
            new_bucket = classify_frozen(new_query)
            if new_bucket != original_bucket:
                rec["bucket"] = new_bucket
                rec["original_bucket"] = original_bucket
                stats["bucket_changed"] += 1

        # Human-corrected target box (the review store snapshot is the final
        # box per annotated item; untouched seeds equal the assembly bbox, so
        # applying it is a no-op except where the reviewer adjusted the box).
        if item_id in human_boxes:
            original_bbox = rec.get("bbox")
            reviewed_bbox = human_boxes[item_id]
            stats["box_seen"] += 1
            if reviewed_bbox != original_bbox:
                rec["bbox"] = reviewed_bbox
                rec["original_bbox"] = original_bbox
                stats["box_changed"] += 1

        kept.append(rec)
    records = kept

    # --- Detect collisions after overlay ---
    collisions = _detect_collisions(records)
    for rec in records:
        item_id = f"{rec['sample_id']}#{rec['object_index']:02d}"
        rec["collision"] = item_id in collisions
        if rec["collision"]:
            stats["collision"] += 1
            flagged.append({"item_id": item_id, "reason": "collision",
                            "query": rec.get("query", "")})

    # --- Text QC pass ---
    # Build minimal AssemblyRecord-like objects for apply_text_qc
    class _QCRecord:
        __slots__ = ("sample_id", "object_index", "query", "edited")
        def __init__(self, sample_id, object_index, query, edited=False):
            self.sample_id = sample_id
            self.object_index = object_index
            self.query = query
            self.edited = edited

    qc_records = [_QCRecord(r["sample_id"], r["object_index"], r["query"], r.get("edited", False))
                  for r in records]
    edits = apply_text_qc(qc_records)
    for i, rec in enumerate(records):
        rec["query"] = qc_records[i].query
        rec["edited"] = qc_records[i].edited
    stats["qc_edited"] = len(edits)

    # --- Assemble output ---
    out_dir = output_root / tag
    if out_dir.exists() and not force:
        raise SystemExit(f"output dir already exists: {out_dir} (pass --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)

    new_metadata = {
        "assembler_version": 1,
        "run_tag": tag,
        "supersedes": assembly_tag,
        "census_run_id": metadata.get("census_run_id"),
        "census_source_fingerprint": metadata.get("census_source_fingerprint"),
        "census_preparation_fingerprint": metadata.get("census_preparation_fingerprint"),
        "split": split,
        "all_frames": metadata.get("all_frames"),
        "max_teacher_per_frame": metadata.get("max_teacher_per_frame"),
        "word_window": metadata.get("word_window"),
        "text_qc_edits": stats["qc_edited"],
        "apply_stats": dict(stats),
    }
    output = {
        "metadata": new_metadata,
        "records": records,
        "shortfall": manifest.get("shortfall", []),
        "flagged": flagged,
    }
    atomic_write_json(out_dir / "assembly.json", output)
    if edits:
        atomic_write_json(out_dir / "text_edits.json", edits)

    return {
        "tag": tag,
        "out_dir": out_dir,
        "stats": dict(stats),
        "records": records,
        "flagged": flagged,
        "edits": edits,
    }


def main() -> None:
    args = build_parser().parse_args()
    result = apply(
        assembly_path=args.assembly,
        queries_path=args.review_queries,
        run_tag=args.run_tag,
        output_root=args.output_root,
        force=args.force,
    )
    stats = result["stats"]
    print(f"applied review to {result['tag']}")
    print(f"  total:       {stats['total']:>5}")
    print(f"  human:       {stats['human']:>5}")
    print(f"  original:    {stats['original']:>5}")
    print(f"  collision:   {stats['collision']:>5}")
    print(f"  bucket_chg:  {stats['bucket_changed']:>5}")
    print(f"  qc_edited:   {stats['qc_edited']:>5}")
    print(f"  box_seen:    {stats['box_seen']:>5} | box_changed: {stats['box_changed']:>5}")
    print(f"  absent_rm:   {stats['absent_removed']:>5} | todo_excluded: {stats['todo_excluded']:>5}")
    if result["flagged"]:
        print(f"\nflagged for human adjudication ({len(result['flagged'])}):")
        for f in result["flagged"][:10]:
            print(f"  {f['item_id']}: {f['reason']}")
        if len(result["flagged"]) > 10:
            print(f"  ... and {len(result['flagged']) - 10} more")
    print(f"\nwrote {result['out_dir'] / 'assembly.json'}")

    if args.show:
        print("\nsample records:")
        for rec in result["records"][: args.show]:
            src = rec.get("review_source", "original")
            q = rec["query"]
            print(f"  [{src:<9}] {q}   ({rec['sample_id']}#{rec['object_index']:02d})")


if __name__ == "__main__":
    main()
