#!/usr/bin/env python3
"""Phase 2: assemble v5 queries from a census run — local only, zero API calls.

Consumes a census run directory (merged.json + the dataset index) and the
frozen style spec, selects per-frame targets (1 real + up to N teacher
objects), and emits code-assembled, fact-supported, uniqueness-verified
query records plus an acceptance audit. Nothing here publishes an
annotation run; the manifest is the input for human review and the Phase 3
planner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.assembly import assemble_run, audit_assembly, classify_frozen  # noqa: E402
from foundry.io import atomic_write_json, load_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--census-run",
        type=Path,
        required=True,
        help="census run directory containing merged.json (outputs/census/<run_id>)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="dataset root holding the annotation-source index",
    )
    parser.add_argument("--spec", type=Path, default=PROJECT_ROOT / "spec" / "style_spec.json")
    parser.add_argument(
        "--split", choices=("train", "val"), default="train",
        help="which index file to load for GT boxes",
    )
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "assembly")
    parser.add_argument("--max-teacher-per-frame", type=int, default=2,
                        help="teacher targets per frame; -1 = take all quality-sorted")
    parser.add_argument("--min-words", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=18)
    parser.add_argument("--show", type=int, default=15, help="sample records to print")
    parser.add_argument("--all-frames", action="store_true",
                        help="keep every frame with records (review all, not 1/sequence)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merged_path = args.census_run / "merged.json"
    if not merged_path.exists():
        raise SystemExit(f"merged.json not found under {args.census_run}")
    merged = load_json(merged_path)
    index = load_json(args.data_root / "indexes" / f"{args.split}.json")
    spec = load_json(args.spec) if args.spec.exists() else None
    enumeration = None
    enum_path = args.census_run / "enumeration.json"
    if enum_path.exists():
        enumeration = load_json(enum_path).get("results", {})
        print(f"enumeration pass loaded: {len(enumeration)} frames")

    result = assemble_run(
        merged,
        index,
        spec,
        enumeration=enumeration,
        max_teacher_per_frame=args.max_teacher_per_frame,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    audit = audit_assembly(result.records)

    metadata = merged.get("metadata", {})
    run_id = metadata.get("run_id", args.census_run.name)
    tag = args.run_tag or f"asm-{run_id}"
    out_dir = args.output_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "metadata": {
            "assembler_version": 1,
            "run_tag": tag,
            "census_run_id": run_id,
            "census_source_fingerprint": metadata.get("source_fingerprint"),
            "census_preparation_fingerprint": metadata.get("preparation_fingerprint"),
            "spec_status": (spec or {}).get("status"),
            "spec_version": (spec or {}).get("version"),
            "split": args.split,
            "all_frames": bool(args.all_frames),
            "max_teacher_per_frame": args.max_teacher_per_frame,
            "word_window": [args.min_words, args.max_words],
        },
        "records": [record.__dict__ for record in result.records],
        "shortfall": result.shortfall,
    }
    atomic_write_json(out_dir / "assembly.json", manifest)
    atomic_write_json(out_dir / "audit.json", audit)

    print(f"assembled {audit['count']} records "
          f"(real {audit['sources']['real']} / teacher {audit['sources']['teacher']}) "
          f"from {len(result.sequences)} sequences")
    print(f"verbatim repeat {audit['verbatim_repeat_rate']} (test {audit['test_reference']['verbatim_repeat_rate']}), "
          f"mean words {audit['mean_words']} (test {audit['test_reference']['mean_words']})")
    print("bucket per-mille (assembled vs test):")
    for bucket, per_mille in audit["bucket_per_mille"].items():
        print(f"  {bucket:<18} {per_mille:>4} vs "
              f"{audit['test_reference']['bucket_per_mille'][bucket]:>4}")
    print(f"shortfall: {len(result.shortfall)} target(s)")
    print(f"wrote {out_dir / 'assembly.json'}")
    print(f"wrote {out_dir / 'audit.json'}")
    print("\nsample records:")
    for record in result.records[: args.show]:
        print(f"  [{record.source:<7}] {record.query}   ({record.sample_id}, {record.family})")
    buckets_seen = sorted({record.bucket for record in result.records})
    print("\nclassifier sanity (frozen rule):")
    for bucket in buckets_seen:
        example = next(r for r in result.records if r.bucket == bucket)
        print(f"  {bucket}: {example.query} -> {classify_frozen(example.query)}")


if __name__ == "__main__":
    main()
