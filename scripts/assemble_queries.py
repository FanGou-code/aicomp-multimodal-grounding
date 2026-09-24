#!/usr/bin/env python3
"""Phase 2: assemble queries from a census run.

Consumes a census run directory (merged.json + the dataset index), selects
per-frame targets, and for each target writes out every fact that is true of
it inside its own frame. The wording is one API call per target,
unless ``--no-realize`` is passed. Nothing judges the wording: whatever comes
back goes to human review as-is.

The manifest is the input for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.pipeline.api import client_for, resolve_api_key, validate_concurrency  # noqa: E402
from foundry.pipeline.assembly import assemble_run, audit_assembly  # noqa: E402
from foundry.pipeline.realize import (  # noqa: E402
    REALIZE_PROMPT_HASH,
    REALIZE_STAGE,
    parse_realize_response,
    realize_messages,
)
from foundry.utils import (  # noqa: E402
    atomic_write_json,
    load_json,
    resolve_index_dir,
)
from foundry.pipeline.text_qc import apply_text_qc  # noqa: E402
from foundry.pipeline.views import jpeg_data_url  # noqa: E402


def _target_view(image, bbox) -> str:
    """The frame with one box drawn around this target only.

    Only the target is boxed: the teacher's job here is to look inside that box
    and describe it, not to re-count the scene.
    """
    from PIL import ImageDraw

    view = image.convert("RGB").copy()
    draw = ImageDraw.Draw(view)
    width, height = view.size
    x1, y1, x2, y2 = bbox
    draw.rectangle((x1 * width, y1 * height, x2 * width, y2 * height), outline=(220, 40, 40), width=3)
    return jpeg_data_url(view)


def build_realizer(data_root: Path, index: dict, api_key: str, retry: bool = True):
    """One API call per target: the target's facts in, one sentence out."""
    from PIL import Image

    client = client_for(REALIZE_STAGE, api_key=api_key, retry=retry)
    cache: dict[str, object] = {}

    def realize(facts, sample_id):
        if sample_id not in cache:
            with Image.open(data_root / index[sample_id]["visible"]) as opened:
                cache[sample_id] = opened.convert("RGB")
        url = _target_view(cache[sample_id], facts.bbox)
        response = client.complete(
            messages=realize_messages(url, facts),
            max_tokens=REALIZE_STAGE["max_tokens"],
            temperature=REALIZE_STAGE["temperature"],
            do_sample=REALIZE_STAGE["do_sample"],
        )
        return parse_realize_response(response.content)

    return realize


def load_sentences(path: Path) -> dict:
    """Wording already obtained in an earlier run, keyed by item id."""
    if not path.is_file():
        return {}
    done: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done[row["item_id"]] = row["query"]
    return done


def append_sentence(path: Path, item_id: str, sentence) -> None:
    """Persist one wording result as it arrives, so a crash costs nothing."""
    row = json.dumps({"item_id": item_id, "query": sentence}, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(row + "\n")


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
        required=True,
        help="dataset root holding the annotation-source index",
    )
    parser.add_argument(
        "--split", choices=("train", "val"), default="train",
        help="which index file to load for GT boxes",
    )
    parser.add_argument("--index-dir", type=Path, default=None)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "assembly")
    parser.add_argument("--max-teacher-per-frame", type=int, default=2,
                        help="teacher targets per frame; -1 = take all quality-sorted")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="parallel wording calls (default 8)")
    parser.add_argument("--api-key", default=None,
                        help="key for this run; defaults to $ANNOTATION_API_KEY")
    parser.add_argument("--show", type=int, default=15, help="sample records to print")
    parser.add_argument("--all-frames", action="store_true",
                        help="mark this assembly as full-frame (review all frames, "
                             "not 1 per sequence). Metadata-only: assembly always "
                             "covers the census-selected frames; the flag drives the "
                             "downstream review session's sampling mode.")
    parser.add_argument("--retry", action=argparse.BooleanOptionalAction, default=True,
                        help="retry transient failures before stopping (default: on)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="reuse the wording already in the output dir (default: on)")
    parser.add_argument("--force", action="store_true",
                        help="start over in an existing output dir, discarding its wording")
    parser.add_argument("--no-realize", action="store_true",
                        help="skip the wording call; produces no records, only shortfall")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_concurrency(args.concurrency)
    merged_path = args.census_run / "merged.json"
    if not merged_path.exists():
        raise SystemExit(f"merged.json not found under {args.census_run}")
    merged = load_json(merged_path)
    index = load_json(resolve_index_dir(args.data_root, args.index_dir) / f"{args.split}.json")

    metadata = merged.get("metadata", {})
    run_id = metadata.get("run_id", args.census_run.name)
    tag = args.run_tag or f"asm-{run_id}"
    out_dir = args.output_root / tag
    if out_dir.exists() and not (args.resume or args.force):
        raise SystemExit(
            f"output dir already exists: {out_dir} "
            "(pass --resume to continue it or --force to start over)"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    sentences_path = out_dir / "sentences.jsonl"
    if args.force and sentences_path.exists():
        sentences_path.unlink()
    sentences = load_sentences(sentences_path) if args.resume else {}
    if sentences:
        print(f"resuming: {len(sentences)} wording(s) already on file")

    realize = None if args.no_realize else build_realizer(
        args.data_root, index, resolve_api_key(args.api_key), retry=args.retry
    )
    result = assemble_run(
        merged,
        index,
        max_teacher_per_frame=args.max_teacher_per_frame,
        max_workers=args.concurrency,
        realize=realize,
        sentences=sentences,
        on_sentence=lambda item_id, sentence: append_sentence(sentences_path, item_id, sentence),
    )
    text_edits = apply_text_qc(result.records)
    audit = audit_assembly(result.records) if result.records else {
        "count": 0, "sources": {"real": 0, "teacher": 0},
        "verbatim_repeat_rate": 0.0, "mean_words": 0.0,
    }

    manifest = {
        "metadata": {
            "assembler_version": 1,
            "run_tag": tag,
            "census_run_id": run_id,
            "census_source_fingerprint": metadata.get("source_fingerprint"),
            "census_preparation_fingerprint": metadata.get("preparation_fingerprint"),
            "split": args.split,
            "all_frames": bool(args.all_frames),
            "max_teacher_per_frame": args.max_teacher_per_frame,
            "concurrency": args.concurrency,
            "realized": not args.no_realize,
            "realize_prompt_hash": REALIZE_PROMPT_HASH,
            "realize_stage": REALIZE_STAGE,
            "text_qc_edits": len(text_edits),
        },
        "records": [record.__dict__ for record in result.records],
        "shortfall": result.shortfall,
    }
    atomic_write_json(out_dir / "assembly.json", manifest)
    atomic_write_json(out_dir / "audit.json", audit)
    if text_edits:
        atomic_write_json(out_dir / "text_edits.json", text_edits)

    print(f"assembled {audit['count']} records "
          f"(real {audit['sources']['real']} / teacher {audit['sources']['teacher']}) "
          f"from {len(result.sequences)} sequences")
    print(f"verbatim repeat {audit['verbatim_repeat_rate']}, "
          f"mean words {audit['mean_words']}")
    print(f"shortfall: {len(result.shortfall)} target(s)")
    print(f"wrote {out_dir / 'assembly.json'}")
    print(f"wrote {out_dir / 'audit.json'}")
    print("\nsample records:")
    for record in result.records[: args.show]:
        print(f"  [{record.source:<7}] {record.query}   ({record.sample_id})")


if __name__ == "__main__":
    main()
