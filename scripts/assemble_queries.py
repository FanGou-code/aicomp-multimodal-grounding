#!/usr/bin/env python3
"""Phase 2: assemble queries from a census run.

Consumes a census run directory (merged.json + the dataset index), selects
per-frame targets, and for each target lets code choose the dimension that
disambiguates it inside its own frame. The wording is one API call per target,
unless ``--no-realize`` is passed. Code verifies every sentence against the
facts it handed over.

The manifest is the input for human review.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from foundry.pipeline.api import (  # noqa: E402
    DEFAULT_KEY_FILE,
    OpenAIProtocolClient,
    comment_out_key,
    load_api_keys,
)
from foundry.pipeline.api import APIKeyPool  # noqa: E402
from foundry.pipeline.assembly import assemble_run, audit_assembly  # noqa: E402
from foundry.pipeline.realize import (  # noqa: E402
    REALIZE_PROMPT_HASH,
    REALIZE_STAGE,
    parse_realize_response,
    realize_messages,
)
from foundry.utils import (  # noqa: E402
    ANNOTATION_API_BASE_URL,
    ANNOTATION_MODEL_NAME,
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


def build_realizer(data_root: Path, index: dict):
    """One API call per target: the chosen dimension in, one sentence out."""
    from PIL import Image

    keys = load_api_keys()
    if not keys:
        raise SystemExit("No API keys found; write one per line into keys/api_keys.txt")
    key_file = Path(os.environ.get("ANNOTATION_API_KEY_FILE", str(DEFAULT_KEY_FILE)))
    pool = APIKeyPool(
        keys, notify=print,
        persist_retire=lambda key, reason: comment_out_key(key_file, key, reason),
    )
    client = OpenAIProtocolClient(
        key_pool=pool,
        model=ANNOTATION_MODEL_NAME,
        base_url=ANNOTATION_API_BASE_URL,
        timeout_seconds=180.0,
        rate_limiter=None,
        enable_thinking=None,
        thinking_mode=REALIZE_STAGE["thinking_mode"],
        json_mode=REALIZE_STAGE["response_format"] == "json_object",
    )
    cache: dict[str, object] = {}

    def realize(dimension, bbox, sample_id):
        if sample_id not in cache:
            with Image.open(data_root / index[sample_id]["visible"]) as opened:
                cache[sample_id] = opened.convert("RGB")
        url = _target_view(cache[sample_id], bbox)
        response = client.complete(
            messages=realize_messages(url, dimension),
            max_tokens=REALIZE_STAGE["max_tokens"],
            temperature=REALIZE_STAGE["temperature"],
            do_sample=REALIZE_STAGE["do_sample"],
        )
        return parse_realize_response(response.content)

    return realize


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
    parser.add_argument("--show", type=int, default=15, help="sample records to print")
    parser.add_argument("--all-frames", action="store_true",
                        help="mark this assembly as full-frame (review all frames, "
                             "not 1 per sequence). Metadata-only: assembly always "
                             "covers the census-selected frames; the flag drives the "
                             "downstream review session's sampling mode.")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into an existing output dir (default: refuse)")
    parser.add_argument("--no-realize", action="store_true",
                        help="skip the wording call; produces no records, only shortfall")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merged_path = args.census_run / "merged.json"
    if not merged_path.exists():
        raise SystemExit(f"merged.json not found under {args.census_run}")
    merged = load_json(merged_path)
    index = load_json(resolve_index_dir(args.data_root, args.index_dir) / f"{args.split}.json")
    realize = None if args.no_realize else build_realizer(args.data_root, index)
    result = assemble_run(
        merged,
        index,
        max_teacher_per_frame=args.max_teacher_per_frame,
        realize=realize,
    )
    text_edits = apply_text_qc(result.records)
    audit = audit_assembly(result.records) if result.records else {
        "count": 0, "sources": {"real": 0, "teacher": 0},
        "verbatim_repeat_rate": 0.0, "mean_words": 0.0,
    }

    metadata = merged.get("metadata", {})
    run_id = metadata.get("run_id", args.census_run.name)
    tag = args.run_tag or f"asm-{run_id}"
    out_dir = args.output_root / tag
    if out_dir.exists() and not args.force:
        raise SystemExit(f"output dir already exists: {out_dir} (pass --force to overwrite)")
    out_dir.mkdir(parents=True, exist_ok=True)
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
        print(f"  [{record.source:<7}] {record.query}   ({record.sample_id}, {record.family})")


if __name__ == "__main__":
    main()
