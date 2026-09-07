#!/usr/bin/env python3
"""Build a review manifest from an assembly.json or any query JSON.

Two input modes:
  --assembly  : from assembly.json (internal pipeline)
  --source    : from any {id, image, query} JSON (generic)

Supports splitting into parts for multi-user annotation.

Usage::

    # From assembly
    python scripts/make_manifest.py --assembly outputs/assembly/asm-train-r5/assembly.json \
        --data-root /path/to/dataset

    # From any query JSON (generic — zero pipeline dependency)
    python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images

    # Split into parts
    python scripts/make_manifest.py --source my_queries.json --images-root /path/to/images \
        --split 2 --part 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_manifest_from_assembly(
    assembly_path: Path,
    data_root: Path,
    *,
    split: int = 0,
    part: int = 0,
    out: Path | None = None,
) -> dict:
    """Build a review manifest from assembly records."""
    manifest = json.loads(assembly_path.read_text(encoding="utf-8"))
    metadata = manifest.get("metadata", {})
    split_name = metadata.get("split", "train")
    run_tag = metadata.get("run_tag", assembly_path.parent.name)

    data_root = Path(data_root)
    index = json.loads((data_root / "indexes" / f"{split_name}.json").read_text(encoding="utf-8"))

    items: list[dict] = []
    for record in manifest.get("records", []):
        sample_id = record["sample_id"]
        entry = index.get(sample_id)
        if entry is None:
            continue
        item_id = f"{sample_id}#{record['object_index']:02d}"
        items.append({
            "id": item_id,
            "image": entry["visible"],
            "query": record["query"],
            "bbox": record["bbox"],
            "category": record["category"],
            "frame_id": sample_id,
            "corpus": split_name,
            "source": record.get("source", "teacher"),
            "object_index": record["object_index"],
        })

    if split > 0 and 1 <= part <= split:
        chunk_size = (len(items) + split - 1) // split
        start = (part - 1) * chunk_size
        end = min(start + chunk_size, len(items))
        items = items[start:end]
        name = f"{run_tag}-part{part}of{split}"
    else:
        name = run_tag

    result = {"name": name, "run_tag": run_tag, "split": split_name, "items": items}
    if out:
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(out)
    return result


def build_manifest_from_source(
    source_path: Path,
    images_root: Path,
    *,
    split: int = 0,
    part: int = 0,
    out: Path | None = None,
) -> dict:
    """Build a review manifest from any query JSON.

    Supports two shapes:
      mapping: {"<id>": {"image": path, "query": text}, ...}
      list:    [{"id": "...", "image": path, "query": text}, ...]

    bbox is optional — null means manual annotation from scratch.
    """
    source = json.loads(source_path.read_text(encoding="utf-8"))
    images_root = Path(images_root).resolve()

    if isinstance(source, dict):
        entries = [{"id": k, **v} for k, v in source.items()]
    elif isinstance(source, list):
        entries = source
    else:
        raise ValueError("source must be a JSON object or array")

    items = []
    for entry in entries:
        item_id = entry.get("id", entry.get("item_id", ""))
        image = entry.get("image", entry.get("visible", ""))
        query = entry.get("query", "")
        bbox = entry.get("bbox")
        items.append({
            "id": str(item_id),
            "image": str(image),
            "query": str(query),
            "bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
            "category": entry.get("category", ""),
            "frame_id": entry.get("frame_id", str(item_id)),
            "corpus": "default",
            "source": "manual",
            "object_index": 0,
        })

    if split > 0 and 1 <= part <= split:
        chunk_size = (len(items) + split - 1) // split
        start = (part - 1) * chunk_size
        end = min(start + chunk_size, len(items))
        items = items[start:end]
        name = f"{source_path.stem}-part{part}of{split}"
    else:
        name = source_path.stem

    result = {"name": name, "run_tag": source_path.stem, "split": "default", "items": items}
    if out:
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, default=None,
                        help="path to assembly.json")
    parser.add_argument("--source", type=Path, default=None,
                        help="path to any query JSON (mapping or list shape)")
    parser.add_argument("--images-root", type=Path, default=None,
                        help="root directory for image paths (required with --source)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="dataset root holding indexes/ (required with --assembly)")
    parser.add_argument("--split", type=int, default=0,
                        help="divide into N parts (0 = no split)")
    parser.add_argument("--part", type=int, default=0,
                        help="select part I (1-based; requires --split)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path")
    args = parser.parse_args()

    if not args.assembly and not args.source:
        parser.error("either --assembly or --source is required")

    if args.source:
        if not args.images_root:
            parser.error("--images-root is required with --source")
        result = build_manifest_from_source(
            source_path=args.source, images_root=args.images_root,
            split=args.split, part=args.part, out=args.out,
        )
        print(f"manifest: {len(result['items'])} items, name={result['name']}")
        if args.out:
            print(f"wrote {args.out}")
        return

    if not args.data_root:
        parser.error("--data-root is required with --assembly")
    if args.split > 0 and not (1 <= args.part <= args.split):
        parser.error(f"--part must be 1..{args.split} when --split={args.split}")

    out = args.out
    if out is None:
        tag = args.assembly.parent.name
        out = Path(f"{tag}-part{args.part}of{args.split}.json") if args.split > 0 else Path(f"{tag}.json")

    result = build_manifest_from_assembly(
        assembly_path=args.assembly, data_root=args.data_root,
        split=args.split, part=args.part, out=out,
    )
    print(f"manifest: {len(result['items'])} items, run_tag={result['run_tag']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
