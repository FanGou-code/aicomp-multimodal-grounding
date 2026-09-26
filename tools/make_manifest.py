#!/usr/bin/env python3
"""Build a review manifest from any query JSON.

Input shape (mapping or list)::

    mapping: {"<id>": {"image"|"visible": path, "query": text, "bbox"?: [x1,y1,x2,y2]}, ...}
    list:    [{"id": "...", "image": path, "query": text, "bbox"?: [...]}, ...]

``bbox`` is optional — omitting it means manual annotation from scratch.
Supports splitting into parts for multi-user annotation.

Usage::

    python tools/make_manifest.py --source my_queries.json --images-root /path/to/images

    # Split into parts
    python tools/make_manifest.py --source my_queries.json --images-root /path/to/images \
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


def build_manifest_from_source(
    source_path: Path,
    images_root: Path | None = None,
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
    images_root_path = Path(images_root).resolve() if images_root is not None else None

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
        if images_root_path is not None and not Path(str(image)).is_absolute():
            candidate = images_root_path / str(image)
            if candidate.is_file():
                image = str(candidate)
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
    parser.add_argument("--source", type=Path, required=True,
                        help="path to any query JSON (mapping or list shape)")
    parser.add_argument("--images-root", type=Path, default=None,
                        help="optional root directory for image paths")
    parser.add_argument("--split", type=int, default=0,
                        help="divide into N parts (0 = no split)")
    parser.add_argument("--part", type=int, default=0,
                        help="select part I (1-based; requires --split)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output path")
    args = parser.parse_args()

    if args.split > 0 and not (1 <= args.part <= args.split):
        parser.error(f"--part must be 1..{args.split} when --split={args.split}")

    result = build_manifest_from_source(
        source_path=args.source, images_root=args.images_root,
        split=args.split, part=args.part, out=args.out,
    )
    print(f"manifest: {len(result['items'])} items, name={result['name']}")
    if args.out:
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
