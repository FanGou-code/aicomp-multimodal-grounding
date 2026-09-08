#!/usr/bin/env python3
"""Build train/val split indexes from the raw dataset.

Two-stage pipeline:
  1. Hash dedup — exclude train samples whose visible image matches a Test
     image (SHA-256). Exclusion evidence is written to the experiment unit.
  2. Split — assign remaining samples to train/val at the sequence level.

The split assignment is frozen to match the current production indexes
(seed=42, train_ratio=0.8). Output indexes are clean: sample_id → {visible,
infrared, depth, bbox, width, height}. No query field — queries are produced
downstream by the assembly pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def _build_split(sequences: list[str], seed: int, train_ratio: float) -> tuple[list[str], list[str]]:
    """Deterministic sequence-level split."""
    import random
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    if not 0 < split_idx < len(shuffled):
        raise ValueError(f'train_ratio={train_ratio} produces empty split')
    return sorted(shuffled[:split_idx]), sorted(shuffled[split_idx:])


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_groundtruth(path: Path) -> dict[str, tuple[int, int, int, int]]:
    """Parse groundtruth.txt: filename,x,y,width,height per line."""
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        filename, x, y, w, h = parts
        entries[filename] = (int(float(x)), int(float(y)), int(float(w)), int(float(h)))
    return entries


def _normalize_bbox(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> list[float] | None:
    """Normalize pixel bbox to 0-1 XYXY; return None if invalid."""
    x2, y2 = x + w, y + h
    if x < 0 or y < 0 or x2 > img_w or y2 > img_h or w <= 0 or h <= 0:
        return None
    return [x / img_w, y / img_h, x2 / img_w, y2 / img_h]


import struct


def _png_size(p: Path) -> tuple[int, int]:
    with p.open("rb") as fh:
        sig = fh.read(8)
        if sig[0] != 0x89 or sig[1:4] != b"PNG":
            raise ValueError("not a PNG")
        fh.read(4)
        tag = fh.read(4)
        if tag != b"IHDR":
            raise ValueError("IHDR not first")
        w = struct.unpack(">I", fh.read(4))[0]
        h = struct.unpack(">I", fh.read(4))[0]
        return w, h


def _collect_test_hashes(
    *,
    test_hashes_path: Path | None = None,
    test_images_dir: Path | None = None,
    raw_root: Path | None = None,
) -> dict[str, list[str]]:
    """Return {sha256 -> [test_image_stems]} mapping."""
    if test_hashes_path and test_hashes_path.is_file():
        raw_data = json.loads(test_hashes_path.read_text(encoding="utf-8"))
        if isinstance(raw_data, dict):
            first_val = next(iter(raw_data.values()), None)
            if isinstance(first_val, list):
                return {k: [str(s) for s in v] for k, v in raw_data.items()}
            elif isinstance(first_val, str) and len(first_val) == 64:
                mapping: dict[str, list[str]] = {}
                for stem, h in raw_data.items():
                    mapping.setdefault(h, []).append(stem)
                return mapping
            elif isinstance(next(iter(raw_data.keys()), ""), str) and len(next(iter(raw_data.keys()), "")) == 64:
                return {k: [str(v)] if not isinstance(v, list) else [str(x) for x in v] for k, v in raw_data.items()}
        elif isinstance(raw_data, list):
            return {str(h): [] for h in raw_data}

    # Auto-detect Test images directory
    candidates: list[Path] = []
    if test_images_dir:
        candidates.append(test_images_dir)
    if raw_root:
        candidates.extend([
            raw_root / "Test" / "Images" / "visible",
            raw_root / "Test" / "visible",
            raw_root / "Test" / "color",
        ])

    for cdir in candidates:
        if cdir.is_dir():
            mapping = {}
            for p in sorted(cdir.iterdir()):
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    mapping.setdefault(_hash_file(p), []).append(p.stem)
            if mapping:
                return mapping
    return {}


def build_indexes(
    raw_root: Path,
    *,
    out_dir: Path,
    dry_run: bool = False,
    test_hashes_path: Path | None = None,
    test_images_dir: Path | None = None,
    seed: int = 42,
    train_ratio: float = 0.8,
) -> dict:
    """Build train/val indexes from raw data.

    Returns stats dict with keys: sequences, samples, excluded_hash, train, val.
    """
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw data root not found: {raw_root}")

    raw_train = raw_root / "Train"
    if not raw_train.is_dir():
        raise FileNotFoundError(f"Train directory not found under {raw_root}")

    # --- Stage 1: hash dedup against test images ---
    test_hashes = _collect_test_hashes(
        test_hashes_path=test_hashes_path,
        test_images_dir=test_images_dir,
        raw_root=raw_root,
    )

    excluded_records: dict[str, list[dict]] = {"train": [], "val": []}
    all_sequences = sorted(
        p.name for p in raw_train.iterdir() if p.is_dir() and p.name.isdigit()
    )
    train_seqs, val_seqs = _build_split(all_sequences, seed, train_ratio)
    train_set = set(train_seqs)
    val_set = set(val_seqs)

    outputs: dict[str, dict] = {"train": {}, "val": {}}
    stats = {"sequences": len(all_sequences), "samples": 0, "excluded_hash": 0,
             "excluded_invalid_bbox": 0, "train": 0, "val": 0}

    for seq in all_sequences:
        seq_root = raw_train / seq
        gt_path = seq_root / "groundtruth.txt"
        if not gt_path.is_file():
            continue
        annotations = _parse_groundtruth(gt_path)
        target = outputs["train"] if seq in train_set else outputs["val"]

        for filename, (x, y, w, h) in sorted(annotations.items()):
            stem = Path(filename).stem
            sample_id = f"{seq}_{stem}"

            visible_path = seq_root / "color" / filename

            # Validate image existence
            infrared_path = seq_root / "infrared" / filename
            depth_path = seq_root / "depth" / filename
            if not visible_path.is_file() or not infrared_path.is_file() or not depth_path.is_file():
                continue

            # Read image dimensions
            try:
                img_w, img_h = _png_size(visible_path)
            except Exception:
                continue

            bbox = _normalize_bbox(x, y, w, h, img_w, img_h)
            if bbox is None:
                stats["excluded_invalid_bbox"] += 1
                continue

            # Hash dedup against test images
            if test_hashes and visible_path.is_file():
                vis_hash = _hash_file(visible_path)
                if vis_hash in test_hashes:
                    split_key = "train" if seq in train_set else "val"
                    excluded_records[split_key].append({
                        "sample_id": sample_id,
                        "visible": f"Train/{seq}/color/{filename}",
                        "test_images": sorted(set(test_hashes[vis_hash])),
                    })
                    stats["excluded_hash"] += 1
                    continue

            target[sample_id] = {
                "visible": f"Train/{seq}/color/{filename}",
                "infrared": f"Train/{seq}/infrared/{filename}",
                "depth": f"Processed/Train/{seq}/depth_jet/{filename}",
                "bbox": bbox,
                "width": img_w,
                "height": img_h,
            }
            stats["samples"] += 1

    stats["train"] = len(outputs["train"])
    stats["val"] = len(outputs["val"])

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "val"):
            path = out_dir / f"{split_name}.json"
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                json.dumps(outputs[split_name], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(path)

        manifest = {
            "status": "complete",
            "split_method": "frozen-sequence-assignment",
            "train_sequences": sorted(train_set),
            "val_sequences": sorted(val_set),
            "index_fingerprints": {
                "train": hashlib.sha256(
                    json.dumps(outputs["train"], sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
                "val": hashlib.sha256(
                    json.dumps(outputs["val"], sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            },
            "index_sample_counts": {"train": stats["train"], "val": stats["val"]},
            "preparation_protocol_version": 2,
        }
        manifest_path = out_dir / "split_manifest.json"
        tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        tmp.replace(manifest_path)

        total_excluded = len(excluded_records["train"]) + len(excluded_records["val"])
        if total_excluded > 0:
            audit_data = {
                "description": "Samples excluded because their visible image matches a Test image (SHA-256).",
                "test_images_hashed": len(test_hashes),
                "train_excluded": len(excluded_records["train"]),
                "val_excluded": len(excluded_records["val"]),
                "records": excluded_records,
            }
            overlap_path = out_dir / "excluded_overlap.json"
            tmp = overlap_path.with_name(overlap_path.name + ".tmp")
            tmp.write_text(
                json.dumps(audit_data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(overlap_path)

            excl_path = out_dir / "excluded.json"
            tmp_excl = excl_path.with_name(excl_path.name + ".tmp")
            tmp_excl.write_text(
                json.dumps(audit_data, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp_excl.replace(excl_path)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True,
                        help="path to dataset root (contains Train/ and optionally Test/)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for train/val split (default: 42)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="fraction of sequences for train (default: 0.8)")
    parser.add_argument("--test-hashes", type=Path, default=None,
                        help="path to pre-computed test image hashes JSON")
    parser.add_argument("--test-images-dir", type=Path, default=None,
                        help="path to directory of test visible images")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory for index files")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate without writing")
    args = parser.parse_args()

    stats = build_indexes(
        raw_root=args.raw_root,
        test_hashes_path=args.test_hashes,
        test_images_dir=args.test_images_dir,
        out_dir=args.out_dir,
        dry_run=args.dry_run,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    print(f"sequences: {stats['sequences']}")
    print(f"samples: {stats['samples']} "
          f"(train={stats['train']}, val={stats['val']})")
    print(f"excluded: hash={stats['excluded_hash']}, "
          f"invalid_bbox={stats['excluded_invalid_bbox']}")
    if not args.dry_run:
        print(f"wrote {args.out_dir}/")


if __name__ == "__main__":
    main()
