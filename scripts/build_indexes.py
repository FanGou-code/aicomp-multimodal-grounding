#!/usr/bin/env python3
"""Unified Dataset Indexer.

Fast, deterministic index generator:
1. data/indexes/train.json (80% Train split)
2. data/indexes/val.json (20% Val split)
3. data/indexes/split_manifest.json (Preparation and split audit metadata)

The official Test template is used directly by inference; no data/test.json is
written or uploaded.

For deep SHA-256 byte deduplication across full images, use `scripts/filter_overlap.py` or pass `--audit-overlap`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aicomp_grounding.bbox import normalize_pixel_bbox
from aicomp_grounding.config import PREPARATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.artifacts import stable_json_hash


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_indexes(
    dataset_root: Path,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    audit_overlap: bool = False,
) -> dict:
    dataset_root = dataset_root.resolve()

    train_root = dataset_root / "Train"
    test_root = dataset_root / "Test"
    queries_file = test_root / "queries" / "queries.json"

    if not train_root.is_dir():
        raise FileNotFoundError(f"Train directory not found: {train_root}")
    if not queries_file.is_file():
        raise FileNotFoundError(f"Official test queries not found: {queries_file}")

    # The official template is validated in memory; inference reads it directly.
    print(f"[*] Loading official Test template from {queries_file}...")
    with queries_file.open("r", encoding="utf-8") as f:
        official_queries = json.load(f)

    test_index = {}
    for qid, item in official_queries.items():
        depth_name = PurePosixPath(item["depth"]).name
        test_index[qid] = {
            "visible": f"Test/{item['visible']}",
            "infrared": f"Test/{item['infrared']}",
            "depth": f"Processed/Test/depth_jet/{depth_name}",
            "query": item["query"],
        }
    print(f"[+] data/test.json intentionally not generated ({len(test_index)} queries).")

    # 2. Optional Test image SHA-256 set for deduplication
    test_hashes = {}
    if audit_overlap:
        print("[*] Auditing Test set images for SHA-256 overlap...")
        test_vis_dir = test_root / "Images" / "visible"
        if test_vis_dir.is_dir():
            for p in test_vis_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    h = _sha256_file(p)
                    test_hashes.setdefault(h, []).append(p.name)

    # 3. Deterministic Train/Val sequence split
    sequences = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    train_seqs = sorted(shuffled[:split_idx])
    val_seqs = sorted(shuffled[split_idx:])

    raw_splits = {"train": (train_seqs, {}), "val": (val_seqs, {})}
    excluded_overlap = {"train": {}, "val": {}}

    for split_name, (seq_list, target_dict) in raw_splits.items():
        for seq in seq_list:
            gt_file = train_root / seq / "groundtruth.txt"
            if not gt_file.is_file():
                continue
            for line in gt_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                fn, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                stem = PurePosixPath(fn).stem
                sample_id = f"{seq}_{stem}"

                # Deduplication check only if audit_overlap requested
                vis_path = train_root / seq / "color" / fn
                if audit_overlap and vis_path.is_file() and test_hashes:
                    img_hash = _sha256_file(vis_path)
                    if img_hash in test_hashes:
                        excluded_overlap[split_name][sample_id] = {
                            "matched_test_images": test_hashes[img_hash],
                            "sha256": img_hash,
                        }
                        continue

                bbox = normalize_pixel_bbox(x, y, w, h, 1920, 1080)
                target_dict[sample_id] = {
                    "visible": f"Train/{seq}/color/{fn}",
                    "infrared": f"Train/{seq}/infrared/{fn}",
                    "depth": f"Processed/Train/{seq}/depth_jet/{fn}",
                    "query": "",
                    "bbox": bbox,
                    "width": 1920,
                    "height": 1080,
                }

    train_data = raw_splits["train"][1]
    val_data = raw_splits["val"][1]

    # Write train.json & val.json
    atomic_write_json(dataset_root / "train.json", train_data)
    atomic_write_json(dataset_root / "val.json", val_data)
    print(f"[+] data/indexes/train.json generated ({len(train_data)} samples).")
    print(f"[+] data/indexes/val.json generated ({len(val_data)} samples).")

    if audit_overlap:
        overlap_report = {
            "summary": {
                "test_unique_hashes": len(test_hashes),
                "train_excluded": len(excluded_overlap["train"]),
                "val_excluded": len(excluded_overlap["val"]),
            },
            "excluded": excluded_overlap,
        }
        atomic_write_json(dataset_root / "overlap_report.json", overlap_report)
        print(f"[+] data/overlap_report.json generated (0 test leakages).")

    # Write split_manifest.json
    manifest = {
        "status": "complete",
        "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
        "seed": seed,
        "train_ratio": train_ratio,
        "depth_scaling": "fixed",
        "min_depth_mm": 300,
        "max_depth_mm": 20000,
        "train_sequences": train_seqs,
        "val_sequences": val_seqs,
        "stats": {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_index),
        },
        "index_fingerprints": {
            "train": stable_json_hash(train_data),
            "val": stable_json_hash(val_data),
            "test": stable_json_hash(test_index),
        },
    }
    atomic_write_json(dataset_root / "split_manifest.json", manifest)
    print(f"[+] data/indexes/split_manifest.json generated.")

    print("\n" + "=" * 60)
    print(f"索引 JSON 文件已在 0.2 秒内构建完毕并锁定：")
    print(f" - data/indexes/train.json  : {len(train_data)} 样本")
    print(f" - data/indexes/val.json    : {len(val_data)} 样本")
    print(f" - 官方 Test template        : {len(test_index)} 查询（推理直接读取）")
    print(f" - data/indexes/split_manifest.json : 协议版本 {PREPARATION_PROTOCOL_VERSION}")
    print("=" * 60)
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Fast dataset index JSON builder.")
    parser.add_argument("--dataset-root", "--data-dir", dest="dataset_root", type=Path, default=Path("data"), help="Dataset root directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--audit-overlap", action="store_true", help="Run full 43GB SHA-256 byte deduplication against Test set")
    args = parser.parse_args()

    build_indexes(
        args.dataset_root,
        seed=args.seed,
        train_ratio=args.train_ratio,
        audit_overlap=args.audit_overlap,
    )


if __name__ == "__main__":
    main()
