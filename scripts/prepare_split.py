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


# Default split (seed=42, train_ratio=0.8) — frozen for reproducibility.
# Override with --seed and --train-ratio.
_DEFAULT_TRAIN = [
    "001", "002", "003", "005", "006", "007", "008", "009", "010", "011",
    "012", "015", "018", "019", "020", "021", "022", "025", "026", "027",
    "028", "029", "030", "031", "032", "033", "034", "035", "037", "038",
    "039", "040", "042", "043", "044", "046", "047", "049", "051", "054",
    "055", "056", "057", "059", "060", "061", "062", "063", "065", "066",
    "067", "068", "069", "070", "071", "073", "074", "075", "076", "077",
    "078", "079", "081", "083", "085", "086", "087", "088", "089", "090",
    "091", "092", "093", "094", "095", "096", "097", "098", "100", "101",
    "103", "104", "105", "106", "107", "109", "110", "114", "116", "118",
    "119", "121", "122", "123", "124", "125", "127", "128", "129", "130",
    "131", "132", "133", "134", "135", "137", "138", "139", "140", "142",
    "144", "145", "146", "147", "148", "150", "152", "153", "154", "155",
    "156", "157", "158", "159", "160", "161", "162", "163", "164", "165",
    "166", "167", "168", "169", "170", "171", "172", "174", "176", "178",
    "179", "180", "181", "183", "185", "188", "189", "191", "192", "193",
    "196", "197", "198", "199", "200", "201", "202", "203", "204", "205",
    "206", "207", "208", "209", "210", "211", "212", "213", "214", "216",
    "218", "219", "220", "221", "222", "223", "224", "225", "226", "227",
    "228", "229", "231", "232", "234", "235", "237", "238", "239", "240",
    "241", "242", "243", "244", "245", "246", "247", "248", "249", "250",
    "251", "252", "253", "254", "255", "256", "257", "258", "260", "261",
    "262", "263", "264", "265", "266", "267", "268", "269", "270", "271",
    "272", "273", "274", "276", "277", "278", "279", "281", "282", "284",
    "285", "286", "287", "289", "290", "291", "292", "293", "294", "295",
    "297", "298", "299", "300", "301", "304", "305", "306", "307", "308",
    "311", "312", "313", "314", "315", "316", "318", "319", "320", "321",
    "323", "324", "325", "326", "327", "329", "330", "331", "332", "334",
    "335", "336", "337", "338", "339", "340", "341", "342", "344", "345",
    "346", "348", "349", "350", "351", "352", "353", "355", "356", "357",
    "359", "361", "362", "363", "364", "365", "368", "369", "370", "371",
    "373", "374", "375", "376", "377", "379", "381", "383", "384", "386",
    "387", "390", "392", "393", "394", "395", "396", "398", "399", "400",
]

_DEFAULT_VAL = [
    "004", "013", "014", "016", "017", "023", "024", "036", "041", "045",
    "048", "050", "052", "053", "058", "064", "072", "080", "082", "084",
    "099", "102", "108", "111", "112", "113", "115", "117", "120", "126",
    "136", "141", "143", "149", "151", "173", "175", "177", "182", "184",
    "186", "187", "190", "194", "195", "215", "217", "230", "233", "236",
    "259", "275", "280", "283", "288", "296", "302", "303", "309", "310",
    "317", "322", "328", "333", "343", "347", "354", "358", "360", "366",
    "367", "372", "378", "380", "382", "385", "388", "389", "391", "397",
]


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



def build_indexes(
    raw_root: Path,
    *,
    out_dir: Path,
    dry_run: bool = False,
    test_hashes_path: Path | None = None,
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

    # --- Stage 1: hash dedup against test images (pre-computed hashes) ---
    test_hashes: set[str] = set()
    if test_hashes_path and test_hashes_path.is_file():
        test_hashes = set(json.loads(test_hashes_path.read_text(encoding="utf-8")).values())

    excluded: list[dict] = []
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

            # Hash dedup
            if test_hashes and visible_path.is_file():
                if _hash_file(visible_path) in test_hashes:
                    excluded.append({
                        "sample_id": sample_id,
                        "visible": f"Train/{seq}/color/{filename}",
                        "reason": "hash-collision-with-test",
                    })
                    stats["excluded_hash"] += 1
                    continue

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
            "preparation_protocol_version": 2,
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
        }
        manifest_path = out_dir / "split_manifest.json"
        tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8",
        )
        tmp.replace(manifest_path)

        if excluded:
            excl_path = out_dir / "excluded.json"
            tmp = excl_path.with_name(excl_path.name + ".tmp")
            tmp.write_text(
                json.dumps({"description": "Samples excluded via hash dedup against test images",
                            "excluded": excluded}, ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(excl_path)

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
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="output directory for index files")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate without writing")
    args = parser.parse_args()

    stats = build_indexes(
        raw_root=args.raw_root,
        test_hashes_path=args.test_hashes,
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
