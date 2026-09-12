import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_split import (
    _collect_test_hashes,
    _hash_file,
    _normalize_bbox,
    _parse_groundtruth,
    _png_size,
    build_indexes,
)


def _create_minimal_png(width: int, height: int, payload: bytes = b"\x00") -> bytes:
    """Create a minimal valid PNG image header."""
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = b"\x00\x00\x00\x00"
    ihdr_chunk = struct.pack(">I", len(ihdr_data)) + b"IHDR" + ihdr_data + ihdr_crc
    idat_chunk = struct.pack(">I", len(payload)) + b"IDAT" + payload + b"\x00\x00\x00\x00"
    iend_chunk = struct.pack(">I", 0) + b"IEND" + b"\x00\x00\x00\x00"
    return sig + ihdr_chunk + idat_chunk + iend_chunk


class PrepareSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_normalize_bbox(self):
        bbox = _normalize_bbox(10, 20, 30, 40, 100, 200)
        self.assertEqual(bbox, [0.1, 0.1, 0.4, 0.3])

        self.assertIsNone(_normalize_bbox(-1, 0, 10, 10, 100, 100))
        self.assertIsNone(_normalize_bbox(0, -5, 10, 10, 100, 100))
        self.assertIsNone(_normalize_bbox(10, 10, 0, 10, 100, 100))
        self.assertIsNone(_normalize_bbox(10, 10, 10, -2, 100, 100))
        self.assertIsNone(_normalize_bbox(50, 50, 60, 10, 100, 100))
        self.assertIsNone(_normalize_bbox(50, 50, 10, 60, 100, 100))

    def test_png_size(self):
        png_path = self.root / "sample.png"
        png_path.write_bytes(_create_minimal_png(640, 480))
        w, h = _png_size(png_path)
        self.assertEqual((w, h), (640, 480))

        bad_path = self.root / "bad.png"
        bad_path.write_bytes(b"not a png image")
        with self.assertRaises(ValueError):
            _png_size(bad_path)

    def test_parse_groundtruth(self):
        gt_path = self.root / "groundtruth.txt"
        gt_path.write_text(
            "# Comment line\n"
            "00000001.png, 10, 20, 30, 40\n"
            "\n"
            "00000002.png, 5.5, 6.5, 15.0, 25.0\n"
            "invalid_line\n",
            encoding="utf-8",
        )
        parsed = _parse_groundtruth(gt_path)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed["00000001.png"], (10, 20, 30, 40))
        self.assertEqual(parsed["00000002.png"], (5, 6, 15, 25))

    def test_collect_test_hashes(self):
        test_dir = self.root / "Test" / "Images" / "visible"
        test_dir.mkdir(parents=True, exist_ok=True)
        img1 = test_dir / "000001.png"
        img1.write_bytes(_create_minimal_png(100, 100, b"data1"))
        h1 = _hash_file(img1)

        # Auto-detect from raw_root
        hashes = _collect_test_hashes(raw_root=self.root)
        self.assertIn(h1, hashes)
        self.assertEqual(hashes[h1], ["000001"])

        # From explicit JSON dict {stem: hash}
        json_path = self.root / "test_hashes.json"
        json_path.write_text(json.dumps({"000001": h1}), encoding="utf-8")
        hashes_json = _collect_test_hashes(test_hashes_path=json_path)
        self.assertIn(h1, hashes_json)
        self.assertEqual(hashes_json[h1], ["000001"])

    def test_build_indexes_end_to_end(self):
        raw_train = self.root / "Train"
        test_dir = self.root / "Test" / "Images" / "visible"
        test_dir.mkdir(parents=True, exist_ok=True)

        test_img = test_dir / "000099.png"
        png_overlap = _create_minimal_png(100, 100, b"overlap_payload")
        test_img.write_bytes(png_overlap)

        png_normal = _create_minimal_png(100, 100, b"normal_payload")

        # Setup sequence 001
        seq001 = raw_train / "001"
        (seq001 / "color").mkdir(parents=True)
        (seq001 / "infrared").mkdir(parents=True)
        (seq001 / "depth").mkdir(parents=True)

        (seq001 / "color" / "00000001.png").write_bytes(png_normal)
        (seq001 / "infrared" / "00000001.png").write_bytes(png_normal)
        (seq001 / "depth" / "00000001.png").write_bytes(png_normal)

        (seq001 / "color" / "00000002.png").write_bytes(png_overlap)
        (seq001 / "infrared" / "00000002.png").write_bytes(png_overlap)
        (seq001 / "depth" / "00000002.png").write_bytes(png_overlap)

        (seq001 / "color" / "00000003.png").write_bytes(png_normal)
        (seq001 / "infrared" / "00000003.png").write_bytes(png_normal)
        (seq001 / "depth" / "00000003.png").write_bytes(png_normal)

        (seq001 / "groundtruth.txt").write_text(
            "00000001.png, 10, 10, 20, 20\n"
            "00000002.png, 10, 10, 20, 20\n"
            "00000003.png, -5, 10, 20, 20\n",
            encoding="utf-8",
        )

        # Setup sequence 002
        seq002 = raw_train / "002"
        (seq002 / "color").mkdir(parents=True)
        (seq002 / "infrared").mkdir(parents=True)
        (seq002 / "depth").mkdir(parents=True)

        (seq002 / "color" / "00000001.png").write_bytes(png_normal)
        (seq002 / "infrared" / "00000001.png").write_bytes(png_normal)
        (seq002 / "depth" / "00000001.png").write_bytes(png_normal)

        (seq002 / "groundtruth.txt").write_text(
            "00000001.png, 15, 15, 25, 25\n",
            encoding="utf-8",
        )

        out_dir = self.root / "separate-indexes"
        stats = build_indexes(
            raw_root=self.root,
            out_dir=out_dir,
            dry_run=False,
            seed=42,
            train_ratio=0.5,
        )

        self.assertEqual(stats["sequences"], 2)
        self.assertEqual(stats["samples"], 2)
        self.assertEqual(stats["excluded_hash"], 1)
        self.assertEqual(stats["excluded_invalid_bbox"], 1)
        self.assertEqual(stats["train"] + stats["val"], 2)

        train_idx = json.loads((out_dir / "train.json").read_text(encoding="utf-8"))
        val_idx = json.loads((out_dir / "val.json").read_text(encoding="utf-8"))
        all_ids = set(train_idx.keys()) | set(val_idx.keys())
        self.assertIn("001_00000001", all_ids)
        self.assertIn("002_00000001", all_ids)
        self.assertNotIn("001_00000002", all_ids)
        self.assertNotIn("001_00000003", all_ids)

        manifest = json.loads((out_dir / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["preparation_protocol_version"], 2)
        self.assertEqual(manifest["index_sample_counts"]["train"] + manifest["index_sample_counts"]["val"], 2)

        overlap = json.loads((out_dir / "excluded_overlap.json").read_text(encoding="utf-8"))
        self.assertEqual(overlap["train_excluded"] + overlap["val_excluded"], 1)
        records = overlap["records"]["train"] + overlap["records"]["val"]
        self.assertEqual(records[0]["sample_id"], "001_00000002")
        self.assertEqual(records[0]["test_images"], ["000099"])

        self.assertTrue((out_dir / "excluded.json").is_file())

        from foundry.pipeline.source import load_annotation_source
        self.assertEqual(load_annotation_source(self.root, "train", index_dir=out_dir), train_idx)
        self.assertEqual(load_annotation_source(self.root, "val", index_dir=out_dir), val_idx)
        # A changed source must still be rejected: accepting historical hash
        # serializers is not permission to skip content verification.
        key = next(iter(train_idx))
        train_idx[key]["width"] += 1
        (out_dir / "train.json").write_text(json.dumps(train_idx))
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_annotation_source(self.root, "train", index_dir=out_dir)


if __name__ == "__main__":
    unittest.main()
