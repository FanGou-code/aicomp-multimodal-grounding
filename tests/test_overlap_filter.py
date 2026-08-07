"""Tests for overlap filtering script."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


def _make_rgb_image(path: Path, width: int = 10, height: int = 10, seed: int = 0) -> Path:
    from PIL import Image
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.RandomState(seed).rand(height, width, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    with path.open("wb") as f:
        img.save(f, format="PNG")
    return path


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fsrc:
        with dst.open("wb") as fdst:
            while True:
                chunk = fsrc.read(1024 * 1024)
                if not chunk:
                    break
                fdst.write(chunk)


class FilterOverlapTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_indexes(self, train_items, val_items):
        train_path = self.root / "train.json"
        val_path = self.root / "val.json"
        train_path.write_text(json.dumps(train_items))
        val_path.write_text(json.dumps(val_items))
        return train_path, val_path

    def _write_test_images(self, *image_pairs):
        """Each pair = (filename, train_image_path_to_copy)."""
        test_dir = self.root / "Test" / "Images" / "visible"
        for fname, src in image_pairs:
            _copy_file(src, test_dir / fname)

    def test_filters_overlapping_train_sample(self):
        img = _make_rgb_image(self.root / "Train" / "001" / "color" / "00000001.png", seed=1)
        img2 = _make_rgb_image(self.root / "Train" / "001" / "color" / "00000002.png", seed=2)

        self._write_indexes(
            {
                "001_00000001": {
                    "visible": "Train/001/color/00000001.png",
                    "infrared": "Train/001/infrared/00000001.png",
                    "depth": "Processed/Train/001/depth_jet/00000001.png",
                    "query": "",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "width": 1920,
                    "height": 1080,
                },
                "001_00000002": {
                    "visible": "Train/001/color/00000002.png",
                    "infrared": "Train/001/infrared/00000002.png",
                    "depth": "Processed/Train/001/depth_jet/00000002.png",
                    "query": "",
                    "bbox": [0.5, 0.6, 0.7, 0.8],
                    "width": 1920,
                    "height": 1080,
                },
            },
            {},
        )
        self._write_test_images(("000001.png", img))

        from scripts.filter_overlap import _build_test_image_hashes, _filter_samples

        test_hashes = _build_test_image_hashes(self.root / "Test" / "Images" / "visible")
        self.assertEqual(len(test_hashes), 1)

        with (self.root / "train.json").open() as f:
            index = json.load(f)
        cleaned, excluded = _filter_samples(index, test_hashes, self.root, "train")
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["sample_id"], "001_00000001")
        self.assertEqual(len(cleaned), 1)
        self.assertNotIn("001_00000001", cleaned)
        self.assertIn("001_00000002", cleaned)

    def test_dry_run_does_not_modify_files(self):
        img = _make_rgb_image(self.root / "Train" / "002" / "color" / "00000001.png", seed=3)
        self._write_indexes(
            {
                "002_00000001": {
                    "visible": "Train/002/color/00000001.png",
                    "infrared": "Train/002/infrared/00000001.png",
                    "depth": "Processed/Train/002/depth_jet/00000001.png",
                    "query": "",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "width": 1920,
                    "height": 1080,
                },
            },
            {},
        )
        self._write_test_images(("000002.png", img))
        # Explicitly read before calling — if script does dry-run correctly, file is unchanged
        original = json.loads((self.root / "train.json").read_text())
        from scripts.filter_overlap import _build_test_image_hashes, _filter_samples

        test_hashes = _build_test_image_hashes(self.root / "Test" / "Images" / "visible")
        with (self.root / "train.json").open() as f:
            index = json.load(f)
        cleaned, excluded = _filter_samples(index, test_hashes, self.root, "train")
        self.assertEqual(len(excluded), 1)
        self.assertNotIn("002_00000001", cleaned)
        # In dry run, the file on disk should be unchanged
        self.assertEqual(json.loads((self.root / "train.json").read_text()), original)

    def test_no_overlap_keeps_all(self):
        img = _make_rgb_image(self.root / "Train" / "003" / "color" / "00000001.png", seed=4)
        img_test = _make_rgb_image(self.root / "Train" / "003" / "color" / "00000002.png", seed=5)
        self._write_indexes(
            {
                "003_00000001": {
                    "visible": "Train/003/color/00000001.png",
                    "infrared": "Train/003/infrared/00000001.png",
                    "depth": "Processed/Train/003/depth_jet/00000001.png",
                    "query": "",
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "width": 1920,
                    "height": 1080,
                },
            },
            {},
        )
        self._write_test_images(("000003.png", img_test))
        from scripts.filter_overlap import _build_test_image_hashes, _filter_samples

        test_hashes = _build_test_image_hashes(self.root / "Test" / "Images" / "visible")
        with (self.root / "train.json").open() as f:
            index = json.load(f)
        cleaned, excluded = _filter_samples(index, test_hashes, self.root, "train")
        self.assertEqual(len(excluded), 0)
        self.assertEqual(len(cleaned), 1)


if __name__ == "__main__":
    unittest.main()
