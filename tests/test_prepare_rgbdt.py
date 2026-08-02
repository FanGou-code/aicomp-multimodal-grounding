import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_rgbdt import (
    build_split,
    parse_groundtruth,
    prepare_dataset,
    prepare_test_depth,
    process_depth_to_jet,
    validate_raw_depth,
    validate_test_depth_references,
)


def _write_color(path: Path, shape: tuple[int, int] = (4, 6)) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to create test image {path}")


def _write_depth(
    path: Path,
    *,
    shape: tuple[int, int] = (4, 6),
    dtype="uint16",
) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_dtype = np.dtype(dtype)
    fill_value = min(1000, int(np.iinfo(numpy_dtype).max))
    image = np.full(shape, fill_value, dtype=numpy_dtype)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to create test depth {path}")


def _args(root: Path, *, dry_run: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        dataset_root=root,
        index_dir=root / "indexes",
        train_ratio=0.8,
        seed=42,
        min_depth_mm=300,
        max_depth_mm=20000,
        depth_scaling="fixed",
        overwrite_depth=False,
        overwrite_indexes=False,
        dry_run=dry_run,
        skip_test_validation=True,
    )


def _write_sequence(
    root: Path,
    *,
    gt: str = "00000001.png,1,1,3,2\n",
    color_shape: tuple[int, int] = (4, 6),
    infrared_shape: tuple[int, int] = (4, 6),
    depth_shape: tuple[int, int] = (4, 6),
    depth_dtype: str = "uint16",
    scene: str = "001",
) -> Path:
    sequence = root / "Train" / scene
    sequence.mkdir(parents=True, exist_ok=True)
    (sequence / "groundtruth.txt").write_text(gt, encoding="utf-8")
    filename = "00000001.png"
    _write_color(sequence / "color" / filename, color_shape)
    _write_color(sequence / "infrared" / filename, infrared_shape)
    _write_depth(sequence / "depth" / filename, shape=depth_shape, dtype=depth_dtype)
    return sequence


def _write_official_template(root: Path, processed: dict) -> None:
    import json

    official = {
        query_id: {
            "visible": item["visible"].removeprefix("Test/"),
            "infrared": item["infrared"].removeprefix("Test/"),
            "depth": f"Images/depth/{Path(item['depth']).name}",
            "query": item["query"],
        }
        for query_id, item in processed.items()
    }
    path = root / "Test" / "queries" / "queries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(official), encoding="utf-8")


class PrepareDatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_disjoint(self):
        sequences = [f"{index:03d}" for index in range(1, 11)]
        train_a, val_a = build_split(sequences, 0.8, 42)
        train_b, val_b = build_split(sequences, 0.8, 42)
        self.assertEqual((train_a, val_a), (train_b, val_b))
        self.assertEqual(len(train_a), 8)
        self.assertFalse(set(train_a) & set(val_a))
        self.assertEqual(set(train_a) | set(val_a), set(sequences))

    def test_missing_raw_directory_fails_before_processing(self):
        with self.assertRaises(FileNotFoundError):
            prepare_dataset(_args(Path("does-not-exist")))

    def test_groundtruth_parser_rejects_malformed_duplicate_and_nonfinite_rows(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "groundtruth.txt"
            cases = (
                "0001.png,1,2,3\n",
                "0001.png,1,2,nope,4\n",
                "0001.png,1,2,nan,4\n",
                "0001.png,1,2,3,4\n0001.png,5,6,7,8\n",
            )
            for content in cases:
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_groundtruth(path)
            for content in ("", "../0001.png,1,2,3,4\n", "nested/0001.png,1,2,3,4\n"):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        parse_groundtruth(path)

    def test_raw_depth_rejects_corrupt_and_non_uint16_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            corrupt = root / "corrupt.png"
            corrupt.write_bytes(b"not an image")
            with self.assertRaises(ValueError):
                validate_raw_depth(corrupt)

            uint8_depth = root / "uint8.png"
            _write_depth(uint8_depth, dtype="uint8")
            with self.assertRaises(ValueError):
                validate_raw_depth(uint8_depth)

    def test_dry_run_decodes_modalities_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root)
            _write_sequence(root, scene="002")
            manifest = prepare_dataset(_args(root))
            self.assertEqual(manifest["stats"]["valid_samples"], 2)
            self.assertEqual(manifest["stats"]["depth_would_write"], 2)
            self.assertEqual(manifest["stats"]["depth_written"], 0)
            self.assertFalse((root / "indexes").exists())
            self.assertFalse((root / "Processed").exists())

    def test_dry_run_rejects_corrupt_depth_and_misaligned_modalities(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sequence = _write_sequence(root)
            _write_sequence(root, scene="002")
            (sequence / "depth" / "00000001.png").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                prepare_dataset(_args(root))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root, infrared_shape=(3, 6))
            _write_sequence(root, scene="002")
            with self.assertRaisesRegex(ValueError, "Misaligned modalities"):
                prepare_dataset(_args(root))

    def test_invalid_official_bbox_is_recorded_in_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root, gt="00000001.png,1,1,0,2\n")
            _write_sequence(root, scene="002")
            manifest = prepare_dataset(_args(root))
            self.assertEqual(manifest["stats"]["valid_samples"], 1)
            self.assertEqual(manifest["stats"]["excluded_invalid_bbox"], 1)
            self.assertEqual(len(manifest["exclusions"]), 1)
            self.assertEqual(manifest["exclusions"][0]["reason"], "invalid_ground_truth_bbox")

    def test_core_functions_reject_invalid_configuration_and_empty_splits(self):
        with self.assertRaises(ValueError):
            build_split(["001", "002"], 1.5, 42)
        with self.assertRaises(ValueError):
            build_split(["001"], 0.8, 42)
        with self.assertRaises(ValueError):
            build_split(["001", "001"], 0.8, 42)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "depth.png"
            _write_depth(source)
            with self.assertRaisesRegex(ValueError, "Unsupported depth scaling"):
                process_depth_to_jet(
                    source,
                    root / "output.png",
                    min_depth_mm=300,
                    max_depth_mm=20000,
                    scaling="typo",
                )

    def test_existing_depth_is_reused_only_when_pixels_match_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root)
            _write_sequence(root, scene="002")
            initial = _args(root, dry_run=False)
            completed = prepare_dataset(initial)
            self.assertEqual(completed["status"], "complete")
            self.assertEqual(completed["index_sample_counts"], {"train": 1, "val": 1})

            same = _args(root)
            manifest = prepare_dataset(same)
            self.assertEqual(manifest["stats"]["depth_reused"], 2)

            overwrite_matching = _args(root, dry_run=False)
            overwrite_matching.overwrite_depth = True
            overwrite_matching.overwrite_indexes = True
            with patch("scripts.prepare_rgbdt.process_depth_to_jet") as rewrite:
                manifest = prepare_dataset(overwrite_matching)
            rewrite.assert_not_called()
            self.assertEqual(manifest["stats"]["depth_reused"], 2)
            self.assertEqual(manifest["stats"]["depth_written"], 0)

            changed = _args(root)
            changed.depth_scaling = "per-frame"
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_dataset(changed)

    def test_dry_run_validates_planned_test_png_depth(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root)
            _write_sequence(root, scene="002")
            _write_color(root / "Test" / "Images" / "visible" / "frame.png")
            _write_color(root / "Test" / "Images" / "infrared" / "frame.png")
            _write_depth(root / "Test" / "Images" / "depth" / "frame.png")
            test_data = {
                "000001_001": {
                    "visible": "Test/Images/visible/frame.png",
                    "infrared": "Test/Images/infrared/frame.png",
                    "depth": "Processed/Test/depth_jet/frame.png",
                    "query": "The person wearing a bright yellow jacket",
                }
            }
            (root / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
            _write_official_template(root, test_data)
            args = _args(root)
            args.skip_test_validation = False
            manifest = prepare_dataset(
                args,
                expected_test_query_count=None,
                expected_test_template_sha256=None,
            )
            self.assertEqual(manifest["stats"]["test_depth_would_write"], 1)

    def test_formal_small_test_contract_uses_overridden_template_constraints(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_sequence(root)
            _write_sequence(root, scene="002")
            _write_color(root / "Test" / "Images" / "visible" / "frame.png")
            _write_color(root / "Test" / "Images" / "infrared" / "frame.png")
            _write_depth(root / "Test" / "Images" / "depth" / "frame.png")
            test_data = {
                "000001_001": {
                    "visible": "Test/Images/visible/frame.png",
                    "infrared": "Test/Images/infrared/frame.png",
                    "depth": "Processed/Test/depth_jet/frame.png",
                    "query": "The person wearing a bright yellow jacket",
                }
            }
            (root / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
            _write_official_template(root, test_data)
            args = _args(root, dry_run=False)
            args.skip_test_validation = False

            manifest = prepare_dataset(
                args,
                expected_test_query_count=None,
                expected_test_template_sha256=None,
            )

            self.assertEqual(manifest["test_contract"]["official_query_count"], 1)
            self.assertEqual(manifest["test_contract"]["raw_depth_file_count"], 1)
            self.assertEqual(manifest["test_contract"]["processed_depth_file_count"], 1)

    def test_precolored_test_jpg_is_copied_and_black_stale_output_is_rejected(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "Test" / "Images" / "depth" / "frame.jpg"
            raw.parent.mkdir(parents=True)
            self.assertTrue(cv2.imwrite(str(raw), np.full((4, 6, 3), 140, dtype=np.uint8)))
            output = root / "Processed" / "Test" / "depth_jet" / "frame.jpg"
            _write_color(output)

            with self.assertRaisesRegex(ValueError, "differs from its raw source"):
                prepare_test_depth(_args(root))

            dry_overwrite = _args(root)
            dry_overwrite.overwrite_depth = True
            stale_bytes = output.read_bytes()
            stats = prepare_test_depth(dry_overwrite)
            self.assertEqual(stats["total_precolored"], 1)
            self.assertEqual(stats["would_write"], 1)
            self.assertEqual(output.read_bytes(), stale_bytes)

            overwrite = _args(root, dry_run=False)
            overwrite.overwrite_depth = True
            prepare_test_depth(overwrite)
            self.assertEqual(output.read_bytes(), raw.read_bytes())

            with patch("scripts.prepare_rgbdt.copy_precolored_depth") as rewrite:
                matching = prepare_test_depth(overwrite)
            rewrite.assert_not_called()
            self.assertEqual(matching["reused"], 1)
            self.assertEqual(matching["written"], 0)

    def test_test_validation_requires_uint8_three_channel_depth(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_color(root / "Test" / "Images" / "visible" / "frame.png")
            _write_color(root / "Test" / "Images" / "infrared" / "frame.png")
            _write_depth(root / "Processed" / "Test" / "depth_jet" / "frame.png")
            atomic = {
                "000001_001": {
                    "visible": "Test/Images/visible/frame.png",
                    "infrared": "Test/Images/infrared/frame.png",
                    "depth": "Processed/Test/depth_jet/frame.png",
                    "query": "The person wearing a bright yellow jacket",
                }
            }
            (root / "test.json").write_text(json.dumps(atomic), encoding="utf-8")
            _write_official_template(root, atomic)
            errors = validate_test_depth_references(
                root,
                expected_query_count=None,
                expected_template_sha256=None,
            )
            self.assertEqual(len(errors), 1)
            self.assertIn("depth must be uint8 HxWx3", errors[0])


if __name__ == "__main__":
    unittest.main()
