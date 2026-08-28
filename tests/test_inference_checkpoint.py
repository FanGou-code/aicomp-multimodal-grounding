"""Tests for inference checkpoint isolation and depth reference validation."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.prepare_rgbdt import validate_test_depth_references


def _write_color_image(path: Path, shape: tuple[int, int] = (4, 6)) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((*shape, 3), dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to create test image {path}")


def _write_official_template(root: Path, processed: dict) -> None:
    official = {}
    for query_id, item in processed.items():
        official[query_id] = {
            "visible": item["visible"].removeprefix("Test/"),
            "infrared": item["infrared"].removeprefix("Test/"),
            "depth": f"Images/depth/{Path(item['depth']).name}",
            "query": item["query"],
        }
    path = root / "Test" / "queries" / "queries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(official), encoding="utf-8")


def _validate_small_test(root: Path) -> list[str]:
    return validate_test_depth_references(
        root,
        expected_query_count=None,
        expected_template_sha256=None,
    )


class DepthReferenceTests(unittest.TestCase):
    def test_all_present_passes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_color_image(root / "Test" / "Images" / "visible" / "000001.png")
            _write_color_image(root / "Test" / "Images" / "infrared" / "000001.png")
            _write_color_image(root / "Processed" / "Test" / "depth_jet" / "000001.png")
            import json
            test_data = {
                "000001_001": {
                    "visible": "Test/Images/visible/000001.png",
                    "infrared": "Test/Images/infrared/000001.png",
                    "depth": "Processed/Test/depth_jet/000001.png",
                    "query": "A red car.",
                }
            }
            (root / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
            _write_official_template(root, test_data)
            errors = _validate_small_test(root)
            self.assertEqual(errors, [])

    def test_misaligned_modalities_report_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_color_image(root / "Test" / "Images" / "visible" / "000001.png")
            _write_color_image(root / "Test" / "Images" / "infrared" / "000001.png")
            _write_color_image(
                root / "Processed" / "Test" / "depth_jet" / "000001.png",
                shape=(3, 6),
            )
            import json

            test_data = {
                "000001_001": {
                    "visible": "Test/Images/visible/000001.png",
                    "infrared": "Test/Images/infrared/000001.png",
                    "depth": "Processed/Test/depth_jet/000001.png",
                    "query": "A red car.",
                }
            }
            (root / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
            _write_official_template(root, test_data)
            errors = _validate_small_test(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("shapes differ", errors[0])

    def test_missing_depth_reports_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_color_image(root / "Test" / "Images" / "visible" / "000001.png")
            _write_color_image(root / "Test" / "Images" / "infrared" / "000001.png")
            import json
            test_data = {
                "000001_001": {
                    "visible": "Test/Images/visible/000001.png",
                    "infrared": "Test/Images/infrared/000001.png",
                    "depth": "Processed/Test/depth_jet/000001.png",
                    "query": "A red car.",
                }
            }
            (root / "test.json").write_text(json.dumps(test_data), encoding="utf-8")
            _write_official_template(root, test_data)
            errors = _validate_small_test(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("missing", errors[0])

    def test_empty_or_escaping_test_index_is_rejected(self):
        import json

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "data"
            root.mkdir()
            (root / "test.json").write_text("{}", encoding="utf-8")
            self.assertIn("empty", validate_test_depth_references(root)[0])

            outside = Path(td) / "outside"
            for name in ("visible.png", "infrared.png", "depth.png"):
                _write_color_image(outside / name)
            escaping = {
                "000001_001": {
                    "visible": "../outside/visible.png",
                    "infrared": "../outside/infrared.png",
                    "depth": "../outside/depth.png",
                    "query": "The person wearing a yellow jacket",
                }
            }
            (root / "test.json").write_text(json.dumps(escaping), encoding="utf-8")
            official = {
                "000001_001": {
                    "visible": "Images/visible/frame.png",
                    "infrared": "Images/infrared/frame.png",
                    "depth": "Images/depth/frame.png",
                    "query": "The person wearing a yellow jacket",
                }
            }
            official_path = root / "Test" / "queries" / "queries.json"
            official_path.parent.mkdir(parents=True)
            official_path.write_text(json.dumps(official), encoding="utf-8")
            errors = _validate_small_test(root)
            self.assertEqual(len(errors), 1)
            self.assertIn("does not map", errors[0])

    def test_missing_test_json_reports_error(self):
        with tempfile.TemporaryDirectory() as td:
            errors = validate_test_depth_references(Path(td))
            self.assertEqual(len(errors), 1)
            self.assertIn("test.json not found", errors[0])

    def test_offline_infer_resume_defaults_to_true(self):
        from offline.infer import parse_args
        with unittest.mock.patch("sys.argv", ["infer.py"]):
            args = parse_args()
            self.assertTrue(args.resume)

        with unittest.mock.patch("sys.argv", ["infer.py", "--no-resume"]):
            args = parse_args()
            self.assertFalse(args.resume)

        with unittest.mock.patch("sys.argv", ["infer.py", "--resume"]):
            args = parse_args()
            self.assertTrue(args.resume)


if __name__ == "__main__":
    unittest.main()
