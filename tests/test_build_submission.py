"""Offline tests for exact-template submission packaging."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts.build_submission import build_submission, validate_official_test_template


class BuildSubmissionTests(unittest.TestCase):
    def _template_data(self) -> dict:
        return {
            "000002_001": {
                "visible": "Images/visible/000002.png",
                "infrared": "Images/infrared/000002.png",
                "depth": "Images/depth/000002.png",
                "query": "Red promotional sign with food imagery",
            },
            "000003_001": {
                "visible": "Images/visible/000003.jpg",
                "infrared": "Images/infrared/000003.jpg",
                "depth": "Images/depth/000003.jpg",
                "query": "The person wearing a blue jacket",
            },
        }

    def _write_template(self, directory: Path, name: str = "queries.json") -> Path:
        path = directory / name
        path.write_text(json.dumps(self._template_data()), encoding="utf-8")
        return path

    def _write_predictions(self, directory: Path, data: dict | None = None) -> Path:
        predictions = data or {
            "000002_001": [0.1, 0.2, 0.5, 0.8],
            "000003_001": [0.3, 0.4, 0.7, 0.9],
        }
        path = directory / "predictions.json"
        path.write_text(json.dumps(predictions), encoding="utf-8")
        return path

    def test_full_submission_changes_only_bbox_and_zip_matches(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template_path = self._write_template(root)
            output = root / "output"
            zip_path = build_submission(
                template_path,
                self._write_predictions(root),
                output,
                expected_query_count=None,
                expected_template_sha256=None,
            )
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["result.json"])
                result = json.loads(archive.read("result.json"))
            original = self._template_data()
            self.assertEqual(set(result), set(original))
            for query_id, item in result.items():
                self.assertEqual({key: item[key] for key in original[query_id]}, original[query_id])
                self.assertEqual(set(item), set(original[query_id]) | {"bbox"})

    def test_missing_or_invalid_predictions_fail_closed_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = self._write_template(root)
            predictions = self._write_predictions(
                root,
                {"000002_001": [0.9, 0.9, 0.1, 0.1]},
            )
            with self.assertRaises(ValueError):
                build_submission(
                    template,
                    predictions,
                    root / "output",
                    expected_query_count=None,
                    expected_template_sha256=None,
                )

    def test_explicit_diagnostic_fallback_fills_missing_and_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            predictions = self._write_predictions(
                root,
                {"000002_001": [0.9, 0.9, 0.1, 0.1]},
            )
            output = root / "output"
            zip_path = build_submission(
                self._write_template(root),
                predictions,
                output,
                allow_fallback=True,
                expected_query_count=None,
                expected_template_sha256=None,
            )
            self.assertEqual(zip_path, output / "submission.diagnostic.zip")
            self.assertFalse((output / "result.json").exists())
            self.assertFalse((output / "submission.zip").exists())
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["result.json"])
                result = json.loads(archive.read("result.json"))
            self.assertEqual(result["000002_001"]["bbox"], [0.0, 0.0, 0.001, 0.001])
            self.assertEqual(result["000003_001"]["bbox"], [0.0, 0.0, 0.001, 0.001])

    def test_invalid_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                build_submission(
                    self._write_template(root),
                    self._write_predictions(root),
                    root / "output",
                    default_bbox=[0.8, 0.8, 0.2, 0.2],
                    expected_query_count=None,
                    expected_template_sha256=None,
                )

    def test_processed_test_index_is_rejected(self):
        processed = self._template_data()
        processed["000002_001"] = dict(processed["000002_001"])
        processed["000002_001"]["visible"] = "Test/Images/visible/000002.png"
        processed["000002_001"]["depth"] = "Processed/Test/depth_jet/000002.png"
        with self.assertRaises(ValueError):
            validate_official_test_template(
                processed,
                expected_query_count=None,
                expected_template_sha256=None,
            )

    def test_template_with_bbox_or_extra_fields_is_rejected(self):
        modified = self._template_data()
        modified["000002_001"] = dict(modified["000002_001"], bbox=[0, 0, 1, 1])
        with self.assertRaises(ValueError):
            validate_official_test_template(
                modified,
                expected_query_count=None,
                expected_template_sha256=None,
            )

    def test_function_api_refuses_to_overwrite_template(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = self._write_template(root, name="result.json")
            with self.assertRaises(ValueError):
                build_submission(
                    template,
                    self._write_predictions(root),
                    root,
                    expected_query_count=None,
                    expected_template_sha256=None,
                )

    def test_zip_failure_preserves_previous_valid_zip(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "output"
            output.mkdir()
            old_zip = output / "submission.zip"
            old_zip.write_bytes(b"previous-valid-package")
            with patch("scripts.build_submission.zipfile.ZipFile", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    build_submission(
                        self._write_template(root),
                        self._write_predictions(root),
                        output,
                        expected_query_count=None,
                        expected_template_sha256=None,
                    )
            self.assertEqual(old_zip.read_bytes(), b"previous-valid-package")

    def test_template_count_and_path_separators_are_strict(self):
        with self.assertRaisesRegex(ValueError, "9555"):
            validate_official_test_template(self._template_data())
        modified = self._template_data()
        modified["000002_001"] = dict(modified["000002_001"])
        modified["000002_001"]["visible"] = "Images\\visible\\000002.png"
        with self.assertRaises(ValueError):
            validate_official_test_template(
                modified,
                expected_query_count=None,
                expected_template_sha256=None,
            )

    def test_template_content_fingerprint_is_strict(self):
        modified = self._template_data()
        modified["000002_001"] = dict(modified["000002_001"])
        modified["000002_001"]["query"] = "A different but structurally valid query"
        with self.assertRaisesRegex(ValueError, "canonical SHA-256"):
            validate_official_test_template(modified, expected_query_count=None)


if __name__ == "__main__":
    unittest.main()
