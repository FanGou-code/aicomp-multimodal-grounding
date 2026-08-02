"""Tests for the official Test template to processed-index contract."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.config import PREPARATION_PROTOCOL_VERSION
from aicomp_grounding.test_data import (
    OFFICIAL_TEMPLATE_CANONICAL_SHA256,
    build_test_preparation_contract,
    validate_official_test_template,
    validate_processed_test_index,
    validate_test_preparation_manifest,
)


def _official() -> dict:
    return {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "The pedestrian wearing a bright yellow jacket",
        },
        "000002_001": {
            "visible": "Images/visible/000002.jpg",
            "infrared": "Images/infrared/000002.jpg",
            "depth": "Images/depth/000002.jpg",
            "query": "The cyclist beside the white delivery vehicle",
        },
    }


def _processed() -> dict:
    return {
        query_id: {
            "visible": f"Test/{item['visible']}",
            "infrared": f"Test/{item['infrared']}",
            "depth": f"Processed/Test/depth_jet/{item['depth'].rsplit('/', 1)[-1]}",
            "query": item["query"],
        }
        for query_id, item in _official().items()
    }


class TestDataContractTests(unittest.TestCase):
    def test_exact_mapping_is_accepted(self):
        validate_processed_test_index(
            _processed(),
            _official(),
            expected_query_count=2,
            expected_template_sha256=None,
        )

    def test_ids_query_and_each_path_are_strict(self):
        mutations = []
        missing = _processed()
        del missing["000002_001"]
        mutations.append(missing)
        for field, replacement in (
            ("query", "A different valid query"),
            ("visible", "Test/Images/visible/other.png"),
            ("infrared", "Test/Images/infrared/other.png"),
            ("depth", "Processed/Test/depth_jet/other.png"),
        ):
            changed = copy.deepcopy(_processed())
            changed["000001_001"][field] = replacement
            mutations.append(changed)

        for changed in mutations:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    validate_processed_test_index(
                        changed,
                        _official(),
                        expected_query_count=2,
                        expected_template_sha256=None,
                    )

    def test_production_defaults_pin_count_and_hash(self):
        self.assertEqual(
            OFFICIAL_TEMPLATE_CANONICAL_SHA256,
            "8fae701890bbbf05099e11ac8b2a3ead18990a496449355c9f09825d88ccfbbf",
        )
        with self.assertRaisesRegex(ValueError, "9555"):
            validate_official_test_template(_official())

    def test_preparation_manifest_binds_raw_and_processed_depth_bytes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for item in _official().values():
                raw = root / "Test" / item["depth"]
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(item["depth"].encode("utf-8"))
            for item in _processed().values():
                processed = root / item["depth"]
                processed.parent.mkdir(parents=True, exist_ok=True)
                processed.write_bytes(item["depth"].encode("utf-8"))

            contract = build_test_preparation_contract(
                root,
                _processed(),
                _official(),
                expected_query_count=2,
                expected_template_sha256=None,
            )
            manifest = {
                "status": "complete",
                "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                "depth_scaling": "fixed",
                "min_depth_mm": 300,
                "max_depth_mm": 20000,
                "test_contract": contract,
            }
            validate_test_preparation_manifest(
                root,
                manifest,
                _processed(),
                _official(),
                expected_query_count=2,
                expected_template_sha256=None,
            )
            changed = root / next(iter(_processed().values()))["depth"]
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_test_preparation_manifest(
                    root,
                    manifest,
                    _processed(),
                    _official(),
                    expected_query_count=2,
                    expected_template_sha256=None,
                )

    def test_committed_manifest_identity_does_not_require_depth_file_io(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for item in _official().values():
                raw = root / "Test" / item["depth"]
                raw.parent.mkdir(parents=True, exist_ok=True)
                raw.write_bytes(item["depth"].encode("utf-8"))
            for item in _processed().values():
                processed = root / item["depth"]
                processed.parent.mkdir(parents=True, exist_ok=True)
                processed.write_bytes(item["depth"].encode("utf-8"))

            contract = build_test_preparation_contract(
                root,
                _processed(),
                _official(),
                expected_query_count=2,
                expected_template_sha256=None,
            )
            manifest = {
                "status": "complete",
                "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                "depth_scaling": "fixed",
                "min_depth_mm": 300,
                "max_depth_mm": 20000,
                "test_contract": contract,
            }
            for path in root.rglob("*.png"):
                path.unlink()
            for path in root.rglob("*.jpg"):
                path.unlink()

            validate_test_preparation_manifest(
                root,
                manifest,
                _processed(),
                _official(),
                expected_query_count=2,
                expected_template_sha256=None,
                verify_file_bytes=False,
            )
            with self.assertRaises(FileNotFoundError):
                validate_test_preparation_manifest(
                    root,
                    manifest,
                    _processed(),
                    _official(),
                    expected_query_count=2,
                    expected_template_sha256=None,
                    verify_file_bytes=True,
                )


if __name__ == "__main__":
    unittest.main()
