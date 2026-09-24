"""Tests for the official Test template to processed-index contract."""

from __future__ import annotations

import copy
import unittest

from aicomp_grounding.testset import (
    OFFICIAL_TEMPLATE_CANONICAL_SHA256,
    validate_official_test_template,
    validate_processed_test_index,
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


if __name__ == "__main__":
    unittest.main()
