"""Tests for the official Test template contract."""

from __future__ import annotations

import unittest

from aicomp_grounding.testset import (
    OFFICIAL_TEMPLATE_CANONICAL_SHA256,
    validate_official_test_template,
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


def _validate(template: dict) -> None:
    validate_official_test_template(
        template,
        expected_query_count=2,
        expected_template_sha256=None,
    )


class TestDataContractTests(unittest.TestCase):
    def test_the_untouched_layout_is_accepted(self):
        _validate(_official())

    def test_query_id_shape_is_strict(self):
        template = _official()
        template["000001_01"] = template.pop("000001_001")
        with self.assertRaisesRegex(ValueError, "Query ID"):
            _validate(template)

    def test_each_modality_path_is_strict(self):
        for field in ("visible", "infrared", "depth"):
            with self.subTest(field=field):
                template = _official()
                template["000001_001"][field] = "Images/colour/000001.png"
                with self.assertRaisesRegex(ValueError, f"invalid {field}"):
                    _validate(template)

    def test_modalities_must_refer_to_the_same_scene(self):
        for field in ("visible", "infrared", "depth"):
            with self.subTest(field=field):
                template = _official()
                template["000001_001"][field] = f"Images/{field}/999999.png"
                with self.assertRaisesRegex(ValueError, "different scenes"):
                    _validate(template)

    def test_production_defaults_pin_count_and_hash(self):
        self.assertEqual(
            OFFICIAL_TEMPLATE_CANONICAL_SHA256,
            "8fae701890bbbf05099e11ac8b2a3ead18990a496449355c9f09825d88ccfbbf",
        )
        with self.assertRaisesRegex(ValueError, "9555"):
            validate_official_test_template(_official())


if __name__ == "__main__":
    unittest.main()
