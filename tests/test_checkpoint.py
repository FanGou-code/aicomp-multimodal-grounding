"""Tests for training-data preflight validation."""

import unittest

from aicomp_grounding.query import (
    preflight_check_dataset,
)


def _make_sample(query="A red car on the left.", bbox=None):
    return {
        "visible": "Train/001/color/00000001.png",
        "infrared": "Train/001/infrared/00000001.png",
        "depth": "Train/001/depth/00000001.png",
        "query": query,
        "bbox": bbox or [0.1, 0.2, 0.3, 0.4],
        "width": 1920,
        "height": 1080,
    }


class PreflightTests(unittest.TestCase):
    def test_valid_data_passes(self):
        data = {"a": _make_sample()}
        self.assertEqual(preflight_check_dataset(data, split_name="train"), [])

    def test_empty_query_fails(self):
        data = {"a": _make_sample(query="")}
        errors = preflight_check_dataset(data, split_name="val")
        self.assertEqual(len(errors), 1)
        self.assertIn("empty", errors[0])

    def test_invalid_bbox_fails(self):
        data = {"a": _make_sample(bbox=[0.9, 0.9, 0.1, 0.1])}
        errors = preflight_check_dataset(data, split_name="train")
        self.assertTrue(any("bbox" in e for e in errors))

    def test_missing_field_fails(self):
        item = _make_sample()
        del item["depth"]
        data = {"a": item}
        errors = preflight_check_dataset(data, split_name="train")
        self.assertTrue(any("missing" in e for e in errors))

    def test_short_but_real_query_passes(self):
        data = {"a": _make_sample(query="Red car.")}
        errors = preflight_check_dataset(data, split_name="train")
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
