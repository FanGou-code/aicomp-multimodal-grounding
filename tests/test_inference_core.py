"""Offline tests for the platform-agnostic inference core."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.serving.engine.inference_core import (
    evaluate_predictions,
    load_inference_items,
)
from aicomp_grounding.io import atomic_write_json


class LoadInferenceItemsTests(unittest.TestCase):
    def test_bad_official_entry_is_rejected_before_path_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "queries.json"
            atomic_write_json(path, {
                "a": {"visible": "Images/visible/a.png", "infrared": "Images/infrared/a.png",
                      "depth": "Images/depth/a.png", "query": "the car"},
                "b": None,
            })
            with self.assertRaisesRegex(ValueError, "'b'"):
                load_inference_items(path)

    def test_flat_index_gains_keys_and_returns_no_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.json"
            atomic_write_json(path, {"q1": {"query": "the red car"}})
            items, metadata = load_inference_items(path)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["key"], "q1")
        self.assertEqual(items[0]["query"], "the red car")
        self.assertIsNone(metadata)

    def test_approved_artifact_shape_yields_metadata_and_data_items(self):
        artifact = {
            "metadata": {"run_id": "annot_x", "split": "val"},
            "data": {"q1": {"query": "the red car", "bbox": [0.1, 0.1, 0.4, 0.5]}},
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "approved.json"
            atomic_write_json(path, artifact)
            items, metadata = load_inference_items(path)
        self.assertEqual(metadata, artifact["metadata"])
        self.assertEqual(items[0]["key"], "q1")
        self.assertIn("bbox", items[0])

    def test_flat_index_limit_truncates(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.json"
            atomic_write_json(
                path,
                {"q1": {"query": "a"}, "q2": {"query": "b"}, "q3": {"query": "c"}},
            )
            items, metadata = load_inference_items(path, limit=2)
        self.assertEqual([item["key"] for item in items], ["q1", "q2"])
        self.assertIsNone(metadata)

    def test_official_template_is_mapped_to_the_dataset_layout(self):
        official = {
            "000001_001": {
                "visible": "Images/visible/000001.png",
                "infrared": "Images/infrared/000001.png",
                "depth": "Images/depth/000001.png",
                "query": "The person beside the white van",
            }
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "queries.json"
            atomic_write_json(path, official)
            items, metadata = load_inference_items(path)
        item = items[0]
        self.assertEqual(item["visible"], "Test/Images/visible/000001.png")
        self.assertEqual(item["infrared"], "Test/Images/infrared/000001.png")
        self.assertEqual(item["depth"], "Test/Images/depth/000001.png")
        self.assertEqual(item["key"], "000001_001")
        self.assertIsNone(metadata)

    def test_non_object_shapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.json"
            path.write_text('"just-a-string"', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_inference_items(path)


class EvaluatePredictionsTests(unittest.TestCase):
    @staticmethod
    def _items():
        return [
            {"key": "hit", "query": "a", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"key": "miss", "query": "b", "bbox": [0.0, 0.0, 1.0, 1.0]},
            {"key": "failed", "query": "c", "bbox": [0.0, 0.0, 1.0, 1.0]},
        ]

    def test_metrics_cover_hits_failures_and_mean_iou(self):
        predictions = {
            "hit": [0.0, 0.0, 1.0, 1.0],      # IoU 1.0 -> hit
            "miss": [0.0, 0.0, 0.5, 0.5],     # IoU 0.25 -> miss
            "failed": None,                    # unparsable -> failure
        }
        metrics = evaluate_predictions(self._items(), predictions)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["failures"], 1)
        self.assertAlmostEqual(metrics["acc_at_0_5"], 1 / 3)
        self.assertAlmostEqual(metrics["mean_iou"], (1.0 + 0.25) / 3)

    def test_items_without_ground_truth_return_none(self):
        items = [{"key": "q1", "query": "the red car"}]
        self.assertIsNone(evaluate_predictions(items, {"q1": [0, 0, 1, 1]}))

    def test_empty_items_return_none(self):
        self.assertIsNone(evaluate_predictions([], {}))


if __name__ == "__main__":
    unittest.main()
