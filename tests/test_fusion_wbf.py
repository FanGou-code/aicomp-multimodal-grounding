"""Unit tests for the WBF fusion module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.fusion.wbf import (
    build_fusion_metadata,
    fuse_boxes,
    fuse_predictions,
    normalize_score_files,
    wbf_fuse_key,
)
from aicomp_grounding.io import atomic_write_json


def _write(root: Path, name: str, payload: dict) -> Path:
    path = root / name
    atomic_write_json(path, payload)
    return path


class FuseBoxesTests(unittest.TestCase):
    def test_equal_weights_average(self):
        fused = fuse_boxes(
            [[0.0, 0.0, 0.4, 0.4], [0.2, 0.2, 0.6, 0.6]], [1.0, 1.0]
        )
        self.assertEqual(fused, [0.1, 0.1, 0.5, 0.5])

    def test_weighted_average_pulls_toward_heavier_model(self):
        fused = fuse_boxes(
            [[0.0, 0.0, 0.4, 0.4], [0.2, 0.2, 0.6, 0.6]], [3.0, 1.0]
        )
        self.assertAlmostEqual(fused[0], 0.05)
        self.assertAlmostEqual(fused[2], 0.45)


class WbfFuseKeyTests(unittest.TestCase):
    def test_agreeing_models_fuse_to_weighted_mean(self):
        # IoU([.1,.1,.5,.5], [.15,.15,.55,.55]) = 0.1225/0.1975 ~= 0.62 >= 0.55
        fused = wbf_fuse_key(
            [([0.1, 0.1, 0.5, 0.5], 1.0), ([0.15, 0.15, 0.55, 0.55], 1.0)],
            iou_threshold=0.55,
        )
        self.assertEqual(fused, [0.125, 0.125, 0.525, 0.525])

    def test_two_light_votes_beat_one_heavy_dissenter(self):
        fused = wbf_fuse_key(
            [
                ([0.0, 0.0, 0.4, 0.4], 1.0),
                ([0.02, 0.02, 0.42, 0.42], 1.0),
                ([0.8, 0.8, 1.0, 1.0], 1.5),
            ],
            iou_threshold=0.55,
        )
        # The two agreeing boxes form the heavier cluster (2.0 > 1.5).
        self.assertLess(fused[0], 0.1)
        self.assertGreater(fused[2], 0.3)

    def test_invalid_or_empty_inputs_yield_none(self):
        self.assertIsNone(
            wbf_fuse_key([(None, 1.0), ([0.9, 0.0, 0.1, 1.0], 1.0)], iou_threshold=0.55)
        )
        self.assertIsNone(wbf_fuse_key([], iou_threshold=0.55))
        self.assertIsNone(
            wbf_fuse_key([([0.1, 0.1, 0.2, 0.2], 0.0)], iou_threshold=0.55)
        )


class FusePredictionsTests(unittest.TestCase):
    _write = staticmethod(_write)

    def test_missing_keys_and_none_boxes_are_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self._write(root, "a.json", {"q1": [0.1, 0.1, 0.5, 0.5], "q2": [0.1, 0.1, 0.2, 0.2]})
            b = self._write(root, "b.json", {"q1": None, "q3": [0.3, 0.3, 0.7, 0.7]})
            fused = fuse_predictions([a, b], weights=[1.0, 1.0])
        self.assertEqual(fused["q1"], [0.1, 0.1, 0.5, 0.5])  # only model A valid
        self.assertEqual(fused["q2"], [0.1, 0.1, 0.2, 0.2])  # only in A
        self.assertEqual(fused["q3"], [0.3, 0.3, 0.7, 0.7])  # only in B

    def test_score_files_multiply_model_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self._write(root, "a.json", {"q1": [0.0, 0.0, 0.4, 0.4]})
            b = self._write(root, "b.json", {"q1": [0.2, 0.2, 0.6, 0.6]})
            scores_b = self._write(root, "scores_b.json", {"q1": 1.0})
            scores_a = self._write(root, "scores_a.json", {"q1": 0.1})
            fused = fuse_predictions(
                [a, b],
                weights=[1.0, 1.0],
                score_files=[scores_a, scores_b],
            )
        # Effective weights 0.1 vs 1.0 -> result sits next to model B's box.
        self.assertGreater(fused["q1"][0], 0.18)

    def test_weight_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self._write(root, "a.json", {"q1": [0.1, 0.1, 0.5, 0.5]})
            with self.assertRaises(ValueError):
                fuse_predictions([a], weights=[1.0, 2.0])


class FusionMetadataTests(unittest.TestCase):
    _write = staticmethod(_write)

    def test_identity_is_deterministic_and_input_sensitive(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self._write(root, "a.json", {"q1": [0.1, 0.1, 0.5, 0.5]})
            b = self._write(root, "b.json", {"q1": [0.2, 0.2, 0.6, 0.6]})

            first = build_fusion_metadata(
                [a, b], weights=[1.0, 1.0], iou_threshold=0.55
            )
            second = build_fusion_metadata(
                [a, b], weights=[1.0, 1.0], iou_threshold=0.55
            )
            self.assertEqual(first["run_id"], second["run_id"])

            changed_weights = build_fusion_metadata(
                [a, b], weights=[2.0, 1.0], iou_threshold=0.55
            )
            self.assertNotEqual(first["run_id"], changed_weights["run_id"])

            b2 = self._write(root, "b2.json", {"q1": [0.3, 0.3, 0.8, 0.8]})
            changed_inputs = build_fusion_metadata(
                [a, b2], weights=[1.0, 1.0], iou_threshold=0.55
            )
            self.assertNotEqual(first["run_id"], changed_inputs["run_id"])

    def test_score_file_content_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            a = self._write(root, "a.json", {"q1": [0.1, 0.1, 0.5, 0.5]})
            b = self._write(root, "b.json", {"q1": [0.2, 0.2, 0.6, 0.6]})
            scores = self._write(root, "scores.json", {"q1": 1.0})

            first = build_fusion_metadata(
                [a, b],
                weights=[1.0, 1.0],
                iou_threshold=0.55,
                score_files=[scores, None],
            )
            self.assertEqual(
                first["inputs"]["scores"][0]["path"], "scores.json"
            )
            self.assertIn("sha256", first["inputs"]["scores"][0])

            self._write(root, "scores.json", {"q1": 0.5})
            changed_scores = build_fusion_metadata(
                [a, b],
                weights=[1.0, 1.0],
                iou_threshold=0.55,
                score_files=[scores, None],
            )
            self.assertNotEqual(first["run_id"], changed_scores["run_id"])


class ScoreCliPathsTests(unittest.TestCase):
    def test_blank_cli_score_values_become_none(self):
        self.assertEqual(
            normalize_score_files(["", "scores_dino.json", ""]),
            [None, Path("scores_dino.json"), None],
        )

    def test_no_score_values_returns_none(self):
        self.assertIsNone(normalize_score_files(None))


if __name__ == "__main__":
    unittest.main()
