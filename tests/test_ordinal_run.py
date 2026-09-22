"""Tests for the ordinal run identity, depth-path mapping, and group decision."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aicomp_grounding.ordinal import run
from aicomp_grounding.ordinal.resolve import Decision


def _decision(action: str) -> Decision:
    return Decision(action, [0.1, 0.1, 0.2, 0.2] if action == "replace" else None, "r")


class IdentityTests(unittest.TestCase):
    def _metadata(self, **overrides):
        payload = dict(
            model="qwen3_5",
            model_revision="abc123",
            max_new_tokens=1024,
            temperature=0.7,
            run_tag="t",
            limit=0,
            selected_keys=["b", "a"],
            input_fingerprint="fp",
        )
        payload.update(overrides)
        return run.build_enum_metadata(**payload)

    def test_run_id_is_prefixed_and_hex(self):
        metadata = self._metadata()
        self.assertTrue(metadata["run_id"].startswith(run.ENUM_RUN_PREFIX))
        self.assertEqual(len(metadata["run_id"]), len(run.ENUM_RUN_PREFIX) + 16)

    def test_selective_fields_change_the_run_id(self):
        base = self._metadata()["run_id"]
        for overrides in (
            {"temperature": 0.0},
            {"max_new_tokens": 2048},
            {"model": "qwen3vl"},
            {"model_revision": "other"},
            {"limit": 10},
            {"selected_keys": ["a"]},
            {"input_fingerprint": "other"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self._metadata(**overrides)["run_id"], base)

    def test_key_order_does_not_change_the_run_id(self):
        self.assertEqual(
            self._metadata(selected_keys=["b", "a"])["run_id"],
            self._metadata(selected_keys=["a", "b"])["run_id"],
        )

    def test_invalid_generation_controls_are_rejected(self):
        with self.assertRaises(ValueError):
            self._metadata(max_new_tokens=0)
        with self.assertRaises(ValueError):
            self._metadata(temperature=-0.1)

    def test_resolve_identity_needs_matching_enumerations_and_predictions(self):
        metadata = run.build_resolve_metadata(
            enum_run_ids=["a", "b"],
            prediction_fingerprints=["p", "q"],
            iou_threshold=0.5,
            run_tag="t",
            selected_keys=["k"],
        )
        self.assertTrue(metadata["run_id"].startswith(run.RESOLVE_RUN_PREFIX))
        self.assertEqual(metadata["min_replaces"], run.MIN_REPLACES)
        with self.assertRaises(ValueError):
            run.build_resolve_metadata(
                enum_run_ids=["a", "b"],
                prediction_fingerprints=["p"],
                iou_threshold=0.5,
                run_tag="t",
                selected_keys=["k"],
            )


class GroupDecisionTests(unittest.TestCase):
    def test_one_model_alone_is_not_adopted(self):
        self.assertFalse(run.adopt_replacements([]))
        self.assertFalse(run.adopt_replacements([_decision("replace")]))
        self.assertFalse(run.adopt_replacements([_decision("replace"), _decision("keep")]))

    def test_two_models_agreeing_is_adopted(self):
        self.assertTrue(run.adopt_replacements([_decision("replace"), _decision("replace")]))
        self.assertTrue(run.adopt_replacements([_decision("replace")] * 3))


class RawDepthPathTests(unittest.TestCase):
    def test_both_worker_layouts_map_back_to_the_original_file(self):
        self.assertEqual(
            run.raw_depth_relpath("Processed/Test/depth_jet/000002.png"),
            "Test/Images/depth/000002.png",
        )
        self.assertEqual(
            run.raw_depth_relpath("Processed/Train/004/depth_jet/00000001.png"),
            "Train/004/depth/00000001.png",
        )

    def test_unrecognised_paths_raise(self):
        for bad in ("Test/Images/depth/x.png", "Processed/Test/colour/x.png", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    run.raw_depth_relpath(bad)


class AxisArrayTests(unittest.TestCase):
    def test_arrays_are_loaded_and_missing_files_stay_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Test/Images/depth").mkdir(parents=True)
            (root / "Test/Images/infrared").mkdir(parents=True)
            depth = np.arange(16, dtype=np.uint16).reshape(4, 4) * 10
            Image.fromarray(depth).save(root / "Test/Images/depth/x.png")
            Image.fromarray(np.full((4, 4), 200, dtype=np.uint8)).save(
                root / "Test/Images/infrared/x.png"
            )
            item = {
                "depth": "Processed/Test/depth_jet/x.png",
                "infrared": "Test/Images/infrared/x.png",
                "width": 4,
                "height": 4,
            }
            loaded_depth, loaded_ir, size = run.load_axis_arrays(item, root)
            self.assertEqual(size, (4, 4))
            self.assertEqual(loaded_depth.shape, (4, 4))
            self.assertEqual(loaded_ir.shape, (4, 4, 3))

            missing, missing_ir, _size = run.load_axis_arrays(
                {"depth": "Processed/Test/depth_jet/gone.png", "infrared": "Test/Images/infrared/gone.png"},
                root,
            )
            self.assertIsNone(missing)
            self.assertIsNone(missing_ir)


if __name__ == "__main__":
    unittest.main()
