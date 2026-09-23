"""Tests for the ordinal run identity, depth-path mapping, and group decision."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from aicomp_grounding.ordinal import run


class IdentityTests(unittest.TestCase):
    def _metadata(self, **overrides):
        payload = dict(
            model="qwen3_5",
            model_revision="abc123",
            max_new_tokens=4096,
            temperature=0.6,
            think=True,
            sampling={"top_p": 0.95, "top_k": 20, "presence_penalty": 0.0},
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
            {"think": False},
            {"sampling": {"top_p": 0.8, "top_k": 20, "presence_penalty": 0.0}},
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

    def test_resolve_identity_covers_one_enumeration_and_one_prediction(self):
        metadata = run.build_resolve_metadata(
            enum_run_id="a",
            prediction_fingerprint="p",
            iou_threshold=0.5,
            run_tag="t",
            selected_keys=["k"],
        )
        self.assertTrue(metadata["run_id"].startswith(run.RESOLVE_RUN_PREFIX))


class ThinkingSidecarTests(unittest.TestCase):
    def test_thinking_round_trips_and_survives_newlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = {"a": "I count 3.\nThen I list them.", "b": "one line"}
            run.write_thinking(root, rows)
            self.assertEqual(run.load_thinking(root), rows)

    def test_a_missing_sidecar_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(run.load_thinking(Path(tmp)), {})


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
