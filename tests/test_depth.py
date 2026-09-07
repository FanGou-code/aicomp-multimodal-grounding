"""Tests for raw-depth fact extraction (foundry.depth) and its assembly consumption."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from foundry.pipeline.assembly import (
    extract_frame_facts,
    realizations_for,
    realization_is_unique,
)
from foundry.pipeline.depth import (
    depth_ranks,
    frame_depth_facts,
    is_farthest,
    is_nearest,
    load_depth_millimeters,
    object_depth_medians,
    raw_depth_path,
)


def obj(index, bbox):
    return {"i": index, "category": "swan", "bbox": bbox}


class DepthModuleTest(unittest.TestCase):
    def setUp(self):
        # 100x10 frame: column x = depth in mm (left near, right far), row 0 invalid.
        self.depth = np.tile(np.arange(1, 101, dtype=np.uint16) * 100, (10, 1))
        self.depth[0, :] = 0  # invalid sensor row

    def test_object_medians_exclude_invalid(self):
        medians = object_depth_medians(
            self.depth,
            [obj(1, [0.0, 0.0, 0.2, 1.0]), obj(2, [0.4, 0.2, 0.6, 0.8]), obj(3, [0.9, 0.0, 1.0, 1.0])],
        )
        self.assertEqual(medians[1], 1050)   # columns 0-19 → median of 100..2000
        self.assertEqual(medians[2], 5050)   # columns 40-59
        self.assertEqual(medians[3], 9550)

    def test_rank_nearest_first(self):
        medians = object_depth_medians(
            self.depth,
            [obj(1, [0.0, 0.0, 0.2, 1.0]), obj(2, [0.4, 0.2, 0.6, 0.8]), obj(3, [0.9, 0.0, 1.0, 1.0])],
        )
        self.assertEqual(depth_ranks(medians), {1: 1, 2: 2, 3: 3})

    def test_fully_invalid_object_has_no_depth(self):
        medians = object_depth_medians(self.depth, [obj(1, [0.0, 0.0, 0.5, 0.1])])
        # row 0 is invalid; bbox height 0.1*10 = 1 row → no valid pixels
        self.assertIsNone(medians[1])

    def test_nearest_farthest_margins(self):
        self.assertTrue(is_nearest(1000, [1500, 9000]))
        self.assertFalse(is_nearest(1000, [1100]))       # gap below margin
        self.assertTrue(is_farthest(9500, [1000, 5000]))
        self.assertFalse(is_farthest(9500, [9450]))
        self.assertFalse(is_nearest(None, [1000]))

    def test_png_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.png"
            Image.fromarray(self.depth, mode="I;16").save(path)
            loaded = load_depth_millimeters(path)
            self.assertTrue((loaded == self.depth).all())

    def test_raw_depth_path_derivation(self):
        self.assertEqual(
            raw_depth_path(Path("/data"), "Train/070/color/00000001.png"),
            Path("/data/Train/070/depth/00000001.png"),
        )

    def test_frame_facts_shape(self):
        facts = frame_depth_facts(
            self.depth,
            [obj(1, [0.0, 0.0, 0.2, 1.0]), obj(2, [0.4, 0.2, 0.6, 0.8])],
        )
        self.assertEqual(facts["source"], "raw-uint16-mm")
        self.assertEqual(facts["frame_min_mm"], 100)
        self.assertEqual(facts["frame_max_mm"], 10000)
        self.assertEqual(facts["objects"]["1"]["median_mm"], 1050)
        self.assertEqual(facts["ranks"], {"1": 1, "2": 2})


class AssemblyDepthTest(unittest.TestCase):
    def make_objects(self):
        return [
            obj(1, [0.0, 0.4, 0.1, 0.6]),
            obj(2, [0.4, 0.4, 0.5, 0.6]),
            obj(3, [0.9, 0.4, 1.0, 0.6]),
        ]

    def depth_facts(self, medians):
        return {
            "source": "raw-uint16-mm",
            "frame_min_mm": 100,
            "frame_max_mm": 10000,
            "objects": {str(i): {"median_mm": m} for i, m in medians.items()},
            "ranks": {str(i): r for i, r in depth_ranks(medians).items()},
        }

    def test_depth_extremes_replace_y2_proxy(self):
        # Object 2 has the largest y2 but the largest depth value (farthest):
        # depth must win over the y2 proxy.
        facts = extract_frame_facts(
            self.make_objects(),
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr=None,
            depth_facts=self.depth_facts({1: 1000, 2: 3000, 3: 9800}),
        )
        self.assertTrue(facts[0].is_closest)
        self.assertTrue(facts[2].is_farthest)
        self.assertFalse(facts[1].is_closest or facts[1].is_farthest)

    def test_foreground_background_bands(self):
        facts = extract_frame_facts(
            self.make_objects(),
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr=None,
            depth_facts=self.depth_facts({1: 500, 2: 5000, 3: 9900}),
        )
        self.assertTrue(facts[0].is_in_foreground)
        self.assertTrue(facts[2].is_in_background)
        self.assertFalse(facts[1].is_in_foreground or facts[1].is_in_background)
        texts = [r.text for r in realizations_for(facts[0])]
        self.assertIn("The swan in the foreground", texts)
        bg = [r.text for r in realizations_for(facts[2])]
        self.assertIn("The swan in the background", bg)

    def test_foreground_claim_requires_band_uniqueness(self):
        # Two same-head objects in the near band: the foreground claim is
        # ambiguous and must be rejected by the uniqueness gate.
        facts = extract_frame_facts(
            self.make_objects(),
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr=None,
            depth_facts=self.depth_facts({1: 500, 2: 800, 3: 9900}),
        )
        target = facts[0]
        fg = next(r for r in realizations_for(target) if r.facts == ("foreground",))
        self.assertFalse(realization_is_unique(fg, facts, target))

    def test_no_depth_record_falls_back_to_y2(self):
        facts = extract_frame_facts(
            [
                obj(1, [0.0, 0.0, 0.2, 0.9]),
                obj(2, [0.4, 0.0, 0.6, 0.5]),
            ],
            gt_bbox=[0.0, 0.0, 0.2, 0.9],
            attr=None,
            depth_facts=None,
        )
        self.assertTrue(facts[0].is_closest)
        self.assertIsNone(facts[0].median_mm)
        self.assertFalse(facts[0].is_in_foreground)


if __name__ == "__main__":
    unittest.main()
