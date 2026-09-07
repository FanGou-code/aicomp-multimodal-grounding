"""Tests for the Phase 2 local query assembler (foundry.assembly)."""

import unittest

from foundry.pipeline.assembly import (
    assemble_run,
    audit_assembly,
    realization_is_unique,
    realizations_for,
    select_targets,
)
from foundry.pipeline.buckets import classify_frozen, parse_spec_shares
from foundry.pipeline.facts import ObjectFacts, extract_frame_facts

SPEC = {
    "style_buckets_draft": {
        "shares": {
            "ordinal": "3201 (335 per-mille)",
            "spatial": "2465 (258 per-mille)",
            "attribute_action": "2450 (256 per-mille)",
            "distance": "1439 (151 per-mille)",
        }
    }
}


def obj(index, category, bbox, color=None, features=None):
    entry = {"i": index, "category": category, "bbox": bbox}
    if color is not None or features is not None:
        entry["__attr__"] = {"color": color, "features": features}
    return entry


def facts_from(objects, gt_bbox, attr=None):
    """Build ObjectFacts with per-object attr entries."""
    attr_map = {
        o["i"]: o.pop("__attr__")
        for o in objects
        if "__attr__" in o
    } if any("__attr__" in o for o in objects) else (attr or {})
    return extract_frame_facts(objects, gt_bbox, attr_map)


class ClassifyFrozenTest(unittest.TestCase):
    def test_frozen_bucket_rule(self):
        # Frozen rule: ordinal > distance > spatial > attribute_action, and
        # superlatives fall in the ordinal bucket (RE_ORD includes them).
        self.assertEqual(classify_frozen("The first swan from the left"), "ordinal")
        self.assertEqual(classify_frozen("The swan closest to the camera"), "ordinal")
        self.assertEqual(classify_frozen("The swan on the left side of the boat"), "spatial")
        self.assertEqual(classify_frozen("The white swan"), "attribute_action")
        self.assertEqual(classify_frozen("The person wearing a hat"), "attribute_action")


class ExtractFactsTest(unittest.TestCase):
    def test_ordinal_ranks_and_gap_gate(self):
        objects = [
            obj(1, "car", [0.0, 0.5, 0.1, 0.7]),
            obj(2, "car", [0.2, 0.5, 0.3, 0.7]),
            obj(3, "car", [0.5, 0.5, 0.6, 0.7]),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.5, 0.1, 0.7])
        by_index = {f.index: f for f in facts}
        self.assertEqual(by_index[1].rank_left, 1)
        self.assertEqual(by_index[2].rank_left, 2)
        self.assertEqual(by_index[2].rank_right, 2)
        self.assertEqual(by_index[3].rank_right, 1)
        # Two cars nearly side by side (center-x gap 0.015 < ORDINAL_GAP):
        # rank becomes ambiguous -> None; the distant car keeps its rank.
        objects[1] = obj(2, "car", [0.015, 0.5, 0.115, 0.7])
        facts = facts_from(objects, gt_bbox=[0.0, 0.5, 0.1, 0.7])
        by_index = {f.index: f for f in facts}
        self.assertIsNone(by_index[1].rank_left)
        self.assertIsNone(by_index[2].rank_left)
        self.assertEqual(by_index[3].rank_left, 3)

    def test_head_group_merges_variants(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6]),
            obj(2, "black swan", [0.3, 0.4, 0.4, 0.6]),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        by_index = {f.index: f for f in facts}
        self.assertEqual(by_index[1].count_in_head, 2)
        self.assertEqual(by_index[1].rank_left, 1)
        self.assertEqual(by_index[2].rank_left, 2)

    def test_extreme_flags_require_margin(self):
        objects = [
            obj(1, "cat", [0.0, 0.0, 0.1, 0.2]),
            obj(2, "dog", [0.03, 0.0, 0.13, 0.2]),
            obj(3, "bird", [0.8, 0.0, 0.9, 0.2]),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.0, 0.1, 0.2])
        by_index = {f.index: f for f in facts}
        self.assertFalse(by_index[1].is_leftmost)   # runner-up only 0.015 away
        self.assertFalse(by_index[2].is_leftmost)
        self.assertTrue(by_index[3].is_rightmost)

    def test_closest_farthest_by_bottom_edge(self):
        objects = [
            obj(1, "cat", [0.0, 0.0, 0.2, 0.9]),
            obj(2, "dog", [0.4, 0.0, 0.6, 0.5]),
            obj(3, "bird", [0.7, 0.0, 0.9, 0.1]),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.0, 0.2, 0.9])
        by_index = {f.index: f for f in facts}
        self.assertTrue(by_index[1].is_closest)
        self.assertTrue(by_index[3].is_farthest)
        self.assertFalse(by_index[2].is_closest)
        self.assertFalse(by_index[2].is_farthest)

    def test_anchor_requires_unique_head(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6]),
            obj(2, "boat", [0.5, 0.4, 0.6, 0.6]),
            obj(3, "boat", [0.8, 0.4, 0.9, 0.6]),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        swan = facts[0]
        self.assertEqual(swan.anchors_right, ())  # two boats: ambiguous anchor
        objects.pop()
        facts = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        swan = facts[0]
        # The boat sits wholly right of the swan -> phrase "left side of the boat".
        self.assertEqual([category for _, category in swan.anchors_left], ["boat"])
        self.assertEqual(swan.anchors_right, ())

    def test_canary_and_side_of_image(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6]),
            obj(2, "duck", [0.95, 0.4, 1.0, 0.6]),
        ]
        facts = facts_from(objects, gt_bbox=[0.01, 0.4, 0.1, 0.6])
        self.assertTrue(facts[0].is_canary)
        self.assertFalse(facts[1].is_canary)
        self.assertEqual(facts[0].side_of_image, "left")
        self.assertEqual(facts[1].side_of_image, "right")


class RealizationTest(unittest.TestCase):
    def test_ordinal_text_and_color_order(self):
        f = ObjectFacts(
            index=2, category="swan", bbox=(0.3, 0.4, 0.4, 0.6), color="white",
            features=None, is_canary=False, count_in_head=3, rank_left=1, rank_right=3,
            is_leftmost=False, is_rightmost=False, is_topmost=False, is_bottommost=False,
            is_closest=False, is_farthest=False, side_of_image=None,
            anchors_left=(), anchors_right=(),
        )
        texts = [r.text for r in realizations_for(f)]
        self.assertIn("The first white swan from the left", texts)
        self.assertIn("The first white swan from left to right", texts)
        self.assertIn("The third white swan from right to left", texts)
        self.assertNotIn("The white first swan from the left", texts)

    def test_superlative_realization(self):
        f = ObjectFacts(
            index=1, category="crane", bbox=(0.3, 0.4, 0.4, 0.6), color="white",
            features=None, is_canary=False, count_in_head=1, rank_left=None, rank_right=None,
            is_leftmost=False, is_rightmost=False, is_topmost=False, is_bottommost=False,
            is_closest=True, is_farthest=False, side_of_image=None,
            anchors_left=(), anchors_right=(),
        )
        texts = [r.text for r in realizations_for(f)]
        self.assertIn("The white crane closest to the camera", texts)
        self.assertIn("The closest white crane", texts)

    def test_attribute_realization_articles(self):
        f = ObjectFacts(
            index=1, category="orange ball", bbox=(0.3, 0.4, 0.4, 0.6), color="red",
            features=None, is_canary=False, count_in_head=1, rank_left=None, rank_right=None,
            is_leftmost=False, is_rightmost=False, is_topmost=False, is_bottommost=False,
            is_closest=False, is_farthest=False, side_of_image=None,
            anchors_left=(), anchors_right=(),
        )
        texts = [r.text for r in realizations_for(f)]
        self.assertIn("The red ball", texts)
        self.assertIn("A red ball", texts)
        self.assertNotIn("The red orange", texts)

    def test_action_feature_realization(self):
        f = ObjectFacts(
            index=1, category="person", bbox=(0.3, 0.4, 0.4, 0.6), color=None,
            features="wearing a hat", is_canary=False, count_in_head=1,
            rank_left=None, rank_right=None,
            is_leftmost=False, is_rightmost=False, is_topmost=False, is_bottommost=False,
            is_closest=False, is_farthest=False, side_of_image=None,
            anchors_left=(), anchors_right=(),
        )
        texts = [r.text for r in realizations_for(f)]
        self.assertIn("The person wearing a hat", texts)

    def test_any_ing_feature_uses_action_form(self):
        facts = facts_from(
            [obj(1, "swan", [0.0, 0.4, 0.1, 0.6])],
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr={1: {"color": "white", "features": "swimming in water"}},
        )
        texts = [r.text for r in realizations_for(facts[0])]
        self.assertIn("The swan swimming in water", texts)
        self.assertNotIn("The swan with swimming in water", texts)

    def test_features_head_prefix_stripped(self):
        facts = facts_from(
            [obj(1, "swan", [0.0, 0.4, 0.1, 0.6])],
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr={1: {"color": "white", "features": "swan with neck curved downward"}},
        )
        texts = [r.text for r in realizations_for(facts[0])]
        self.assertIn("The swan with neck curved downward", texts)
        self.assertFalse(any("swan with swan" in t for t in texts))

    def test_person_head_suppresses_bare_color(self):
        facts = facts_from(
            [obj(1, "person", [0.0, 0.4, 0.1, 0.6]), obj(2, "car", [0.4, 0.4, 0.5, 0.6])],
            gt_bbox=[0.0, 0.4, 0.1, 0.6],
            attr={1: {"color": "white", "features": ""}, 2: {"color": "red", "features": ""}},
        )
        person = next(f for f in facts if f.head == "person")
        car = next(f for f in facts if f.head == "car")
        person_texts = [r.text for r in realizations_for(person)]
        car_texts = [r.text for r in realizations_for(car)]
        self.assertFalse(any("white person" in t for t in person_texts))
        self.assertNotIn("The white person", person_texts)
        self.assertIn("The red car", car_texts)

    def test_uniqueness_rejects_ambiguous_color(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6], color="white"),
            obj(2, "swan", [0.4, 0.4, 0.5, 0.6], color="white"),
        ]
        frame = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        target = next(f for f in frame if f.index == 1)
        color_variant = next(r for r in realizations_for(target) if r.facts == ("color",))
        self.assertFalse(realization_is_unique(color_variant, frame, target))

    def test_uniqueness_conservative_unknown_color(self):
        # A same-head object with unknown color counts as a potential match.
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6], color="white"),
            obj(2, "swan", [0.4, 0.4, 0.5, 0.6], color=None),
        ]
        frame = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        target = next(f for f in frame if f.index == 1)
        color_variant = next(r for r in realizations_for(target) if r.facts == ("color",))
        self.assertFalse(realization_is_unique(color_variant, frame, target))

    def test_uniqueness_unique_color_passes(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6], color="white"),
            obj(2, "swan", [0.4, 0.4, 0.5, 0.6], color="black"),
        ]
        frame = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        target = next(f for f in frame if f.index == 1)
        color_variant = next(r for r in realizations_for(target) if r.facts == ("color",))
        self.assertTrue(realization_is_unique(color_variant, frame, target))


def make_merged(frames_spec):
    """frames_spec: {sample_id: (objects, attr_or_None)}; sequence id from prefix."""
    results = {}
    for sample_id, (objects, attr) in frames_spec.items():
        sequence_id = sample_id.split("_", 1)[0]
        seq = results.setdefault(sequence_id, {"status": "completed", "frames": {}, "selected": []})
        frame = {
            "findall_1": {"status": "completed", "attempts": 1, "error": "", "objects": objects},
            "findall_2": {"status": "completed", "attempts": 1, "error": "", "objects": objects},
            "status": "completed",
            "error": "",
            "agreement": {
                "matched": len(objects), "count_a": len(objects), "count_b": len(objects),
                "count_agree": True, "jaccard": 1.0,
            },
        }
        if attr is not None:
            frame["attr"] = {"status": "completed", **{str(k): v for k, v in attr.items()}}
        seq["frames"][sample_id] = frame
        seq["selected"].append(sample_id)
    return {"metadata": {"run_id": "census_test"}, "results": results}


def make_index(frames_spec):
    return {
        sample_id: {"bbox": objects[0]["bbox"], "visible": f"Train/x/color.png"}
        for sample_id, (objects, _) in frames_spec.items()
    }


class AssembleRunTest(unittest.TestCase):
    def setUp(self):
        self.frames = {
            "070_00000001": (
                [
                    obj(1, "car", [0.0, 0.5, 0.1, 0.7], color="white", features="sedan shape"),
                    obj(2, "car", [0.2, 0.5, 0.3, 0.7], color="black", features="sedan shape"),
                    obj(3, "car", [0.5, 0.5, 0.6, 0.7], color="white", features="van shape"),
                ],
                {1: {"color": "white", "features": "sedan shape"},
                 2: {"color": "black", "features": "sedan shape"},
                 3: {"color": "white", "features": "van shape"}},
            ),
            "070_00000043": (
                [
                    obj(1, "swan", [0.05, 0.3, 0.2, 0.5], color="white"),
                    obj(2, "swan", [0.4, 0.3, 0.55, 0.5], color="white"),
                    obj(3, "boat", [0.7, 0.2, 0.9, 0.6], color="brown"),
                ],
                {1: {"color": "white", "features": "long neck"},
                 2: {"color": "white", "features": "long neck"},
                 3: {"color": "brown", "features": "wooden hull"}},
            ),
        }
        self.index = {
            "070_00000001": {"bbox": [0.2, 0.5, 0.3, 0.7], "visible": "x"},
            "070_00000043": {"bbox": [0.4, 0.3, 0.55, 0.5], "visible": "x"},
        }

    def test_end_to_end_deterministic_and_accepted(self):
        merged = make_merged(self.frames)
        first = assemble_run(merged, self.index, SPEC)
        second = assemble_run(merged, self.index, SPEC)
        self.assertEqual([r.__dict__ for r in first.records], [r.__dict__ for r in second.records])
        # 2 frames x (1 real + up to 2 teachers)
        self.assertLessEqual(len(first.records), 6)
        self.assertTrue(all(r.query[0].isupper() for r in first.records))
        self.assertTrue(all(3 <= r.words <= 18 for r in first.records))
        audit = audit_assembly(first.records)
        self.assertTrue(audit["acceptance"]["verbatim_repeat_le_0_10"])
        self.assertEqual(audit["sources"]["real"] + audit["sources"]["teacher"], audit["count"])
        # Real targets must carry the organizer GT box.
        real_records = [r for r in first.records if r.source == "real"]
        for record in real_records:
            self.assertEqual(record.bbox, self.index[record.sample_id]["bbox"])

    def test_shortfall_when_frame_incomplete_or_unmapped(self):
        merged = make_merged(self.frames)
        merged["results"]["070"]["frames"]["070_00000043"]["status"] = "failed"
        result = assemble_run(merged, self.index, SPEC)
        reasons = {s["reason"] for s in result.shortfall}
        self.assertIn("frame-not-completed", reasons)
        missing = make_merged({"099_00000001": self.frames["070_00000001"]})
        result = assemble_run(missing, self.index, SPEC)
        self.assertIn(
            "sample-missing-from-index", {s["reason"] for s in result.shortfall}
        )

    def test_canary_absence_reported_and_teachers_kept(self):
        # GT box matches nothing in the agreed set: no real target, but the
        # teacher targets must still assemble, and the gap must be visible.
        frames = {
            "070_00000001": (
                [
                    obj(1, "car", [0.0, 0.5, 0.1, 0.7], color="red"),
                    obj(2, "car", [0.3, 0.5, 0.4, 0.7], color="blue"),
                ],
                {1: {"color": "red", "features": ""},
                 2: {"color": "blue", "features": ""}},
            ),
        }
        index = {"070_00000001": {"bbox": [0.9, 0.9, 0.95, 0.95], "visible": "x"}}
        result = assemble_run(make_merged(frames), index, SPEC)
        self.assertIn(
            "no-canary-in-agreed-objects", {s["reason"] for s in result.shortfall}
        )
        self.assertTrue(all(r.source == "teacher" for r in result.records))
        self.assertEqual(len(result.records), 2)

    def test_quota_overshoot_reported(self):
        # Only ordinal-supported targets: ordinal quota fills, rest overshoots.
        frames = {
            f"070_{i:08d}": (
                [
                    obj(1, "car", [0.0, 0.5, 0.1, 0.7]),
                    obj(2, "car", [0.3, 0.5, 0.4, 0.7]),
                ],
                None,
            )
            for i in range(1, 6)
        }
        index = {sid: {"bbox": o[0]["bbox"], "visible": "x"} for sid, (o, _) in frames.items()}
        result = assemble_run(make_merged(frames), index, SPEC)
        self.assertTrue(result.records)
        self.assertTrue(
            any(r.quota_state == "overshoot" for r in result.records),
            "supply skewed to one bucket should produce overshoot records",
        )
        audit = audit_assembly(result.records)
        self.assertEqual(
            audit["overshoot_records"],
            sum(1 for r in result.records if r.quota_state == "overshoot"),
        )

    def test_parse_spec_shares_and_fallback(self):
        self.assertEqual(
            parse_spec_shares(SPEC),
            {"ordinal": 335, "spatial": 258, "attribute_action": 256, "distance": 151},
        )
        self.assertEqual(parse_spec_shares(None), parse_spec_shares({}))
        broken = {"style_buckets_draft": {"shares": {"ordinal": "oops"}}}
        self.assertEqual(parse_spec_shares(broken), parse_spec_shares(None))

    def test_select_targets_quality_order_and_area_gate(self):
        objects = [
            obj(1, "car", [0.0, 0.5, 0.1, 0.7], color="white", features="clean"),
            obj(2, "car", [0.2, 0.5, 0.3, 0.7], color="black"),
            obj(3, "car", [0.4, 0.5, 0.5, 0.7]),
            obj(4, "dust", [0.6, 0.5, 0.605, 0.51]),  # area < MIN_TEACHER_AREA
        ]
        gt = [0.0, 0.5, 0.1, 0.7]
        frame = facts_from(objects, gt, {1: {"color": "white", "features": "clean"},
                                         2: {"color": "black", "features": ""},
                                         3: {"color": "", "features": ""}})
        targets = select_targets(frame)
        self.assertEqual(targets[0][0], "real")
        teachers = [source for source, _ in targets if source == "teacher"]
        self.assertEqual(len(teachers), 2)  # dust gated out; max 2
        self.assertIn(2, [f.index for _, f in targets[1:]])


if __name__ == "__main__":
    unittest.main()
