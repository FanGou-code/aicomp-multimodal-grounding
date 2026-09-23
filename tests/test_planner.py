"""Tests for the Phase 3 planner (foundry.planner) and area-comparative facts."""

import unittest

from foundry.pipeline.facts import ObjectFacts, extract_frame_facts
from foundry.pipeline.planner import TargetSupply, plan


def make_target(sample_id, realizations):
    facts = ObjectFacts(
        index=1, category="swan", bbox=(0.3, 0.4, 0.4, 0.6), color=None,
        features=None, is_canary=False, count_in_head=1,
        rank_left=None, rank_right=None,
        is_leftmost=False, is_rightmost=False, is_topmost=False,
        is_bottommost=False, is_closest=False, is_farthest=False,
        side_of_image=None, anchors_left=(), anchors_right=(),
    )
    return TargetSupply(
        sample_id=sample_id, sequence_id=sample_id.split("_", 1)[0],
        source="teacher", facts=facts, gt_bbox=[0.3, 0.4, 0.4, 0.6],
        realizations=realizations,
    )


def realization(text, family="plain_attribute", facts=("color",)):
    from foundry.pipeline.facts import Realization as R

    return R(text, family, facts, len(text.split()))


class PlannerTest(unittest.TestCase):
    def test_one_sentence_per_target_in_supply_order(self):
        supply = [
            make_target("001_00000001", [realization("the white swan")]),
            make_target("001_00000002", [realization("the black swan")]),
        ]
        result = plan(supply)
        self.assertEqual([a.realization.text for a in result.allocations],
                         ["the white swan", "the black swan"])
        self.assertEqual(result.unallocated, [])

    def test_the_same_sentence_is_never_emitted_twice(self):
        supply = [
            make_target("001_00000001", [realization("the white swan")]),
            make_target("001_00000002", [realization("the white swan")]),
        ]
        result = plan(supply)
        self.assertEqual(len(result.allocations), 1)
        self.assertEqual(result.unallocated[0]["reason"], "sentence-already-used")
        self.assertEqual(result.unallocated[0]["sample_id"], "001_00000002")

    def test_deduplication_ignores_case(self):
        supply = [
            make_target("001_00000001", [realization("The white swan")]),
            make_target("001_00000002", [realization("the WHITE swan")]),
        ]
        self.assertEqual(len(plan(supply).allocations), 1)

    def test_a_target_with_no_sentence_is_reported(self):
        supply = [make_target("001_00000001", [])]
        result = plan(supply)
        self.assertEqual(result.allocations, [])
        self.assertEqual(result.unallocated[0]["reason"], "sentence-already-used")

    def test_deterministic(self):
        supply = [
            make_target("001_00000001", [realization("the white swan")]),
            make_target("002_00000001", [realization("the boat")]),
        ]
        first, second = plan(supply), plan(supply)
        self.assertEqual(
            [(a.supply.sample_id, a.realization.text) for a in first.allocations],
            [(a.supply.sample_id, a.realization.text) for a in second.allocations],
        )

    def test_only_the_order_follows_the_input(self):
        a = make_target("001_00000001", [realization("the white swan")])
        b = make_target("001_00000002", [realization("the black swan")])
        self.assertEqual(
            [x.realization.text for x in plan([a, b]).allocations],
            ["the white swan", "the black swan"],
        )
        self.assertEqual(
            [x.realization.text for x in plan([b, a]).allocations],
            ["the black swan", "the white swan"],
        )


class AreaComparativeTest(unittest.TestCase):
    def make_objects(self, bbox1, bbox2, bbox3=None):
        objects = [
            {"i": 1, "category": "rock", "bbox": bbox1},
            {"i": 2, "category": "rock", "bbox": bbox2},
        ]
        if bbox3:
            objects.append({"i": 3, "category": "rock", "bbox": bbox3})
        return objects

    def test_area_ratio_fact_still_computed(self):
        # Object 1 covers 4x the area of object 2.
        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.4, 0.4], [0.5, 0.0, 0.6, 0.1]),
            gt_bbox=[0.0, 0.0, 0.4, 0.4],
            attr=None,
        )
        self.assertEqual(facts[0].area_ratio_lead, 16.0)

    def test_comparative_needs_strict_ratio(self):
        # Ratio 2.0 vs 1.2: only the strict one earns the comparative.
        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.3, 0.4], [0.5, 0.0, 0.62, 0.2]),
            gt_bbox=[0.0, 0.0, 0.3, 0.4],
            attr=None,
        )
        self.assertLess(facts[1].area_ratio_lead, 1.5)


if __name__ == "__main__":
    unittest.main()
