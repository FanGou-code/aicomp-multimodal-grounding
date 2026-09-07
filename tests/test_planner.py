"""Tests for the Phase 3 planner (foundry.planner) and area-comparative facts."""

import unittest

from foundry.pipeline.assembly import extract_frame_facts, realizations_for
from foundry.pipeline.facts import ObjectFacts
from foundry.pipeline.planner import Allocation, TargetSupply, plan


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


def make_target(sample_id, realizations, count_in_head=1, depth=False):
    facts = ObjectFacts(
        index=1, category="swan", bbox=(0.3, 0.4, 0.4, 0.6), color=None,
        features=None, is_canary=False, count_in_head=count_in_head,
        rank_left=None, rank_right=None,
        is_leftmost=False, is_rightmost=False, is_topmost=False,
        is_bottommost=False, is_closest=False, is_farthest=False,
        side_of_image=None, anchors_left=(), anchors_right=(),
    )
    return TargetSupply(
        sample_id=sample_id, sequence_id=sample_id.split("_", 1)[0],
        source="teacher", facts=facts, gt_bbox=[0.3, 0.4, 0.4, 0.6],
        realizations=realizations, depth_available=depth,
    )


def realization(text, family="plain_attribute", facts=("color",)):
    from foundry.pipeline.assembly import Realization as R

    return R(text, family, facts, len(text.split()))


class PlannerPolicyTest(unittest.TestCase):
    def test_deterministic(self):
        supply = [
            make_target("070_00000001", [realization("The white swan")]),
            make_target("070_00000043", [realization("The black swan", facts=("area-comparative",))]),
        ]
        first = plan(supply, SPEC)
        second = plan(supply, SPEC)
        self.assertEqual(
            [(a.supply.sample_id, a.realization.text, a.bucket) for a in first.allocations],
            [(a.supply.sample_id, a.realization.text, a.bucket) for a in second.allocations],
        )

    def test_verbatim_uniqueness_across_targets(self):
        supply = [
            make_target("070_00000001", [realization("The white swan")]),
            make_target("070_00000043", [realization("The white swan")]),
        ]
        result = plan(supply, SPEC)
        self.assertEqual(len(result.allocations), 1)
        self.assertEqual(result.unallocated[0]["reason"], "no-unique-realization")

    def _crowd_supply(self, crowd_count_in_head):
        band = realization("The swan in the foreground", "superlative_camera", ("foreground",))
        ordinal = realization("The first swan from the left", "ordinal_direction", ("rank:1",))
        supply = [
            make_target("070_00000001", [ordinal, band], count_in_head=crowd_count_in_head, depth=True)
        ]
        for n in range(2, 21):
            supply.append(
                make_target(f"070_{n:08d}", [realization(f"The white swan number {n}")])
            )
        return supply

    def test_crowd_frame_prefers_distance_bucket(self):
        # A crowd frame (3+ same-head) with a band realization: the distance
        # bucket gets first pick even though ordinal has more remaining quota.
        result = plan(self._crowd_supply(3), SPEC)
        # "in the foreground" lands in attribute_action under the frozen rule
        self.assertEqual(result.allocations[0].bucket, "attribute_action")
        self.assertEqual(result.allocations[0].realization.text, "The swan in the foreground")

    def test_non_crowd_frame_keeps_quota_order(self):
        result = plan(self._crowd_supply(1), SPEC)
        self.assertEqual(result.allocations[0].bucket, "ordinal")

    def test_family_diversity_within_sequence(self):
        # Same family twice in a sequence: the second pick prefers the other
        # family when word counts are equal.
        supply = [
            make_target("070_00000001", [realization("The white swan pair one")]),
            make_target("070_00000002", [
                realization("The white swan pair two"),
                realization("The swan on the left side of the image", "side_of_anchor",
                            ("image:left",)),
            ]),
        ]
        result = plan(supply, SPEC)
        self.assertNotEqual(
            result.allocations[0].realization.family,
            result.allocations[1].realization.family,
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

    def test_area_ratio_fact_and_realizations(self):
        # Object 1 covers 4x the area of object 2.
        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.4, 0.4], [0.5, 0.0, 0.6, 0.1]),
            gt_bbox=[0.0, 0.0, 0.4, 0.4],
            attr=None,
        )
        self.assertEqual(facts[0].area_ratio_lead, 16.0)
        texts = [r.text for r in realizations_for(facts[0])]
        self.assertIn("The larger rock", texts)
        self.assertNotIn("The largest rock", texts)  # only 2 rivals

    def test_comparative_needs_strict_ratio(self):
        # Ratio 2.0 vs 1.2: only the strict one earns the comparative.
        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.3, 0.4], [0.5, 0.0, 0.62, 0.2]),
            gt_bbox=[0.0, 0.0, 0.3, 0.4],
            attr=None,
        )
        self.assertLess(facts[1].area_ratio_lead, 1.5)
        texts2 = [r.text for r in realizations_for(facts[1])]
        self.assertNotIn("The larger rock", texts2)

    def test_superlative_with_three(self):
        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.4, 0.4], [0.5, 0.0, 0.6, 0.1], [0.7, 0.0, 0.8, 0.1]),
            gt_bbox=[0.0, 0.0, 0.4, 0.4],
            attr=None,
        )
        texts = [r.text for r in realizations_for(facts[0])]
        self.assertIn("The largest rock", texts)
        self.assertIn("The larger rock", texts)

    def test_comparative_unique_by_construction(self):
        from foundry.pipeline.assembly import realization_is_unique

        facts = extract_frame_facts(
            self.make_objects([0.0, 0.0, 0.4, 0.4], [0.5, 0.0, 0.6, 0.1]),
            gt_bbox=[0.0, 0.0, 0.4, 0.4],
            attr=None,
        )
        target = facts[0]
        comp = next(r for r in realizations_for(target) if r.facts == ("area-comparative",))
        self.assertTrue(realization_is_unique(comp, facts, target))


if __name__ == "__main__":
    unittest.main()
