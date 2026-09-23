"""Tests for the local query assembler (foundry.pipeline.assembly)."""

import unittest

from foundry.pipeline.assembly import (
    assemble_run,
    audit_assembly,
    select_targets,
)
from foundry.pipeline.facts import ObjectFacts, extract_frame_facts
from foundry.pipeline.realize import (
    DIRECTION_SIDE,
    choose_dimension,
    dimension_prompt_text,
    parse_realize_response,
    rank_and_direction,
)

#: The teacher's own English, not the module's: it receives a number and picks
#: the word. Kept local so nothing under test carries an ordinal vocabulary.
TEACHER_ORDINALS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)

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


def realize(dimension, _bbox, _sample_id):
    """Stand-in for the teacher: says the facts back in fixed order."""
    fields = dimension["fields"]
    head = fields["category"]
    k = fields.get("k")
    if k is not None:
        word = TEACHER_ORDINALS[k - 1] if k <= len(TEACHER_ORDINALS) else f"number {k}"
        bits = [f"The {word}"]
        if fields.get("color"):
            bits.append(fields["color"])
        bits.append(head)
        bits.append(f"from the {DIRECTION_SIDE[fields['direction']]}")
        return " ".join(bits)
    position = fields.get("position")
    if position is not None:
        word = {
            "y2-max": "closest", "y2-min": "farthest",
            "x-min": "leftmost", "x-max": "rightmost",
            "y-min": "topmost", "y-max": "bottommost",
        }[position]
        return f"The {word} {head}"
    band = fields.get("depth_band")
    if band is not None:
        return f"The {head} in the {band}"
    side = fields.get("side_of_image")
    if side is not None:
        return f"The {head} on the {side} side"
    if fields.get("color"):
        return f"The {fields['color']} {head}"
    if fields.get("feature"):
        return f"The {head} with {fields['feature']}"
    return f"The {head}"


class ExtractFactsTest(unittest.TestCase):
    def test_ordinal_ranks_use_the_left_edge(self):
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
        # Two cars nearly side by side (left-edge gap 0.015 < ORDINAL_GAP):
        # rank becomes ambiguous -> None; the distant car keeps its rank.
        objects[1] = obj(2, "car", [0.015, 0.5, 0.115, 0.7])
        facts = facts_from(objects, gt_bbox=[0.0, 0.5, 0.1, 0.7])
        by_index = {f.index: f for f in facts}
        self.assertIsNone(by_index[1].rank_left)
        self.assertIsNone(by_index[2].rank_left)
        self.assertEqual(by_index[3].rank_left, 3)

    def test_extremes_use_edges_not_centres(self):
        # Wide box starting at x=0 vs narrow box centred further left: the
        # left edge keys the extremes, so the wide box is the leftmost one
        # even though its centre sits right of the narrow box's.
        objects = [
            obj(1, "bench", [0.00, 0.4, 0.30, 0.6]),
            obj(2, "sign", [0.08, 0.4, 0.12, 0.6]),
        ]
        facts = facts_from(objects, gt_bbox=[0.00, 0.4, 0.30, 0.6])
        by_index = {f.index: f for f in facts}
        self.assertTrue(by_index[1].is_leftmost)
        self.assertTrue(by_index[2].is_rightmost)
        # Top edge, same rule: wide-tall box owns "topmost" by its y1.
        objects = [
            obj(1, "bench", [0.00, 0.05, 0.10, 0.60]),
            obj(2, "sign", [0.20, 0.10, 0.30, 0.20]),
        ]
        facts = facts_from(objects, gt_bbox=[0.00, 0.05, 0.10, 0.60])
        by_index = {f.index: f for f in facts}
        self.assertTrue(by_index[1].is_topmost)
        self.assertTrue(by_index[2].is_bottommost)

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
        self.assertEqual([category for _, category in swan.anchors_left], ["boat"])
        self.assertEqual(swan.anchors_right, ())

    def test_reference_and_side_of_image(self):
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6]),
            obj(2, "duck", [0.95, 0.4, 1.0, 0.6]),
        ]
        facts = facts_from(objects, gt_bbox=[0.01, 0.4, 0.1, 0.6])
        self.assertTrue(facts[0].is_canary)
        self.assertFalse(facts[1].is_canary)
        self.assertEqual(facts[0].side_of_image, "left")
        self.assertEqual(facts[1].side_of_image, "right")


class DimensionTest(unittest.TestCase):
    def _target(self, **overrides) -> ObjectFacts:
        base = dict(
            index=1, category="swan", bbox=(0.3, 0.4, 0.4, 0.6), color=None,
            features=None, is_canary=False, count_in_head=1, rank_left=None,
            rank_right=None, is_leftmost=False, is_rightmost=False,
            is_topmost=False, is_bottommost=False, is_closest=False,
            is_farthest=False, side_of_image=None, anchors_left=(), anchors_right=(),
        )
        base.update(overrides)
        return ObjectFacts(**base)

    def test_rank_uses_the_smaller_side(self):
        target = self._target(rank_left=1, rank_right=4, count_in_head=4)
        self.assertEqual(rank_and_direction(target), (1, "asc"))
        target = self._target(rank_left=4, rank_right=2, count_in_head=5)
        self.assertEqual(rank_and_direction(target), (2, "desc"))
        # Ties go to "from the left".
        self.assertEqual(rank_and_direction(self._target(rank_left=2, rank_right=2)), (2, "asc"))
        self.assertIsNone(rank_and_direction(self._target()))

    def test_ordinal_wins_when_a_rank_exists(self):
        target = self._target(rank_left=2, rank_right=1, count_in_head=2, is_closest=True)
        dimension = choose_dimension(target, [target])
        self.assertEqual(dimension["family"], "ordinal_direction")
        self.assertEqual(dimension["fields"]["k"], 1)
        self.assertEqual(dimension["fields"]["direction"], "desc")

    def test_falls_back_to_an_exclusive_extreme(self):
        target = self._target(is_leftmost=True)
        dimension = choose_dimension(target, [target])
        self.assertEqual(dimension["facts"], ("x-min",))

    def test_anchor_relation_when_the_target_is_alone_on_that_side(self):
        target = self._target(anchors_left=((7, "boat"),))
        dimension = choose_dimension(target, [target])
        self.assertEqual(dimension["family"], "side_of_anchor")
        self.assertEqual(dimension["fields"]["anchor"], "boat")
        self.assertEqual(dimension["fields"]["anchor_side"], "left")
        self.assertIn("it is on the: left side of the boat", dimension_prompt_text(dimension))

    def test_anchor_relation_dropped_when_a_peer_shares_the_side(self):
        # Two swans flank the same boat: "on the left side of the boat" names
        # neither of them.
        a = self._target(index=1, bbox=(0.0, 0.4, 0.1, 0.6), anchors_left=((7, "boat"),))
        b = self._target(index=2, bbox=(0.2, 0.4, 0.3, 0.6), anchors_left=((7, "boat"),))
        self.assertIsNone(choose_dimension(a, [a, b]))

    def test_anchor_outranks_an_extreme(self):
        target = self._target(anchors_left=((7, "boat"),), is_leftmost=True)
        dimension = choose_dimension(target, [target])
        self.assertEqual(dimension["family"], "side_of_anchor")

    def test_person_head_never_gets_a_bare_color(self):
        target = self._target(category="person", color="white", count_in_head=1)
        dimension = choose_dimension(target, [target])
        self.assertIsNone(dimension["fields"].get("color"))

    def test_no_dimension_when_nothing_disambiguates(self):
        a = self._target(index=1, category="swan", bbox=(0.1, 0.4, 0.2, 0.6), count_in_head=2)
        b = self._target(index=2, category="swan", bbox=(0.5, 0.4, 0.6, 0.6), count_in_head=2)
        self.assertIsNone(choose_dimension(a, [a, b]))

    def test_the_teacher_receives_a_number_not_a_word(self):
        target = self._target(rank_left=3, rank_right=3, count_in_head=5, color="white")
        dimension = choose_dimension(target, [target])
        rendered = dimension_prompt_text(dimension)
        self.assertIn("its position: 3", rendered)
        self.assertIn("counting: from the left", rendered)
        self.assertIn("its color is: white", rendered)
        self.assertNotIn("third", rendered)
        self.assertNotIn("ordered along", rendered)
        self.assertNotIn("there are", rendered)

    def test_a_rank_past_ten_is_still_offered(self):
        # 21 same-head objects: the middle target is rank 11 from both sides and
        # is handed over as the number 11.
        target = self._target(rank_left=11, rank_right=11, count_in_head=21, is_rightmost=True)
        dimension = choose_dimension(target, [target])
        self.assertEqual(dimension["family"], "ordinal_direction")
        self.assertEqual(dimension["fields"]["k"], 11)
        self.assertIn("its position: 11", dimension_prompt_text(dimension))


class RealizeResponseTest(unittest.TestCase):
    def test_a_well_formed_reply_is_read(self):
        self.assertEqual(parse_realize_response('{"query": "The third swan"}'), "The third swan")

    def test_anything_else_is_rejected(self):
        for bad in ('{"query": ""}', '{"text": "x"}', "not json", None, "[1]"):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_realize_response(bad))


def make_merged(frames_spec):
    """frames_spec: {sample_id: (objects, attr_or_None)}; sequence id from prefix."""
    results = {}
    for sample_id, (objects, attr) in frames_spec.items():
        sequence_id = sample_id.split("_", 1)[0]
        seq = results.setdefault(sequence_id, {"status": "completed", "frames": {}, "selected": []})
        frame = {
            "findall": {
                "status": "completed", "attempts": 1, "error": "",
                "mode": "instances", "objects": objects,
            },
            "status": "completed",
            "error": "",
        }
        if attr is not None:
            frame["attr"] = {"status": "completed", **{str(k): v for k, v in attr.items()}}
        seq["frames"][sample_id] = frame
        seq["selected"].append(sample_id)
    return {"metadata": {"run_id": "census_test"}, "results": results}


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
        first = assemble_run(merged, self.index, realize=realize)
        second = assemble_run(merged, self.index, realize=realize)
        self.assertEqual([r.__dict__ for r in first.records], [r.__dict__ for r in second.records])
        self.assertLessEqual(len(first.records), 6)
        self.assertTrue(all(r.query[0].isupper() for r in first.records))
        audit = audit_assembly(first.records)
        self.assertTrue(audit["acceptance"]["verbatim_repeat_le_0_10"])
        self.assertNotIn("bucket_counts", audit)
        self.assertEqual(audit["sources"]["real"] + audit["sources"]["teacher"], audit["count"])
        # Real targets must carry the organizer GT box.
        for record in first.records:
            if record.source == "real":
                self.assertEqual(record.bbox, self.index[record.sample_id]["bbox"])

    def test_shortfall_when_frame_incomplete_or_unmapped(self):
        merged = make_merged(self.frames)
        merged["results"]["070"]["frames"]["070_00000043"]["status"] = "failed"
        result = assemble_run(merged, self.index, realize=realize)
        self.assertIn("frame-not-completed", {s["reason"] for s in result.shortfall})
        missing = make_merged({"099_00000001": self.frames["070_00000001"]})
        result = assemble_run(missing, self.index, realize=realize)
        self.assertIn("sample-missing-from-index", {s["reason"] for s in result.shortfall})

    def test_a_missing_reference_still_yields_teacher_targets(self):
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
        result = assemble_run(make_merged(frames), index, realize=realize)
        self.assertTrue(all(r.source == "teacher" for r in result.records))
        self.assertTrue(result.records)

    def test_no_records_without_a_realizer(self):
        result = assemble_run(make_merged(self.frames), self.index)
        self.assertEqual(result.records, [])

    def test_every_distinct_sentence_is_kept(self):
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

        def unique_realize(dimension, _bbox, sample_id):
            fields = dimension["fields"]
            return (
                f"The {TEACHER_ORDINALS[fields['k'] - 1]} {fields['category']} "
                f"from the {DIRECTION_SIDE[fields['direction']]} {sample_id}"
            )

        result = assemble_run(make_merged(frames), index, realize=unique_realize)
        self.assertTrue(result.records)
        self.assertTrue(all(r.family == "ordinal_direction" for r in result.records))
        audit = audit_assembly(result.records)
        self.assertEqual(audit["count"], len(result.records))
        self.assertNotIn("bucket_counts", audit)

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
