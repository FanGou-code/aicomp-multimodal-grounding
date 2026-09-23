"""Tests for the local query assembler (foundry.pipeline.assembly)."""

import unittest

from foundry.pipeline.assembly import (
    assemble_run,
    audit_assembly,
    select_targets,
)
from foundry.pipeline.facts import ObjectFacts, extract_frame_facts
from foundry.pipeline.realize import (
    fact_lines,
    facts_prompt_text,
    parse_realize_response,
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


def realize(target, _sample_id):
    """Stand-in for the teacher: reads the fact list and writes one phrase."""
    lines = facts_prompt_text(target)

    def value_of(prefix):
        for line in lines.splitlines():
            if line.startswith(f"- {prefix}"):
                rest = line[len(prefix) + 2:]
                return rest[2:] if rest.startswith(": ") else rest
        return None

    color = value_of("its color is")
    rank = value_of("counting from the left, it is number")
    if rank is not None:
        k = int(rank)
        word = TEACHER_ORDINALS[k - 1] if k <= len(TEACHER_ORDINALS) else f"number {k}"
        bits = [f"The {word}"]
        if color:
            bits.append(color)
        bits.append(target.head)
        return " ".join(bits) + " from the left"
    if color:
        return f"The {color} {target.head}"
    return f"The {target.head}"


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


class FactLinesTest(unittest.TestCase):
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

    def test_every_true_fact_is_listed(self):
        target = self._target(
            color="white", features="an open beak", count_in_head=4,
            rank_left=3, rank_right=2, side_of_image="left",
            anchors_left=((7, "boat"),), is_leftmost=True, median_mm=2400,
        )
        rendered = facts_prompt_text(target)
        for expected in (
            "the object is a: swan",
            "its color is: white",
            "notable features: an open beak",
            "the frame contains 4 of that category",
            "counting from the left, it is number 3",
            "counting from the right, it is number 2",
            "it is on the left side of the image",
            "it is on the left side of the boat",
            "no other object in the frame is further left",
            "it is about 2400 mm from the camera",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, rendered)

    def test_a_shared_color_is_still_handed_over(self):
        # Two white swans in the frame: the color is ambiguous on its own, but
        # the teacher can combine it with a rank. Nothing strips it.
        objects = [
            obj(1, "swan", [0.0, 0.4, 0.1, 0.6], color="white"),
            obj(2, "swan", [0.3, 0.4, 0.4, 0.6], color="white"),
        ]
        facts = facts_from(objects, gt_bbox=[0.0, 0.4, 0.1, 0.6])
        self.assertIn("its color is: white", facts_prompt_text(facts[0]))

    def test_nothing_is_stripped_for_frames_without_those_facts(self):
        rendered = facts_prompt_text(self._target())
        self.assertNotIn("color", rendered)
        self.assertNotIn("counting", rendered)
        self.assertNotIn("foreground", rendered)

    def test_every_line_is_a_fact_not_an_instruction(self):
        # No ranking of facts, no "use this one": the list is flat.
        target = self._target(color="white", is_leftmost=True, rank_left=1, rank_right=1)
        lines = fact_lines(target)
        self.assertTrue(all(line.startswith("- ") for line in lines))


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

        def unique_realize(target, sample_id):
            word = TEACHER_ORDINALS[target.rank_left - 1] if target.rank_left else "first"
            return f"The {word} {target.head} from the left {sample_id}"

        result = assemble_run(make_merged(frames), index, realize=unique_realize)
        self.assertTrue(result.records)
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
