"""Tests for the reverse pass (foundry.pipeline.reverse)."""

import json
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.annotation.reverse import (
    AXES,
    axis_value,
    parse_direct_response,
    parse_enumeration_response,
    parse_intent_response,
    pick_kth,
)


def intent(payload) -> str:
    return json.dumps(payload)


def rank(k, axis="x", direction="asc", category="car") -> str:
    return intent({
        "category": category,
        "selection": {"mode": "rank", "k": k, "axis": axis, "direction": direction},
    })


class IntentParseTest(unittest.TestCase):
    def test_a_rank_query_is_read(self):
        self.assertEqual(
            parse_intent_response(rank(3, "y", "desc", "swan")),
            {"category": "swan", "mode": "rank", "k": 3, "axis": "y", "direction": "desc"},
        )

    def test_a_unique_query_is_read(self):
        self.assertEqual(
            parse_intent_response(intent({"category": "boat", "selection": {"mode": "unique"}})),
            {"category": "boat", "mode": "unique"},
        )

    def test_bad_k_is_rejected(self):
        for k in (0, -1, 1.5, "2", True, None):
            with self.subTest(k=k):
                self.assertIsNone(parse_intent_response(rank(k)))

    def test_axis_outside_the_reverse_set_is_rejected(self):
        # depth / ir need images this pass does not send.
        for axis in ("depth", "ir", "z"):
            with self.subTest(axis=axis):
                self.assertIsNone(parse_intent_response(rank(1, axis=axis)))

    def test_a_missing_axis_is_rejected(self):
        payload = intent({"category": "car", "selection": {"mode": "rank", "k": 1, "direction": "asc"}})
        self.assertIsNone(parse_intent_response(payload))

    def test_direction_outside_the_set_is_rejected(self):
        self.assertIsNone(parse_intent_response(rank(1, direction="up")))

    def test_malformed_replies_are_rejected(self):
        for bad in ("not json", "", None, "[1]", intent({"category": ""})):
            with self.subTest(bad=bad):
                self.assertIsNone(parse_intent_response(bad))


class EnumerationParseTest(unittest.TestCase):
    def test_a_counted_list_is_read(self):
        payload = intent({
            "count": 2,
            "objects": [{"bbox": [0.1, 0.2, 0.3, 0.4]}, {"bbox": [0.5, 0.2, 0.7, 0.4]}],
        })
        self.assertEqual(len(parse_enumeration_response(payload)["boxes"]), 2)

    def test_a_count_that_does_not_match_the_list_is_rejected(self):
        payload = intent({"count": 3, "objects": [{"bbox": [0.1, 0.2, 0.3, 0.4]}]})
        self.assertIsNone(parse_enumeration_response(payload))

    def test_bad_boxes_are_rejected(self):
        for bbox in ([0.1, 0.2, 0.3], [0.3, 0.2, 0.1, 0.4], [0, 0, 1, 2], ["a", 0, 1, 1]):
            with self.subTest(bbox=bbox):
                payload = intent({"count": 1, "objects": [{"bbox": bbox}]})
                self.assertIsNone(parse_enumeration_response(payload))

    def test_extra_keys_are_rejected(self):
        payload = intent({"count": 1, "objects": [{"bbox": [0.1, 0.2, 0.3, 0.4], "confidence": 0.9}]})
        self.assertIsNone(parse_enumeration_response(payload))

    def test_an_empty_list_is_read(self):
        self.assertEqual(parse_enumeration_response(intent({"count": 0, "objects": []}))["boxes"], [])


class DirectParseTest(unittest.TestCase):
    def test_a_box_is_read(self):
        self.assertEqual(
            parse_direct_response(intent({"bbox": [0.1, 0.2, 0.3, 0.4]})),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_a_degenerate_box_is_rejected(self):
        self.assertIsNone(parse_direct_response(intent({"bbox": [0.3, 0.2, 0.1, 0.4]})))


class AxisValueTest(unittest.TestCase):
    def test_the_three_reverse_axes(self):
        box = [0.1, 0.2, 0.5, 0.8]
        self.assertEqual(axis_value("x", box), 0.1)
        self.assertEqual(axis_value("y", box), 0.2)
        self.assertAlmostEqual(axis_value("area", box), 0.24)

    def test_an_unsupported_axis_raises(self):
        for axis in ("depth", "ir", "x1", ""):
            with self.subTest(axis=axis):
                with self.assertRaises(ValueError):
                    axis_value(axis, [0.1, 0.2, 0.5, 0.8])


class PickKthTest(unittest.TestCase):
    boxes = [
        [0.50, 0.50, 0.60, 0.70],
        [0.10, 0.50, 0.30, 0.70],
        [0.40, 0.50, 0.60, 0.70],
    ]

    def test_ascending_and_descending(self):
        self.assertEqual(pick_kth(self.boxes, k=1, axis="x", direction="asc"), [0.10, 0.50, 0.30, 0.70])
        self.assertEqual(pick_kth(self.boxes, k=1, axis="x", direction="desc"), [0.50, 0.50, 0.60, 0.70])
        self.assertEqual(pick_kth(self.boxes, k=2, axis="x", direction="desc"), [0.40, 0.50, 0.60, 0.70])

    def test_k_out_of_range_returns_none(self):
        for k in (0, -1, 4, 99):
            with self.subTest(k=k):
                self.assertIsNone(pick_kth(self.boxes, k=k, axis="x", direction="asc"))

    def test_an_empty_list_does_not_crash(self):
        for k in (0, 1):
            with self.subTest(k=k):
                self.assertIsNone(pick_kth([], k=k, axis="x", direction="asc"))

    def test_ties_keep_the_box_tuple_ascending_in_both_directions(self):
        # Same y1 for both boxes. The serving side
        # (aicomp_grounding.serving.ordinal.resolve.rank_instances) breaks ties by
        # (x1, y1) ascending whatever the direction, so this must too.
        boxes = [[0.40, 0.50, 0.60, 0.70], [0.10, 0.50, 0.30, 0.70]]
        self.assertEqual(pick_kth(boxes, k=1, axis="y", direction="asc"), [0.10, 0.50, 0.30, 0.70])
        self.assertEqual(pick_kth(boxes, k=1, axis="y", direction="desc"), [0.10, 0.50, 0.30, 0.70])

    def test_the_result_is_a_copy(self):
        picked = pick_kth(self.boxes, k=1, axis="x", direction="asc")
        picked[0] = 0.99
        self.assertEqual(self.boxes[1][0], 0.10)

    def test_axes_are_the_reverse_set(self):
        self.assertEqual(AXES, ("x", "y", "area"))


class ResultJournalTest(unittest.TestCase):
    """The reverse pass journals each answer so an interrupted run resumes."""

    def _row(self, key: str) -> dict:
        return {"key": key, "bbox": [0.1, 0.2, 0.3, 0.4], "route": "kernel",
                "calls": [{"response_id": "r1"}], "thinking": ["why"]}

    def test_append_then_load_round_trips(self):
        from tools.run_reverse import _append_result, _load_results

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            self.assertEqual(_load_results(path), {})
            _append_result(path, "a", {"bbox": [0.0, 0.0, 1.0, 1.0], "route": "direct",
                                       "calls": [], "thinking": []})
            _append_result(path, "b", {"bbox": None, "route": "fallback", "calls": [], "thinking": []})
            done = _load_results(path)
            self.assertEqual(sorted(done), ["a", "b"])
            self.assertEqual(done["b"]["bbox"], None)
            self.assertNotIn("key", done["a"])

    def test_an_empty_journal_file_loads_as_nothing_done(self):
        from tools.run_reverse import _load_results

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.jsonl"
            path.write_text("\n\n", encoding="utf-8")
            self.assertEqual(_load_results(path), {})
