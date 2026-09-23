"""Contract tests for the ordinal resolver (pure CPU, no weights)."""

from __future__ import annotations

import unittest

import numpy as np

from aicomp_grounding.ordinal.parse import split_thinking, strict_json_object
from aicomp_grounding.ordinal.resolve import (
    Decision,
    axis_value,
    coerce_instances,
    parse_selection,
    rank_instances,
    resolve_query,
)

I1 = [0.10, 0.10, 0.20, 0.30]
I2 = [0.40, 0.10, 0.50, 0.30]
I3 = [0.70, 0.10, 0.80, 0.30]


def run(*boxes: list[float]) -> list[dict]:
    return [{"bbox": list(box), "confidence": 0.9} for box in boxes]


def enumeration(*boxes: list[float], count: int | None = None) -> dict:
    return {"count": len(boxes) if count is None else count, "instances": run(*boxes)}


def rank_payload(k: int, axis: str = "x", direction: str = "asc") -> dict:
    return {"category": "person", "selection": {"mode": "rank", "k": k, "axis": axis, "direction": direction}}


class ParseSelectionTests(unittest.TestCase):
    def test_rank_and_unique_payloads_are_accepted(self):
        rank = parse_selection(rank_payload(3))
        self.assertEqual((rank.category, rank.mode, rank.k, rank.axis, rank.direction), ("person", "rank", 3, "x", "asc"))
        unique = parse_selection({"category": "car", "selection": {"mode": "unique"}})
        self.assertEqual((unique.category, unique.mode, unique.k), ("car", "unique", None))

    def test_invalid_payloads_are_rejected(self):
        bad = [
            None,
            {},
            {"category": " ", "selection": {"mode": "unique"}},
            {"category": "car", "selection": {"mode": "maybe"}},
            {"category": "car", "selection": {"mode": "rank", "k": 0, "axis": "x", "direction": "asc"}},
            {"category": "car", "selection": {"mode": "rank", "k": True, "axis": "x", "direction": "asc"}},
            {"category": "car", "selection": {"mode": "rank", "k": 1, "axis": "radius", "direction": "asc"}},
            {"category": "car", "selection": {"mode": "rank", "k": 1, "axis": "x", "direction": "up"}},
        ]
        for payload in bad:
            with self.subTest(payload=payload):
                self.assertIsNone(parse_selection(payload))

    def test_category_is_stripped(self):
        self.assertEqual(parse_selection({"category": "  person ", "selection": {"mode": "unique"}}).category, "person")


class CoerceInstanceTests(unittest.TestCase):
    def test_malformed_entries_reject_the_whole_run(self):
        self.assertIsNotNone(coerce_instances(run(I1, I2)))
        self.assertIsNone(coerce_instances(run(I1) + [{"bbox": [0.5, 0.5]}]))
        self.assertIsNone(coerce_instances("not-a-list"))
        self.assertIsNone(coerce_instances([{"bbox": [0.5, 0.5, 0.4, 0.6]}]))


class SplitThinkingTests(unittest.TestCase):
    def test_a_reply_without_thinking_passes_through(self):
        self.assertEqual(split_thinking('{"count": 1}'), (None, '{"count": 1}'))

    def test_the_block_is_removed_from_the_answer(self):
        thinking, answer = split_thinking("<think>left to right</think>{\"count\": 1}")
        self.assertEqual(thinking, "left to right")
        self.assertEqual(answer, '{"count": 1}')

    def test_an_unclosed_block_leaves_no_answer(self):
        thinking, answer = split_thinking("<think>I count 3")
        self.assertEqual(thinking, "I count 3")
        self.assertEqual(answer, "")

    def test_non_strings_are_not_thinking(self):
        self.assertEqual(split_thinking(None), (None, ""))

    def test_chatter_before_the_block_does_not_spoil_the_answer(self):
        thinking, answer = split_thinking(
            'Certainly! <think>two cars</think>```json\n{"count": 2, "instances": []}\n```'
        )
        self.assertEqual(thinking, "two cars")
        self.assertEqual(strict_json_object(answer), {"count": 2, "instances": []})

    def test_text_before_the_block_is_used_when_nothing_follows(self):
        _thinking, answer = split_thinking('{"count": 1} <think>one car</think>')
        self.assertEqual(strict_json_object(answer), {"count": 1})


class AxisValueTests(unittest.TestCase):
    def setUp(self):
        # depth grows to the left (value = 100 * (10 - x)); infrared grows to the right
        self.depth = np.array([[100 * (10 - x) for x in range(10)] for _ in range(10)], dtype=np.uint16)
        self.ir = np.array([[10 * x for x in range(10)] for _ in range(10)], dtype=np.uint8)

    def test_box_axes(self):
        self.assertAlmostEqual(axis_value("x", I1), 0.10)
        self.assertAlmostEqual(axis_value("y", I1), 0.10)
        self.assertAlmostEqual(axis_value("area", I1), 0.10 * 0.20)

    def test_depth_uses_the_median_of_valid_pixels(self):
        left = axis_value("depth", I1, depth_mm=self.depth, image_size=(10, 10))
        right = axis_value("depth", I3, depth_mm=self.depth, image_size=(10, 10))
        self.assertIsNotNone(left)
        self.assertGreater(left, right)

    def test_depth_ignores_invalid_zero_pixels_and_reports_none_when_all_invalid(self):
        empty = np.zeros((10, 10), dtype=np.uint16)
        self.assertIsNone(axis_value("depth", I1, depth_mm=empty, image_size=(10, 10)))

    def test_ir_reads_the_first_channel(self):
        self.assertIsNotNone(axis_value("ir", I3, ir=self.ir, image_size=(10, 10)))
        self.assertGreater(
            axis_value("ir", I3, ir=self.ir, image_size=(10, 10)),
            axis_value("ir", I1, ir=self.ir, image_size=(10, 10)),
        )

    def test_missing_data_or_wrong_size_is_unsupported(self):
        self.assertIsNone(axis_value("depth", I1))
        self.assertIsNone(axis_value("ir", I1))
        self.assertIsNone(axis_value("depth", I1, depth_mm=self.depth, image_size=(20, 20)))
        self.assertIsNone(axis_value("radius", I1))


class RankInstancesTests(unittest.TestCase):
    def test_direction_orders_both_ways(self):
        items = coerce_instances(run(I1, I2, I3))
        asc = rank_instances(items, "x", "asc")
        desc = rank_instances(items, "x", "desc")
        self.assertEqual([item.bbox for item in asc], [I1, I2, I3])
        self.assertEqual([item.bbox for item in desc], [I3, I2, I1])

    def test_ties_fall_back_to_the_box_tuple_regardless_of_direction(self):
        low = [0.10, 0.50, 0.20, 0.70]
        high = [0.10, 0.10, 0.20, 0.30]
        items = coerce_instances(run(low, high))
        for direction in ("asc", "desc"):
            with self.subTest(direction=direction):
                ordered = rank_instances(items, "x", direction)
                self.assertEqual([item.bbox for item in ordered], [high, low])

    def test_unsupported_axis_returns_none(self):
        items = coerce_instances(run(I1, I2))
        self.assertIsNone(rank_instances(items, "depth", "asc"))


class ResolveQueryTests(unittest.TestCase):
    def test_replaces_the_base_box_with_the_kth_instance(self):
        decision = resolve_query(I1, rank_payload(2), enumeration(I1, I2, I3))
        self.assertEqual(decision.action, "replace")
        self.assertEqual(decision.reason, "replaced")
        self.assertEqual(decision.bbox, I2)

    def test_keeps_everything_when_the_kth_is_already_the_base_box(self):
        decision = resolve_query(I1, rank_payload(1), enumeration(I1, I2, I3))
        self.assertEqual((decision.action, decision.bbox, decision.reason), ("keep", None, "already-correct"))

    def test_single_instance_frame_is_never_rewritten(self):
        decision = resolve_query(I1, rank_payload(1), enumeration(I1))
        self.assertEqual(decision.bbox, None)

    def test_every_gate_failure_keeps_the_base_box(self):
        cases = [
            ("parse-invalid", I1, {"category": "", "selection": {"mode": "unique"}}, enumeration(I1), None),
            ("not-rank", I1, {"category": "person", "selection": {"mode": "unique"}}, enumeration(I1), None),
            ("run-malformed", I1, rank_payload(1), {"count": 1, "instances": [{"bbox": [0.5, 0.5]}]}, None),
            ("run-malformed", I1, rank_payload(1), "not-a-payload", None),
            ("count-missing", I1, rank_payload(1), {"instances": run(I1)}, None),
            ("truncated", I1, rank_payload(1), enumeration(I1, I2, count=1), None),
            ("count-zero", I1, rank_payload(1), enumeration(count=0), None),
            ("k-out-of-range", I1, rank_payload(4), enumeration(I1, I2), None),
            ("axis-unsupported", I1, rank_payload(1, axis="depth"), enumeration(I1, I2), None),
        ]
        for reason, base, payload, enum_payload, _thinking in cases:
            with self.subTest(reason=reason):
                decision = resolve_query(base, payload, enum_payload)
                self.assertEqual(decision.action, "keep", reason)
                self.assertIsNone(decision.bbox, reason)
                self.assertEqual(decision.reason, reason)

    def test_a_self_corrected_thinking_pass_does_not_change_the_decision(self):
        # The gate that read numbers out of the thinking block is gone: the
        # model's reasoning is never used to veto its own enumeration.
        decision = resolve_query(I1, rank_payload(2), enumeration(I1, I2))
        self.assertEqual(decision.reason, "replaced")

    def test_unusable_base_box_is_replaced_by_the_kth_instance(self):
        decision = resolve_query([0.5, 0.5, 0.4, 0.6], rank_payload(1), enumeration(I1, I2))
        self.assertEqual((decision.action, decision.bbox), ("replace", I1))

    def test_base_box_absent_from_the_list_is_still_replaced(self):
        decision = resolve_query([0.85, 0.10, 0.95, 0.30], rank_payload(1), enumeration(I1, I2))
        self.assertEqual((decision.action, decision.bbox), ("replace", I1))

    def test_depth_axis_can_reverse_the_x_order(self):
        depth = np.array([[100 * (10 - x) for x in range(10)] for _ in range(10)], dtype=np.uint16)
        decision = resolve_query(
            I1, rank_payload(1, axis="depth", direction="asc"),
            enumeration(I1, I2, I3),
            depth_mm=depth, image_size=(10, 10),
        )
        self.assertEqual(decision.action, "replace")
        self.assertEqual(decision.bbox, I3)

    def test_decision_dataclass_is_frozen(self):
        decision = Decision("keep", None, "not-rank")
        with self.assertRaises(Exception):
            decision.action = "replace"


if __name__ == "__main__":
    unittest.main()
