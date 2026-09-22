"""Contract tests for the ordinal resolver (pure CPU, no weights)."""

from __future__ import annotations

import unittest

import numpy as np

from aicomp_grounding.ordinal.resolve import (
    Decision,
    axis_value,
    coerce_instances,
    parse_selection,
    rank_instances,
    reconcile_runs,
    resolve_query,
    truncation_reason,
)

I1 = [0.10, 0.10, 0.20, 0.30]
I2 = [0.40, 0.10, 0.50, 0.30]
I3 = [0.70, 0.10, 0.80, 0.30]


def run(*boxes: list[float]) -> list[dict]:
    return [{"bbox": list(box), "confidence": 0.9} for box in boxes]


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


class ReconcileRunsTests(unittest.TestCase):
    def test_identical_lists_reconcile(self):
        merged, reason = reconcile_runs(coerce_instances(run(I1, I2, I3)), coerce_instances(run(I1, I2, I3)))
        self.assertEqual(reason, "")
        self.assertEqual([item.bbox for item in merged], [I1, I2, I3])

    def test_order_does_not_matter(self):
        merged, reason = reconcile_runs(coerce_instances(run(I3, I1, I2)), coerce_instances(run(I2, I3, I1)))
        self.assertEqual(reason, "")
        self.assertEqual([item.bbox for item in merged], [I1, I2, I3])

    def test_missing_or_extra_instance_fails_the_gate(self):
        for other in (run(I1, I2), run(I1, I2, I3, [0.85, 0.10, 0.95, 0.30])):
            with self.subTest(other=other):
                merged, reason = reconcile_runs(coerce_instances(run(I1, I2, I3)), coerce_instances(other))
                self.assertIsNone(merged)
                self.assertEqual(reason, "runs-disagree")

    def test_empty_run_is_rejected(self):
        self.assertEqual(reconcile_runs([], coerce_instances(run(I1))), (None, "empty-run"))


class TruncationTests(unittest.TestCase):
    def test_matching_counts_pass(self):
        lists = [coerce_instances(run(I1, I2)), coerce_instances(run(I1, I2))]
        self.assertEqual(truncation_reason(lists, [2, 2]), "")

    def test_mismatched_or_missing_counts_fail(self):
        lists = [coerce_instances(run(I1, I2)), coerce_instances(run(I1, I2))]
        self.assertEqual(truncation_reason(lists, [3, 2]), "truncated")
        self.assertEqual(truncation_reason(lists, [2]), "count-missing")
        self.assertEqual(truncation_reason(lists, [2, None]), "count-missing")


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
        decision = resolve_query(I1, rank_payload(2), [run(I1, I2, I3), run(I1, I2, I3)], [3, 3])
        self.assertEqual(decision.action, "replace")
        self.assertEqual(decision.reason, "replaced")
        self.assertEqual(decision.bbox, I2)

    def test_keeps_everything_when_the_kth_is_already_the_base_box(self):
        decision = resolve_query(I1, rank_payload(1), [run(I1, I2, I3), run(I1, I2, I3)], [3, 3])
        self.assertEqual((decision.action, decision.bbox, decision.reason), ("keep", None, "already-correct"))

    def test_single_instance_frame_is_never_rewritten(self):
        # k=1..N all resolve to the only instance, which is the base box
        for k in (1,):
            with self.subTest(k=k):
                decision = resolve_query(I1, rank_payload(k), [run(I1), run(I1)], [1, 1])
                self.assertEqual(decision.bbox, None)

    def test_every_gate_failure_keeps_the_base_box(self):
        cases = [
            ("parse-invalid", I1, {"category": "", "selection": {"mode": "unique"}}, [run(I1), run(I1)], [1, 1]),
            ("not-rank", I1, {"category": "person", "selection": {"mode": "unique"}}, [run(I1), run(I1)], [1, 1]),
            ("run-malformed", I1, rank_payload(1), [[{"bbox": [0.5, 0.5]}], run(I1)], [1, 1]),
            ("truncated", I1, rank_payload(1), [run(I1, I2), run(I1, I2)], [1, 2]),
            ("runs-disagree", I1, rank_payload(1), [run(I1, I2), run(I1)], [2, 1]),
            ("k-out-of-range", I1, rank_payload(4), [run(I1, I2), run(I1, I2)], [2, 2]),
            ("axis-unsupported", I1, rank_payload(1, axis="depth"), [run(I1, I2), run(I1, I2)], [2, 2]),
        ]
        for reason, base, payload, runs, counts in cases:
            with self.subTest(reason=reason):
                decision = resolve_query(base, payload, runs, counts)
                self.assertEqual(decision.action, "keep", reason)
                self.assertIsNone(decision.bbox, reason)
                self.assertEqual(decision.reason, reason)

    def test_unusable_base_box_is_replaced_by_the_kth_instance(self):
        decision = resolve_query([0.5, 0.5, 0.4, 0.6], rank_payload(1), [run(I1, I2), run(I1, I2)], [2, 2])
        self.assertEqual((decision.action, decision.bbox), ("replace", I1))

    def test_base_box_absent_from_the_list_is_still_replaced(self):
        decision = resolve_query([0.85, 0.10, 0.95, 0.30], rank_payload(1), [run(I1, I2), run(I1, I2)], [2, 2])
        self.assertEqual((decision.action, decision.bbox), ("replace", I1))

    def test_depth_axis_can_reverse_the_x_order(self):
        depth = np.array([[100 * (10 - x) for x in range(10)] for _ in range(10)], dtype=np.uint16)
        decision = resolve_query(
            I1, rank_payload(1, axis="depth", direction="asc"),
            [run(I1, I2, I3), run(I1, I2, I3)], [3, 3],
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
