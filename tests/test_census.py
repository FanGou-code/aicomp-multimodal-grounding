"""Offline tests for the census protocol logic (no API)."""

from __future__ import annotations

import unittest

from PIL import Image

from foundry.census import (
    attr_messages,
    draw_census_card,
    findall_messages,
    parse_attr_response,
    parse_findall_response,
    pass_agreement,
    reconcile_sequence,
    select_frames,
)

GT = [0.40, 0.40, 0.55, 0.60]


def _objects(*x1s):
    return [
        {"i": i + 1, "bbox": [x, 0.40, x + 0.08, 0.60]}
        for i, x in enumerate(x1s)
    ]


def _response(x1s, category="person"):
    import json
    return json.dumps({
        "category": category,
        "objects": [{"i": i + 1, "bbox": [x, 0.40, x + 0.08, 0.60]} for i, x in enumerate(x1s)],
    })


class FindallParseTests(unittest.TestCase):
    def test_valid_response_passes_all_gates(self):
        # 0.42 box overlaps GT 0.40-0.55 -> canary ok; sorted; no dups
        result = parse_findall_response(_response([0.10, 0.42, 0.80]), gt_bbox=GT)
        self.assertEqual(result["category"], "person")
        self.assertEqual([o["i"] for o in result["objects"]], [1, 2, 3])
        self.assertAlmostEqual(result["objects"][1]["bbox"][0], 0.42)

    def test_canary_failure_rejected(self):
        with self.assertRaisesRegex(ValueError, "canary"):
            parse_findall_response(_response([0.05, 0.70, 0.90]), gt_bbox=GT)

    def test_order_violation_rejected(self):
        with self.assertRaisesRegex(ValueError, "ordered left to right"):
            parse_findall_response(_response([0.42, 0.10, 0.80]), gt_bbox=GT)

    def test_self_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "same object twice"):
            parse_findall_response(_response([0.42, 0.421, 0.80]), gt_bbox=GT)

    def test_out_of_range_bbox_rejected(self):
        import json
        bad = json.dumps({
            "category": "person",
            "objects": [{"i": 1, "bbox": [0.42, 0.40, 1.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_nonsequential_index_rejected(self):
        import json
        bad = json.dumps({
            "category": "person",
            "objects": [{"i": 2, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "sequential"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_extra_field_rejected(self):
        import json
        bad = json.dumps({
            "category": "person",
            "extra": 1,
            "objects": [{"i": 1, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_slack_clipping_accepted(self):
        result = parse_findall_response(_response([-0.01, 0.42, 0.90]), gt_bbox=GT)
        self.assertEqual(result["objects"][0]["bbox"][0], 0.0)

    def test_pixel_convention_auto_normalized(self):
        import json
        # Teacher answered in pixels of the shown 1536x864 view; the first box
        # lands on the GT target (615/1536 ~ 0.40).
        payload = json.dumps({
            "category": "person",
            "objects": [{"i": 1, "bbox": [615, 346, 845, 518]}],
        })
        result = parse_findall_response(payload, gt_bbox=GT, image_size=(1536, 864))
        self.assertAlmostEqual(result["objects"][0]["bbox"][0], 615 / 1536, places=4)

    def test_pixel_convention_without_size_rejected(self):
        import json
        payload = json.dumps({
            "category": "person",
            "objects": [{"i": 1, "bbox": [615, 346, 845, 518]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(payload, gt_bbox=GT)

    def test_per_mille_convention_auto_detected(self):
        import json
        # GLM/Qwen family convention: 0-1000 per-mille. The teacher's deer box
        # lands on the GT target after /1000 (IoU ~0.93).
        payload = json.dumps({
            "category": "deer",
            "objects": [{"i": 1, "bbox": [400, 400, 550, 600]}],
        })
        result = parse_findall_response(payload, gt_bbox=GT, image_size=(1536, 864))
        self.assertEqual(result["bbox_convention"], "per-mille-0-1000")
        self.assertAlmostEqual(result["objects"][0]["bbox"][0], 0.572)

    def test_all_conventions_failing_reports_combined(self):
        import json
        payload = json.dumps({
            "category": "deer",
            "objects": [{"i": 1, "bbox": [10, 10, 30, 30]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(payload, gt_bbox=GT, image_size=(1536, 864))


class AttrParseTests(unittest.TestCase):
    def test_valid_attr(self):
        import json
        payload = json.dumps({"1": {"color": "red", "features": "metal frame"}})
        result = parse_attr_response(payload, indices=[1])
        self.assertEqual(result["1"]["color"], "red")

    def test_missing_index_rejected(self):
        import json
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_attr_response(json.dumps({"1": {"color": "red", "features": "x"}}), indices=[1, 2])

    def test_extra_index_rejected(self):
        import json
        payload = json.dumps({
            "1": {"color": "red", "features": "x"},
            "2": {"color": "blue", "features": "y"},
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_attr_response(payload, indices=[1])

    def test_empty_field_rejected(self):
        import json
        with self.assertRaisesRegex(ValueError, "missing or oversized"):
            parse_attr_response(json.dumps({"1": {"color": "", "features": "x"}}), indices=[1])


class AgreementTests(unittest.TestCase):
    def test_perfect_agreement(self):
        objs = _objects(0.10, 0.42, 0.80)
        result = pass_agreement(objs, _objects(0.10, 0.42, 0.80))
        self.assertTrue(result["count_agree"])
        self.assertEqual(result["matched"], 3)
        self.assertAlmostEqual(result["jaccard"], 1.0)
        self.assertEqual(len(result["agreed_objects"]), 3)

    def test_partial_agreement_gates_count(self):
        result = pass_agreement(_objects(0.10, 0.42, 0.80), _objects(0.10, 0.42))
        self.assertFalse(result["count_agree"])
        self.assertEqual(result["matched"], 2)
        self.assertAlmostEqual(result["jaccard"], 2 / 3)

    def test_disjoint_passes(self):
        result = pass_agreement(_objects(0.05), _objects(0.80))
        self.assertEqual(result["matched"], 0)
        self.assertEqual(result["jaccard"], 0.0)


class SelectionTests(unittest.TestCase):
    def _candidates(self):
        return [
            {"sample_id": f"001_{i:08d}", "frame_no": i, "count": c}
            for i, c in zip(range(1, 11), [1, 2, 3, 3, 2, 1, 4, 2, 3, 1])
        ]

    def test_counts_descending_and_spread_tiebreak(self):
        chosen = select_frames(self._candidates(), k=3)
        counts = [c["count"] for c in chosen]
        self.assertEqual(counts, [4, 3, 3])
        frames = sorted(c["frame_no"] for c in chosen)
        # frame 7 (count 4); among count-3 frames {3,4,9}: spread picks 3 and 9
        self.assertEqual(frames, [3, 7, 9])

    def test_deterministic(self):
        self.assertEqual(select_frames(self._candidates(), k=3), select_frames(self._candidates(), k=3))

    def test_fewer_than_k(self):
        chosen = select_frames([{"sample_id": "x", "frame_no": 2, "count": 1}], k=3)
        self.assertEqual(len(chosen), 1)


class ReconcileTests(unittest.TestCase):
    def test_peer_seen_in_two_frames_survives(self):
        selected = [
            _objects(0.10, 0.42),
            _objects(0.10, 0.42, 0.80),
            _objects(0.80),
        ]
        peers = reconcile_sequence(selected)
        boxes = [p["bbox"][0] for p in peers]
        # 0.10 seen in 2 frames, 0.42 in 2, 0.80 in 2 -> all survive
        self.assertEqual(len(peers), 3)

    def test_single_frame_peer_dropped(self):
        selected = [_objects(0.10), _objects(0.80), _objects(0.82)]
        peers = reconcile_sequence(selected)
        self.assertEqual(len(peers), 1)
        self.assertEqual(peers[0]["bbox"][0], 0.80)


class CardAndMessageTests(unittest.TestCase):
    def test_draw_card_returns_rgb_same_size(self):
        image = Image.new("RGB", (320, 200), "white")
        card = draw_census_card(image, _objects(0.10, 0.42), gt_bbox=GT)
        self.assertEqual(card.size, (320, 200))
        self.assertEqual(card.mode, "RGB")
        flattened = list(card.get_flattened_data())
        self.assertTrue(any(pixel == (255, 0, 0) for pixel in flattened))

    def test_messages_carry_exactly_one_image_and_prompt(self):
        messages = findall_messages("data:image/jpeg;base64,xxx")
        content = messages[1]["content"]
        self.assertEqual(sum(part["type"] == "image_url" for part in content), 1)
        self.assertIn("red rectangle", "".join(part.get("text", "") for part in content))
        attr = attr_messages("data:image/jpeg;base64,xxx")
        self.assertEqual(sum(part["type"] == "image_url" for part in attr[1]["content"]), 1)
        self.assertIn("numbered boxes", "".join(part.get("text", "") for part in attr[1]["content"]))



if __name__ == "__main__":
    unittest.main()
