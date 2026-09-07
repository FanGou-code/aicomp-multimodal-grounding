"""Offline tests for the census protocol logic (no API)."""

from __future__ import annotations

import unittest

from PIL import Image

from foundry.pipeline.census import (
    attr_messages,
    findall_messages,
    parse_attr_response,
    parse_findall_response,
    pass_agreement,
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
        "objects": [
            {"i": i + 1, "category": category, "bbox": [x, 0.40, x + 0.08, 0.60]}
            for i, x in enumerate(x1s)
        ],
    })


class FindallParseTests(unittest.TestCase):
    def test_valid_response_passes_all_gates(self):
        # 0.42 box overlaps GT 0.40-0.55 -> canary ok; sorted; no dups
        result = parse_findall_response(_response([0.10, 0.42, 0.80]), gt_bbox=GT)
        self.assertEqual([o["category"] for o in result["objects"]], ["person"] * 3)
        self.assertEqual([o["i"] for o in result["objects"]], [1, 2, 3])
        self.assertAlmostEqual(result["objects"][1]["bbox"][0], 0.42)

    def test_cross_category_enumeration_accepted(self):
        import json
        payload = json.dumps({
            "objects": [
                {"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 2, "category": "bench", "bbox": [0.60, 0.40, 0.68, 0.60]},
                {"i": 3, "category": "traffic sign", "bbox": [0.80, 0.40, 0.88, 0.60]},
            ],
        })
        result = parse_findall_response(payload, gt_bbox=GT)
        self.assertEqual(
            [o["category"] for o in result["objects"]],
            ["person", "bench", "traffic sign"],
        )
        self.assertNotIn("category", result)

    def test_missing_category_rejected(self):
        import json
        bad = json.dumps({
            "objects": [{"i": 1, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "exactly \\{i, category, bbox\\}"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_blank_or_oversized_category_rejected(self):
        import json
        blank = json.dumps({
            "objects": [{"i": 1, "category": "   ", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "category is missing"):
            parse_findall_response(blank, gt_bbox=GT)
        oversized = json.dumps({
            "objects": [{"i": 1, "category": "x" * 65, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "category is missing"):
            parse_findall_response(oversized, gt_bbox=GT)

    def test_canary_failure_rejected(self):
        with self.assertRaisesRegex(ValueError, "canary"):
            parse_findall_response(_response([0.05, 0.70, 0.90]), gt_bbox=GT)

    def test_unordered_response_is_sorted_and_renumbered(self):
        import json
        # Panoramic enumeration may arrive out of left->right order; the
        # parser normalizes instead of rejecting (fatal ordering killed 27
        # pilot frames). 0.42 box overlaps GT -> canary ok.
        payload = json.dumps({
            "objects": [
                {"i": 1, "category": "bench", "bbox": [0.80, 0.40, 0.88, 0.60]},
                {"i": 2, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 3, "category": "sign", "bbox": [0.10, 0.40, 0.18, 0.60]},
            ],
        })
        result = parse_findall_response(payload, gt_bbox=GT)
        self.assertEqual([o["i"] for o in result["objects"]], [1, 2, 3])
        self.assertEqual([o["bbox"][0] for o in result["objects"]], [0.10, 0.42, 0.80])
        self.assertEqual([o["category"] for o in result["objects"]], ["sign", "person", "bench"])

    def test_duplicate_box_is_deduplicated_not_fatal(self):
        import json
        # Same object listed twice (two boxes with IoU ~0.98): the parser
        # drops the later duplicate instead of killing the frame.
        payload = json.dumps({
            "objects": [
                {"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 2, "category": "person", "bbox": [0.421, 0.40, 0.501, 0.60]},
                {"i": 3, "category": "sign", "bbox": [0.80, 0.40, 0.88, 0.60]},
            ],
        })
        result = parse_findall_response(payload, gt_bbox=GT)
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual([o["bbox"][0] for o in result["objects"]], [0.42, 0.80])

    def test_degenerate_box_dropped(self):
        import json
        # Zero-width box is unusable under every convention: dropped, the
        # rest of the response survives (canary box stays).
        payload = json.dumps({
            "objects": [
                {"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 2, "category": "sign", "bbox": [0.80, 0.40, 0.80, 0.60]},
            ],
        })
        result = parse_findall_response(payload, gt_bbox=GT)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["objects"][0]["category"], "person")

    def test_all_degenerate_rejected(self):
        import json
        payload = json.dumps({
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.42, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "no non-degenerate"):
            parse_findall_response(payload, gt_bbox=GT)

    def test_out_of_range_bbox_rejected(self):
        import json
        bad = json.dumps({
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 1.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_nonsequential_index_rejected(self):
        import json
        bad = json.dumps({
            "objects": [{"i": 2, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "sequential"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_extra_field_rejected(self):
        import json
        bad = json.dumps({
            "extra": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_findall_response(bad, gt_bbox=GT)

    def test_object_entry_extra_field_rejected(self):
        import json
        bad = json.dumps({
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60], "j": 2}],
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
            "objects": [{"i": 1, "category": "person", "bbox": [615, 346, 845, 518]}],
        })
        result = parse_findall_response(payload, gt_bbox=GT, image_size=(1536, 864))
        self.assertAlmostEqual(result["objects"][0]["bbox"][0], 615 / 1536, places=4)

    def test_pixel_convention_without_size_rejected(self):
        import json
        payload = json.dumps({
            "objects": [{"i": 1, "category": "person", "bbox": [615, 346, 845, 518]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(payload, gt_bbox=GT)

    def test_per_mille_convention_auto_detected(self):
        import json
        # GLM/Qwen family convention: 0-1000 per-mille. The teacher's deer box
        # lands on the GT target after /1000 (IoU ~0.93).
        payload = json.dumps({
            "objects": [{"i": 1, "category": "deer", "bbox": [400, 400, 550, 600]}],
        })
        result = parse_findall_response(payload, gt_bbox=GT, image_size=(1536, 864))
        self.assertEqual(result["bbox_convention"], "per-mille-0-1000")
        self.assertAlmostEqual(result["objects"][0]["bbox"][0], 0.400)

    def test_all_conventions_failing_reports_combined(self):
        import json
        payload = json.dumps({
            "objects": [{"i": 1, "category": "deer", "bbox": [10, 10, 30, 30]}],
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


class CardAndMessageTests(unittest.TestCase):

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


class ProtocolDocSyncTest(unittest.TestCase):
    """spec/census_protocol.md must carry the frozen prompts verbatim.

    Machine check for the doc/code dual source: the doc quotes both prompt
    constants and pins their SHA-256; any prompt edit must update the doc
    (and thus this test) in the same commit.
    """

    def test_doc_quotes_prompts_and_fingerprints(self):
        from pathlib import Path

        from foundry.pipeline.census import ATTR_PROMPT, FINDALL_PROMPT

        root = Path(__file__).resolve().parents[1]
        findall_doc = (root / "configs" / "default" / "prompts" / "findall.md").read_text(encoding="utf-8")
        attr_doc = (root / "configs" / "default" / "prompts" / "attr.md").read_text(encoding="utf-8")
        self.assertEqual(FINDALL_PROMPT, findall_doc.strip(), "code FINDALL_PROMPT drifts from config")
        self.assertEqual(ATTR_PROMPT, attr_doc.strip(), "code ATTR_PROMPT drifts from config")


if __name__ == "__main__":
    unittest.main()
