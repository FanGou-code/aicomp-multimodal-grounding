"""Offline tests for the census protocol logic (no API)."""

from __future__ import annotations

import json
import unittest

from PIL import Image

from aicomp_grounding.annotation.census import (
    attr_messages,
    findall_messages,
    parse_attr_response,
    parse_findall_response,
    select_frames,
    trusted_objects,
)

GT = [0.40, 0.40, 0.55, 0.60]


def _objects(*x1s):
    return [
        {"i": i + 1, "bbox": [x, 0.40, x + 0.08, 0.60]}
        for i, x in enumerate(x1s)
    ]


def _response(x1s, category="person", mode="instances", count=None):
    """One enumeration reply: mode, the reference category, and a count."""
    return json.dumps({
        "mode": mode,
        "category": category,
        "count": len(x1s) if count is None else count,
        "objects": [
            {"i": i + 1, "category": category, "bbox": [x, 0.40, x + 0.08, 0.60]}
            for i, x in enumerate(x1s)
        ],
    })


class FindallParseTests(unittest.TestCase):
    def test_valid_response_passes_all_gates(self):
        result = parse_findall_response(_response([0.10, 0.42, 0.80]))
        self.assertEqual(result["mode"], "instances")
        self.assertEqual(result["category"], "person")
        self.assertEqual([o["i"] for o in result["objects"]], [1, 2, 3])
        self.assertAlmostEqual(result["objects"][1]["bbox"][0], 0.42)

    def test_other_branch_keeps_the_reference_category(self):
        payload = json.dumps({
            "mode": "other",
            "category": "person",
            "count": 2,
            "objects": [
                {"i": 1, "category": "bench", "bbox": [0.60, 0.40, 0.68, 0.60]},
                {"i": 2, "category": "traffic sign", "bbox": [0.80, 0.40, 0.88, 0.60]},
            ],
        })
        result = parse_findall_response(payload)
        self.assertEqual(result["mode"], "other")
        self.assertEqual(result["category"], "person")
        self.assertEqual([o["category"] for o in result["objects"]], ["bench", "traffic sign"])

    def test_count_mismatch_is_the_truncation_gate(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            parse_findall_response(_response([0.10, 0.42], count=7))

    def test_unknown_mode_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be"):
            parse_findall_response(_response([0.42], mode="whatever"))

    def test_missing_category_rejected(self):
        bad = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 1, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "exactly \\{i, category, bbox\\}"):
            parse_findall_response(bad)

    def test_blank_or_oversized_category_rejected(self):
        blank = json.dumps({
            "mode": "instances", "category": "   ", "count": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "category is missing"):
            parse_findall_response(blank)
        oversized = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 1, "category": "x" * 65, "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "category is missing"):
            parse_findall_response(oversized)

    def test_the_reference_box_need_not_be_re_found(self):
        # The red box is a reference, not an authority: an enumeration that
        # misses it is still a valid frame.
        result = parse_findall_response(_response([0.05, 0.70, 0.90]))
        self.assertEqual(len(result["objects"]), 3)

    def test_unordered_response_is_sorted_and_renumbered(self):
        payload = json.dumps({
            "mode": "instances", "category": "person", "count": 3,
            "objects": [
                {"i": 1, "category": "bench", "bbox": [0.80, 0.40, 0.88, 0.60]},
                {"i": 2, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 3, "category": "sign", "bbox": [0.10, 0.40, 0.18, 0.60]},
            ],
        })
        result = parse_findall_response(payload)
        self.assertEqual([o["bbox"][0] for o in result["objects"]], [0.10, 0.42, 0.80])
        self.assertEqual([o["category"] for o in result["objects"]], ["sign", "person", "bench"])

    def test_duplicate_box_is_deduplicated_not_fatal(self):
        payload = json.dumps({
            "mode": "instances", "category": "person", "count": 3,
            "objects": [
                {"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 2, "category": "person", "bbox": [0.421, 0.40, 0.501, 0.60]},
                {"i": 3, "category": "sign", "bbox": [0.80, 0.40, 0.88, 0.60]},
            ],
        })
        result = parse_findall_response(payload)
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual([o["bbox"][0] for o in result["objects"]], [0.42, 0.80])

    def test_degenerate_box_dropped(self):
        payload = json.dumps({
            "mode": "instances", "category": "person", "count": 2,
            "objects": [
                {"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]},
                {"i": 2, "category": "sign", "bbox": [0.80, 0.40, 0.80, 0.60]},
            ],
        })
        result = parse_findall_response(payload)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["objects"][0]["category"], "person")

    def test_all_degenerate_rejected(self):
        payload = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.42, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "no non-degenerate"):
            parse_findall_response(payload)

    def test_out_of_range_bbox_rejected(self):
        # Above 1000 (per-mille) and above 1 (normalized): fails both.
        bad = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 1500, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "every coordinate convention"):
            parse_findall_response(bad)

    def test_nonsequential_index_rejected(self):
        bad = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 2, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "sequential"):
            parse_findall_response(bad)

    def test_extra_field_rejected(self):
        bad = json.dumps({
            "mode": "instances", "category": "person", "count": 1, "extra": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60]}],
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_findall_response(bad)

    def test_object_entry_extra_field_rejected(self):
        bad = json.dumps({
            "mode": "instances", "category": "person", "count": 1,
            "objects": [{"i": 1, "category": "person", "bbox": [0.42, 0.40, 0.50, 0.60], "j": 2}],
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_findall_response(bad)

    def test_slack_clipping_accepted(self):
        result = parse_findall_response(_response([-0.01, 0.42, 0.90]))
        self.assertEqual(result["objects"][0]["bbox"][0], 0.0)

    def test_per_mille_convention_auto_detected(self):
        payload = json.dumps({
            "mode": "instances", "category": "deer", "count": 1,
            "objects": [{"i": 1, "category": "deer", "bbox": [400, 400, 550, 600]}],
        })
        result = parse_findall_response(payload)
        self.assertEqual(result["bbox_convention"], "per-mille-0-1000")
        self.assertAlmostEqual(result["objects"][0]["bbox"][0], 0.400)


class TrustedObjectsTests(unittest.TestCase):
    def test_the_single_enumeration_is_the_trusted_set(self):
        frame = {"findall": {"status": "completed", "objects": _objects(0.1, 0.4)}}
        self.assertEqual(trusted_objects(frame), _objects(0.1, 0.4))


class AttrParseTests(unittest.TestCase):
    def test_valid_attr(self):
        payload = json.dumps({"1": {"color": "red", "features": "metal frame"}})
        result = parse_attr_response(payload, indices=[1])
        self.assertEqual(result["1"]["color"], "red")

    def test_missing_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_attr_response(json.dumps({"1": {"color": "red", "features": "x"}}), indices=[1, 2])

    def test_extra_index_rejected(self):
        payload = json.dumps({
            "1": {"color": "red", "features": "x"},
            "2": {"color": "blue", "features": "y"},
        })
        with self.assertRaisesRegex(ValueError, "exactly"):
            parse_attr_response(payload, indices=[1])

    def test_empty_field_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing or oversized"):
            parse_attr_response(json.dumps({"1": {"color": "", "features": "x"}}), indices=[1])


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
        self.assertEqual(frames, [3, 7, 9])

    def test_selection_order_is_stable(self):
        chosen = select_frames(self._candidates(), k=3)
        self.assertEqual([c["frame_no"] for c in chosen], [7, 3, 9])

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


class ProtocolDocSyncTest(unittest.TestCase):
    """configs/default/prompts/ must carry the prompts verbatim.

    Machine check for the doc/code dual source: the prompt constants are read
    from those files, so any prompt edit lands in both or in neither.
    """

    def test_doc_quotes_prompts(self):
        from pathlib import Path

        from aicomp_grounding.annotation.census import ATTR_PROMPT, FINDALL_PROMPT

        root = Path(__file__).resolve().parents[1]
        findall_doc = (root / "configs" / "default" / "prompts" / "findall.md").read_text(encoding="utf-8")
        attr_doc = (root / "configs" / "default" / "prompts" / "attr.md").read_text(encoding="utf-8")
        self.assertEqual(FINDALL_PROMPT, findall_doc.strip(), "code FINDALL_PROMPT drifts from config")
        self.assertEqual(ATTR_PROMPT, attr_doc.strip(), "code ATTR_PROMPT drifts from config")


class CensusRecoveryTests(unittest.TestCase):
    def test_completed_frames_and_attributes_survive_two_interruptions(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tools import run_census as census
        from aicomp_grounding.annotation.client import APIError
        dataset = {f"001_{i:08d}": {"visible": f"Train/001/color/{i:08d}.png", "bbox": GT}
                   for i in range(1, 4)}
        plan = {"metadata": {"run_id": "census_recovery", "split": "train", "model_name": "fake",
                              "api_base_url": "https://example.invalid", "selected_sequence_ids": ["001"]}}
        calls = []

        def successful_pass(client, image, *, frame_ref):
            calls.append(frame_ref)
            return {"status": "completed", "mode": "instances", "objects": [
                {"i": 1, "category": "person", "bbox": list(GT)}],
                "bbox_convention": "normalized-0-1", "attempts": 1, "api_calls": []}

        def interrupted_pass(*args, **kwargs):
            if len(calls) == 2:
                raise APIError("simulated stop")
            return successful_pass(*args, **kwargs)

        complete_attr = {"status": "completed", "attributes": {}, "api_calls": []}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(census, "load_annotation_source", return_value=dataset), \
             patch.object(census, "_load_plain_frame", return_value=Image.new("RGB", (64, 64))), \
             patch.object(census, "raw_depth_path", return_value=Path(tmp) / "missing.png"), \
             contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            args = dict(shard_id=0, sequence_ids=["001"], plan=plan, data_root=root,
                        output_root=root, api_key="fake", resume=True,
                        retry_failed=True, timeout_seconds=1, retry=True, progress=None)
            with patch.object(census, "_run_findall_pass", side_effect=interrupted_pass):
                with self.assertRaises(APIError):
                    census.census_shard(**args)
            checkpoint = root / "census_recovery/shards/shard_00.json"
            saved = json.loads(checkpoint.read_text())["results"]["001"]
            self.assertEqual(len(saved["frames"]), 2)
            self.assertNotEqual(saved["status"], "completed")
            calls.clear()
            with patch.object(census, "_run_findall_pass", side_effect=successful_pass), \
                 patch.object(census, "_complete_once", side_effect=[complete_attr, APIError("attr stop")]):
                with self.assertRaises(APIError):
                    census.census_shard(**args)
            self.assertEqual(calls, ["001_00000003"])
            saved = json.loads(checkpoint.read_text())["results"]["001"]
            self.assertEqual(sum(f.get("attr", {}).get("status") == "completed" for f in saved["frames"].values()), 1)
            with patch.object(census, "_run_findall_pass") as findall, \
                 patch.object(census, "_complete_once", return_value=complete_attr) as attrs:
                result = census.census_shard(**args)
            findall.assert_not_called()
            self.assertEqual(attrs.call_count, 2)
            self.assertEqual(result["results"]["001"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
