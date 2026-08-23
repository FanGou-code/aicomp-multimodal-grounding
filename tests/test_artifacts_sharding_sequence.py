"""Offline tests for artifact identity, global planning, and sequence annotation."""

from __future__ import annotations

import json
import unittest

from aicomp_grounding.artifacts import key_hash, require_metadata_match
from aicomp_grounding.sequence import (
    parse_frame_query_candidates,
    source_fingerprint,
    validate_annotation_query,
)
from aicomp_grounding.sharding import (
    select_scene_ids,
    shard_keys_by_scene,
    shard_scene_ids,
)


def _item(scene: str, frame: int, area: float = 0.04, query: str = "placeholder") -> dict:
    side = area ** 0.5
    return {
        "visible": f"Train/{scene}/color/{frame:08d}.png",
        "infrared": f"Train/{scene}/infrared/{frame:08d}.png",
        "depth": f"Processed/Train/{scene}/depth_jet/{frame:08d}.png",
        "bbox": [0.1, 0.1, 0.1 + side, 0.1 + side],
        "query": query,
        "width": 1920,
        "height": 1080,
    }


class ArtifactTests(unittest.TestCase):
    def test_key_hash_is_order_independent_and_rejects_duplicates(self):
        self.assertEqual(key_hash(["b", "a"]), key_hash(["a", "b"]))
        with self.assertRaises(ValueError):
            key_hash(["a", "a"])

    def test_required_metadata_must_be_present_and_equal(self):
        expected = {"version": 3, "run_id": "run-a"}
        require_metadata_match(dict(expected), expected, expected)
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            require_metadata_match({"version": 3}, expected, expected)
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            require_metadata_match(
                {"version": 3, "run_id": "run-b"}, expected, expected
            )


class ShardingTests(unittest.TestCase):
    def setUp(self):
        self.data = {
            **{f"001_{idx:08d}": _item("001", idx) for idx in range(5)},
            **{f"002_{idx:08d}": _item("002", idx) for idx in range(3)},
            **{f"003_{idx:08d}": _item("003", idx) for idx in range(2)},
        }

    def test_scene_sharding_keeps_groups_and_covers_each_key_once(self):
        shards = shard_keys_by_scene(list(self.data), self.data, 2)
        flattened = [key for shard in shards for key in shard]
        self.assertEqual(sorted(flattened), sorted(self.data))
        self.assertEqual(len(flattened), len(set(flattened)))
        for scene in ("001", "002", "003"):
            containing = [shard for shard in shards if any(key.startswith(scene) for key in shard)]
            self.assertEqual(len(containing), 1)

    def test_global_limit_is_applied_before_sharding(self):
        selected = select_scene_ids(self.data, limit=2, seed=7)
        shards = shard_scene_ids(selected, self.data, num_shards=10)
        self.assertEqual(sum(len(shard) for shard in shards), 2)
        self.assertEqual(len(shards), 2)
        self.assertEqual({scene for shard in shards for scene in shard}, set(selected))


class SequenceAnnotationTests(unittest.TestCase):
    def setUp(self):
        areas = [0.01, 0.09, 0.04, 0.02, 0.16, 0.03, 0.01, 0.04, 0.25]
        self.keys = [f"001_{idx:08d}" for idx in range(1, 10)]
        self.data = {
            key: _item("001", idx, area=areas[idx - 1])
            for idx, key in enumerate(self.keys, start=1)
        }

    def test_source_fingerprint_ignores_legacy_query_but_not_bbox(self):
        original = source_fingerprint(self.data)
        changed_query = {key: dict(item, query="closed API text") for key, item in self.data.items()}
        self.assertEqual(original, source_fingerprint(changed_query))
        changed_bbox = {key: dict(item) for key, item in self.data.items()}
        changed_bbox[self.keys[0]]["bbox"] = [0.2, 0.2, 0.4, 0.4]
        self.assertNotEqual(original, source_fingerprint(changed_bbox))

    def test_frame_query_parser(self):
        candidates = parse_frame_query_candidates(json.dumps({
            "query": "The pedestrian wearing a bright yellow waterproof jacket",
            "alternate_query": "The yellow coated person beside the metal railing",
            "uncertain": False,
        }))
        self.assertFalse(candidates["uncertain"])
        self.assertIn("yellow", candidates["query"])

    def test_intrinsic_action_and_relations_are_accepted_for_frames(self):
        queries = (
            "The tall black post with a white rectangular panel",
            "The four wheeled robot with a rectangular metal frame",
            "The rectangular appliance behind the wooden picket fence",
            "The person carrying a skateboard under one arm",
            "The yellow signboard above the fixed metal fence",
        )
        for query in queries:
            with self.subTest(query=query):
                valid, reason = validate_annotation_query(query)
                self.assertTrue(valid, reason)

    def test_frame_query_rejects_annotation_scaffolding(self):
        payload = {
            "query": "The person inside the red rectangle wearing a yellow jacket",
            "alternate_query": None,
            "uncertain": False,
        }
        with self.assertRaisesRegex(ValueError, "annotation scaffolding"):
            parse_frame_query_candidates(json.dumps(payload))
        payload["query"] = "The pedestrian wearing a yellow jacket in this frame"
        with self.assertRaisesRegex(ValueError, "annotation scaffolding"):
            parse_frame_query_candidates(json.dumps(payload))
        payload["query"] = "The target pedestrian wearing a yellow jacket"
        with self.assertRaisesRegex(ValueError, "annotation term"):
            parse_frame_query_candidates(json.dumps(payload))
        payload["query"] = "The blurry white thing held in the person's hand"
        with self.assertRaisesRegex(ValueError, "generic category"):
            parse_frame_query_candidates(json.dumps(payload))

    def test_parsers_reject_duplicate_json_keys(self):
        duplicate_top_level = (
            '{"query":"The person wearing a yellow waterproof jacket",'
            '"query":"The person wearing a blue waterproof jacket",'
            '"alternate_query":null,"uncertain":false}'
        )
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_frame_query_candidates(duplicate_top_level)


class VerificationBBoxParsingTests(unittest.TestCase):
    def test_normal_json_bbox(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertEqual(
            parse_verification_bbox('{"bbox":[0.1,0.2,0.3,0.4]}'),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_bare_array_bbox(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertEqual(
            parse_verification_bbox('[0.5,0.5,0.6,0.6]'),
            [0.5, 0.5, 0.6, 0.6],
        )

    def test_0_1000_scaled_down(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertEqual(
            parse_verification_bbox('{"bbox":[125,240,780,910]}'),
            [0.125, 0.24, 0.78, 0.91],
        )

    def test_code_fenced_json(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertEqual(
            parse_verification_bbox('```json\n{"bbox":[0.1,0.2,0.3,0.4]}\n```'),
            [0.1, 0.2, 0.3, 0.4],
        )

    def test_inverted_box_returns_none(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertIsNone(parse_verification_bbox('{"bbox":[0.5,0.5,0.3,0.3]}'))

    def test_garbage_returns_none(self):
        from aicomp_grounding.sequence import parse_verification_bbox
        self.assertIsNone(parse_verification_bbox("no coordinates at all"))


if __name__ == "__main__":
    unittest.main()
