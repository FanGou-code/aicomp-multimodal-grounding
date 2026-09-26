"""Offline tests for artifact identity, global planning, and sequence QC."""

from __future__ import annotations

import unittest

from aicomp_grounding.artifacts import key_hash, require_metadata_match
from aicomp_grounding.contract import source_fingerprint
from aicomp_grounding.query import validate_annotation_query
from aicomp_grounding.sharding import (
    select_scene_ids,
    shard_keys_by_scene,
    shard_scene_ids,
)


def _item(scene: str, frame: int, area: float = 0.04, query: str = "placeholder") -> dict:
    side = area ** 0.5
    return {
        "visible": f"Raw/{scene}/color/{frame:08d}.png",
        "infrared": f"Raw/{scene}/infrared/{frame:08d}.png",
        "depth": f"Raw/{scene}/depth/{frame:08d}.png",
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

    def test_annotation_scaffolding_is_rejected(self):
        """Annotation-speak must stay rejected.

        The scaffold word list is data, not code: nothing else in the suite
        notices if it is emptied or a phrase is dropped, so pin the rejection
        here as well.
        """
        queries = (
            "The highlighted person on the left",
            "The person in the current frame",
            "The object in the visible image",
        )
        for query in queries:
            with self.subTest(query=query):
                valid, reason = validate_annotation_query(query)
                self.assertFalse(valid)
                self.assertIn("scaffolding", reason)





if __name__ == "__main__":
    unittest.main()
