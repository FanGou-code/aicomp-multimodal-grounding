import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_review import apply, _detect_collisions


class ApplyReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_assembly(self, records, metadata=None):
        meta = {"run_tag": "asm-test-r5", "split": "train", "census_run_id": "census_xxx",
                "all_frames": True, "max_teacher_per_frame": -1, "word_window": [3, 18]}
        if metadata:
            meta.update(metadata)
        path = self.tmp_path / "assembly.json"
        path.write_text(json.dumps({"metadata": meta, "records": records, "shortfall": []}))
        return path

    def _write_queries(self, queries):
        path = self.tmp_path / "annotations.queries.json"
        path.write_text(json.dumps(queries))
        return path

    def test_human_query_overrides_original(self):
        records = [{
            "sample_id": "001_00000001", "sequence_id": "001", "source": "real",
            "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
            "query": "The first deer from left to right", "family": "ordinal_direction",
            "bucket": "ordinal", "quota_state": "quota", "facts": ["rank:1"], "words": 7,
            "edited": False,
        }]
        asm = self._write_assembly(records)
        queries = self._write_queries({"001_00000001#01": "The leftmost deer"})

        result = apply(asm, queries, "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["human"], 1)
        self.assertEqual(result["records"][0]["query"], "The leftmost deer")
        self.assertEqual(result["records"][0]["review_source"], "human")

    def test_original_kept_when_no_human_edit(self):
        records = [{
            "sample_id": "001_00000001", "sequence_id": "001", "source": "real",
            "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
            "query": "The first deer from left to right", "family": "ordinal_direction",
            "bucket": "ordinal", "quota_state": "quota", "facts": ["rank:1"], "words": 7,
            "edited": False,
        }]
        asm = self._write_assembly(records)
        queries = self._write_queries({})

        result = apply(asm, queries, "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["original"], 1)
        self.assertEqual(result["records"][0]["query"], "The first deer from left to right")
        self.assertEqual(result["records"][0]["review_source"], "original")

    def test_bucket_reclassification(self):
        records = [{
            "sample_id": "001_00000001", "sequence_id": "001", "source": "real",
            "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
            "query": "The first deer from left to right", "family": "ordinal_direction",
            "bucket": "ordinal", "quota_state": "quota", "facts": ["rank:1"], "words": 7,
            "edited": False,
        }]
        asm = self._write_assembly(records)
        # Human edits to a non-ordinal query
        queries = self._write_queries({"001_00000001#01": "The deer with antlers"})

        result = apply(asm, queries, "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["records"][0]["bucket"], "attribute_action")
        self.assertEqual(result["records"][0]["original_bucket"], "ordinal")
        self.assertEqual(result["stats"]["bucket_changed"], 1)

    def test_collision_detection(self):
        records = [
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "real",
             "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
             "query": "the deer", "family": "plain_attribute", "bucket": "attribute_action",
             "quota_state": "quota", "facts": [], "words": 2, "edited": False},
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "teacher",
             "category": "deer", "bbox": [0.5, 0.2, 0.7, 0.4], "object_index": 2,
             "query": "the deer", "family": "plain_attribute", "bucket": "attribute_action",
             "quota_state": "quota", "facts": [], "words": 2, "edited": False},
        ]
        asm = self._write_assembly(records)
        queries = self._write_queries({})

        result = apply(asm, queries, "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["collision"], 2)
        self.assertTrue(result["records"][0]["collision"])
        self.assertTrue(result["records"][1]["collision"])

    def test_output_dir_exists_no_force(self):
        records = [{"sample_id": "001_00000001", "sequence_id": "001", "source": "real",
                     "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
                     "query": "test", "family": "plain_attribute", "bucket": "attribute_action",
                     "quota_state": "quota", "facts": [], "words": 1, "edited": False}]
        asm = self._write_assembly(records)
        queries = self._write_queries({})
        # Create output dir first
        (self.tmp_path / "asm-test-r6").mkdir()

        with self.assertRaises(SystemExit):
            apply(asm, queries, "asm-test-r6", self.tmp_path, force=False)

    def test_detect_collisions_empty(self):
        self.assertEqual(_detect_collisions([]), set())

    def test_detect_collisions_no_dup(self):
        records = [
            {"sample_id": "A", "query": "foo", "object_index": 1},
            {"sample_id": "A", "query": "bar", "object_index": 2},
        ]
        self.assertEqual(_detect_collisions(records), set())

    def test_multiple_review_queries_merge(self):
        records = [
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "real",
             "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
             "query": "The first deer from left to right", "family": "ordinal_direction",
             "bucket": "ordinal", "quota_state": "quota", "facts": ["rank:1"], "words": 7,
             "edited": False},
            {"sample_id": "001_00000002", "sequence_id": "001", "source": "real",
             "category": "deer", "bbox": [0.4, 0.5, 0.6, 0.7], "object_index": 2,
             "query": "The second deer from left to right", "family": "ordinal_direction",
             "bucket": "ordinal", "quota_state": "quota", "facts": ["rank:2"], "words": 7,
             "edited": False},
        ]
        asm = self._write_assembly(records)
        part1 = self.tmp_path / "part1_queries.json"
        part1.write_text(json.dumps({"001_00000001#01": "The leftmost deer"}))
        part2 = self.tmp_path / "part2_queries.json"
        part2.write_text(json.dumps({"001_00000002#02": "The rightmost deer"}))

        result = apply(asm, [part1, part2], "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["human"], 2)
        self.assertEqual(result["records"][0]["query"], "The leftmost deer")
        self.assertEqual(result["records"][1]["query"], "The rightmost deer")

    def test_human_box_override_and_human_absent_removal(self):
        records = [
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "real",
             "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
             "query": "the deer", "family": "plain_attribute", "bucket": "attribute_action",
             "quota_state": "quota", "facts": [], "words": 2, "edited": False},
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "real",
             "category": "fox", "bbox": [0.5, 0.2, 0.7, 0.4], "object_index": 2,
             "query": "the fox", "family": "plain_attribute", "bucket": "attribute_action",
             "quota_state": "quota", "facts": [], "words": 2, "edited": False},
            {"sample_id": "001_00000001", "sequence_id": "001", "source": "teacher",
             "category": "boar", "bbox": [0.2, 0.2, 0.4, 0.4], "object_index": 3,
             "query": "the boar", "family": "plain_attribute", "bucket": "attribute_action",
             "quota_state": "quota", "facts": [], "words": 2, "edited": False},
        ]
        asm = self._write_assembly(records)
        self._write_queries({})
        # Human-corrected box for #01; #02 untouched (no-op); #03 human-confirmed absent.
        boxes = {
            "001_00000001#01": [0.11, 0.21, 0.31, 0.41],
            "001_00000001#02": [0.5, 0.2, 0.7, 0.4],
        }
        (self.tmp_path / "annotations.predictions.json").write_text(json.dumps(boxes))
        absent = {
            "001_00000001#03": "fang0:absent",
            "001_00000002#01": "glm-4.6v:absent",  # teacher-pending: not removed
        }
        (self.tmp_path / "annotations.absent.json").write_text(json.dumps(absent))

        result = apply(asm, self.tmp_path / "annotations.queries.json",
                       "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["box_changed"], 1)
        self.assertEqual(result["stats"]["box_seen"], 2)
        self.assertEqual(result["stats"]["absent_removed"], 1)
        self.assertEqual(len(result["records"]), 2)
        self.assertEqual(result["records"][0]["bbox"], [0.11, 0.21, 0.31, 0.41])
        self.assertEqual(result["records"][0]["original_bbox"], [0.1, 0.2, 0.3, 0.4])
        self.assertNotIn("001_00000001#03", [r["sample_id"] + "#" + f"{r['object_index']:02d}"
                                              for r in result["records"]])

    def test_todo_items_excluded_into_flagged(self):
        records = [{
            "sample_id": "001_00000001", "sequence_id": "001", "source": "real",
            "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
            "query": "The ambiguous deer", "family": "plain_attribute",
            "bucket": "attribute_action", "quota_state": "quota", "facts": [],
            "words": 3, "edited": False,
        }]
        asm = self._write_assembly(records)
        self._write_queries({})
        # Journal marks the item as reviewer:todo (pending disambiguation).
        journal = self.tmp_path / "annotations.jsonl"
        record = {"id": "001_00000001#01", "bbox": [0.1, 0.2, 0.3, 0.4],
                  "annotator": "reviewer:todo", "ts": "2026-09-08 00:00:00"}
        journal.write_text(json.dumps(record) + "\n", encoding="utf-8")

        result = apply(asm, self.tmp_path / "annotations.queries.json",
                       "asm-test-r6", self.tmp_path, force=True)
        self.assertEqual(result["stats"]["todo_excluded"], 1)
        self.assertEqual(len(result["records"]), 0)
        self.assertTrue(any(f["reason"] == "todo" for f in result["flagged"]))

    def test_missing_explicit_queries_path_raises(self):
        records = [{
            "sample_id": "001_00000001", "sequence_id": "001", "source": "real",
            "category": "deer", "bbox": [0.1, 0.2, 0.3, 0.4], "object_index": 1,
            "query": "The deer", "family": "plain_attribute",
            "bucket": "attribute_action", "quota_state": "quota", "facts": [],
            "words": 2, "edited": False,
        }]
        asm = self._write_assembly(records)
        missing = self.tmp_path / "does_not_exist.json"
        with self.assertRaises(FileNotFoundError):
            apply(asm, missing, "asm-test-r6", self.tmp_path, force=True)


if __name__ == "__main__":
    unittest.main()
