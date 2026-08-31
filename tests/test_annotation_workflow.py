"""Behavior tests for annotation coordination without starting remote jobs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aicomp_grounding.config import PREPARATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.images import is_trusted_image_fingerprint
from aicomp_grounding.query_style import STYLE_PROMPT_HASH
from scripts import generate_queries


class AnnotationEntrypointTests(unittest.TestCase):
    def test_annotation_entrypoint_removed_legacy_free_prompt(self):
        self.assertFalse(hasattr(generate_queries, "FRAME_QUERY_PROMPT"))
        self.assertTrue(STYLE_PROMPT_HASH)

    def test_annotation_request_disables_glm_thinking(self):
        self.assertEqual(generate_queries.GENERATION_CONFIG["thinking_mode"], "disabled")

    def test_test_split_is_rejected_before_preflight(self):
        preflight = MagicMock()
        with patch.object(generate_queries, "preflight_annotation_run", preflight):
            with self.assertRaises(ValueError):
                generate_queries.run_annotation(split="test")
        preflight.assert_not_called()

    def test_api_workers_are_consumed_before_finalizer(self):
        plan = {
            "metadata": {
                "run_id": "annot_test",
                "selected_sequence_ids": ["001", "002"],
            },
            "shards": [["001"], ["002"]],
            "selected_sample_ids": ["001_1", "002_1"],
            "completed_payloads": [],
            "pending_shard_ids": [0, 1],
        }
        consumed = {"done": False}

        def worker(**kwargs):
            self.assertIn("rate_limiter", kwargs)
            if kwargs["shard_id"] == 1:
                consumed["done"] = True
            return {"metadata": {"shard_id": kwargs["shard_id"]}, "results": {}}

        def finalize(**kwargs):
            self.assertTrue(consumed["done"])
            payloads = kwargs["payloads"]
            self.assertEqual(len(payloads), 2)
            self.assertFalse(kwargs["publish"])
            return {
                "run_id": "annot_test",
                "qc": {
                    "completed_sequences": 2,
                    "selected_sequences": 2,
                    "usage": {"api_calls": 2, "prompt_tokens": 10, "completion_tokens": 4},
                },
                "approved_path": None,
                "preview_path": None,
            }

        with (
            patch.object(generate_queries, "preflight_annotation_run", return_value=plan),
            patch.object(generate_queries, "annotate_shard", side_effect=worker) as annotate,
            patch.object(generate_queries, "finalize_annotation_run", side_effect=finalize) as final,
            patch.dict("os.environ", {"API_KEY": "test-key"}),
        ):
            result = generate_queries.run_annotation(split="train", concurrency=2)
        self.assertEqual(result["run_id"], "annot_test")
        self.assertEqual(annotate.call_count, 2)
        final.assert_called_once()

    def test_preflight_only_never_starts_api_workers_or_finalizer(self):
        plan = {
            "metadata": {"run_id": "annot_preflight"},
            "shards": [["001"]],
            "pending_shard_ids": [0],
        }
        with (
            patch.object(generate_queries, "preflight_annotation_run", return_value=plan) as preflight,
            patch.object(generate_queries, "annotate_shard") as annotate,
            patch.object(generate_queries, "finalize_annotation_run") as finalize,
        ):
            result = generate_queries.run_annotation(
                split="train",
                preflight_only=True,
                deep_verify_images=True,
            )
        self.assertIs(result, plan)
        self.assertTrue(preflight.call_args.kwargs["deep_verify_images"])
        annotate.assert_not_called()
        finalize.assert_not_called()


class AnnotationSourceGateTests(unittest.TestCase):
    def test_annotation_source_must_match_completed_split_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            indexes = root / "indexes"
            data = {"001_1": {"query": "placeholder"}}
            atomic_write_json(indexes / "train.json", data)
            manifest = {
                "status": "complete",
                "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                "index_fingerprints": {"train": stable_json_hash(data)},
                "index_sample_counts": {"train": 1},
            }
            atomic_write_json(indexes / "split_manifest.json", manifest)
            self.assertEqual(generate_queries._load_annotation_source(root, "train"), data)

            atomic_write_json(indexes / "train.json", {"001_1": {"query": "changed"}})
            with self.assertRaisesRegex(ValueError, "does not match"):
                generate_queries._load_annotation_source(root, "train")

    def test_local_preflight_trusts_committed_index_without_reading_images(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = {
                "001_00000001": {
                    "visible": "Train/001/color/00000001.png",
                    "infrared": "Train/001/infrared/00000001.png",
                    "depth": "Processed/Train/001/depth_jet/00000001.png",
                    "bbox": [0.1, 0.1, 0.4, 0.5],
                    "width": 1920,
                    "height": 1080,
                }
            }
            atomic_write_json(root / "indexes" / "train.json", data)
            atomic_write_json(
                root / "indexes" / "split_manifest.json",
                {
                    "status": "complete",
                    "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                    "index_fingerprints": {"train": stable_json_hash(data)},
                    "index_sample_counts": {"train": 1},
                },
            )
            with (
                patch.object(
                    generate_queries,
                    "verify_dataset_images",
                    side_effect=AssertionError("deep verification must remain disabled"),
                ),
            ):
                plan = generate_queries.preflight_annotation_run(
                    data_root=root,
                    output_root=root / "outputs" / "annotations",
                    split="train",
                    limit_sequences=1,
                    concurrency=1,
                )
            self.assertTrue(
                is_trusted_image_fingerprint(plan["metadata"]["image_fingerprint"])
            )
            self.assertEqual(plan["pending_shard_ids"], [0])


if __name__ == "__main__":
    unittest.main()
