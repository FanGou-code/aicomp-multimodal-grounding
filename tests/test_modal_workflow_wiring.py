"""Behavior tests for local Modal coordination without starting remote jobs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from cloud import train as train_modal
from aicomp_grounding.config import PREPARATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.artifacts import stable_json_hash
from aicomp_grounding.images import is_trusted_image_fingerprint
from scripts import generate_queries


def _raw(local_entrypoint):
    return local_entrypoint.info.raw_f


class AnnotationEntrypointTests(unittest.TestCase):
    def test_annotation_prompt_uses_one_marked_rgb(self):
        query_prompt = generate_queries.FRAME_QUERY_PROMPT
        self.assertIn("enclosed by the red rectangle", query_prompt)
        self.assertIn("never mention the rectangle", query_prompt)
        self.assertIn("multiple same-category objects", query_prompt)
        self.assertIn("spatial relations", query_prompt)
        self.assertIn("leftmost/rightmost X", query_prompt)
        self.assertIn("Never force a variant", query_prompt)

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
            data = {"001_1": {"query": "placeholder"}}
            atomic_write_json(root / "train.json", data)
            manifest = {
                "status": "complete",
                "preparation_protocol_version": PREPARATION_PROTOCOL_VERSION,
                "index_fingerprints": {"train": stable_json_hash(data)},
                "index_sample_counts": {"train": 1},
            }
            atomic_write_json(root / "split_manifest.json", manifest)
            self.assertEqual(generate_queries._load_annotation_source(root, "train"), data)

            atomic_write_json(root / "train.json", {"001_1": {"query": "changed"}})
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
            atomic_write_json(root / "train.json", data)
            atomic_write_json(
                root / "split_manifest.json",
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


class TrainingEntrypointTests(unittest.TestCase):
    def test_smoke_preflight_does_not_persist_a_formal_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = {
                "metadata": {"training_run_id": "train_smoke"},
                "train_artifact_path": str(root / "train.json"),
                "val_artifact_path": str(root / "val.json"),
                "run_dir": str(root / "output_lora" / "train_smoke"),
                "resume_checkpoint": None,
            }
            with (
                patch.object(train_modal, "prepare_training_plan", return_value=plan),
                patch.object(train_modal.dataset_volume, "reload"),
                patch.object(train_modal.dataset_volume, "commit") as commit,
                patch.object(train_modal, "persist_training_plan") as persist,
            ):
                result = _raw(train_modal.preflight_training_environment)(
                    annotation_run_id="annot_round1",
                    smoke_test=True,
                )
            self.assertIs(result, plan)
            persist.assert_not_called()
            commit.assert_not_called()

    def test_smoke_mode_runs_gpu_path_without_expectation_of_adapters(self):
        plan = {
            "metadata": {"training_run_id": "train_smoke"},
            "skip_training": False,
            "smoke_test": True,
        }
        smoke_result = {
            "metadata": plan["metadata"],
            "status": "smoke_passed",
            "train_loss": 1.0,
            "val_loss": 1.1,
        }
        with (
            patch.object(
                train_modal.preflight_training_environment,
                "remote",
                return_value=plan,
            ) as preflight,
            patch.object(train_modal.train, "remote", return_value=smoke_result) as train,
        ):
            result = _raw(train_modal.main)(
                annotation_run_id="annot_round1",
                smoke_test=True,
            )
        self.assertIs(result, smoke_result)
        self.assertTrue(preflight.call_args.kwargs["smoke_test"])
        train.assert_called_once_with(plan)

    def test_explicit_smoke_still_runs_when_formal_training_is_complete(self):
        plan = {
            "metadata": {"training_run_id": "train_smoke"},
            "skip_training": True,
            "smoke_test": True,
            "completed": {"status": "completed"},
        }
        smoke_result = {
            "metadata": plan["metadata"],
            "status": "smoke_passed",
            "train_loss": 1.0,
            "val_loss": 1.1,
        }
        with (
            patch.object(
                train_modal.preflight_training_environment,
                "remote",
                return_value=plan,
            ) as preflight,
            patch.object(train_modal.train, "remote", return_value=smoke_result) as train,
        ):
            result = _raw(train_modal.main)(
                annotation_run_id="annot_round1",
                smoke_test=True,
            )
        self.assertIs(result, smoke_result)
        train.assert_called_once_with(plan)

    def test_preflight_only_never_starts_training_gpu(self):
        plan = {
            "metadata": {"training_run_id": "train_preflight"},
            "skip_training": False,
        }
        with (
            patch.object(
                train_modal.preflight_training_environment,
                "remote",
                return_value=plan,
            ) as preflight,
            patch.object(train_modal.train, "remote") as train,
        ):
            result = _raw(train_modal.main)(
                annotation_run_id="annot_round1",
                preflight_only=True,
                deep_verify_images=True,
            )
        self.assertIs(result, plan)
        self.assertTrue(preflight.call_args.kwargs["deep_verify_images"])
        train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
