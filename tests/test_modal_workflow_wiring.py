"""Behavior tests for local Modal coordination without starting remote jobs."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import run_inference
import train_modal
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
        self.assertIn("similar same-category objects", query_prompt)
        self.assertIn("viewer's perspective", query_prompt)
        self.assertIn("sole disambiguator", query_prompt)
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
            patch.dict("os.environ", {"ZHIPU_API_KEY": "test-key"}),
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


class InferenceEntrypointTests(unittest.TestCase):
    def test_retry_seed_depends_on_query_identity_not_resume_position(self):
        first = run_inference._retry_query_seed(42, "query-b")
        self.assertEqual(first, run_inference._retry_query_seed(42, "query-b"))
        self.assertNotEqual(first, run_inference._retry_query_seed(42, "query-a"))

    def test_invalid_split_and_parallel_retry_fail_before_preflight(self):
        remote = MagicMock()
        with patch.object(run_inference.preflight_inference_environment, "remote", remote):
            with self.assertRaises(ValueError):
                _raw(run_inference.main)(split="invalid")
            with self.assertRaises(ValueError):
                _raw(run_inference.main)(
                    split="test",
                    retry_failed=True,
                    base_run_id="infer_base",
                    num_shards=2,
                )
        remote.assert_not_called()

    def test_annotation_binding_errors_fail_before_preflight(self):
        remote = MagicMock()
        with patch.object(run_inference.preflight_inference_environment, "remote", remote):
            with self.assertRaisesRegex(ValueError, "requires.*annotation_run_id"):
                _raw(run_inference.main)(split="val")
            with self.assertRaisesRegex(ValueError, "forbids annotation_run_id"):
                _raw(run_inference.main)(
                    split="test",
                    annotation_run_id="annot_round1",
                )
        remote.assert_not_called()

    def test_inference_workers_are_consumed_before_finalizer(self):
        plan = {
            "metadata": {"run_id": "infer_val_base_test"},
            "keys": ["001_1", "002_1"],
            "shards": [["001_1"], ["002_1"]],
            "result_keys": ["001_1", "002_1"],
            "base_predictions": {},
        }
        consumed = {"done": False}

        def payloads():
            yield {"metadata": {"shard_id": 0}, "predictions": {}}
            yield {"metadata": {"shard_id": 1}, "predictions": {}}
            consumed["done"] = True

        def finalize(_plan, received):
            self.assertTrue(consumed["done"])
            self.assertEqual(len(received), 2)
            return {
                "metadata": {"run_id": "infer_val_base_test"},
                "predictions": {"001_1": None, "002_1": None},
                "result_is_full_split": True,
                "metrics": {
                    "hits": 0,
                    "total": 2,
                    "acc_at_0_5": 0.0,
                    "mean_iou": 0.0,
                    "failures": 2,
                },
            }

        with (
            patch.object(run_inference.preflight_inference_environment, "remote", return_value=plan),
            patch.object(run_inference.run_inference_shard, "starmap", return_value=payloads()) as starmap,
            patch.object(run_inference.finalize_inference_run, "remote", side_effect=finalize),
        ):
            result = _raw(run_inference.main)(
                split="val",
                annotation_run_id="annot_round1",
                num_shards=2,
            )
        self.assertEqual(result["metrics"]["failures"], 2)
        self.assertEqual(len(starmap.call_args.args[0]), 2)

    def test_incomplete_full_test_does_not_overwrite_submission_predictions(self):
        plan = {
            "metadata": {"run_id": "infer_test_base_incomplete"},
            "keys": ["001_1"],
            "shards": [],
            "result_keys": ["001_1"],
            "base_predictions": {},
            "pending_shard_ids": [],
            "completed_payloads": [],
        }
        result = {
            "metadata": plan["metadata"],
            "predictions": {"001_1": None},
            "metrics": None,
            "result_is_full_split": True,
            "submission_ready": False,
        }
        with (
            patch.object(run_inference.preflight_inference_environment, "remote", return_value=plan),
            patch.object(run_inference.finalize_inference_run, "remote", return_value=result),
            patch.object(run_inference, "atomic_write_json") as write_json,
        ):
            returned = _raw(run_inference.main)(split="test")
        self.assertIs(returned, result)
        write_json.assert_not_called()

    def test_preflight_only_never_starts_inference_gpu_or_finalizer(self):
        plan = {
            "metadata": {"run_id": "infer_preflight"},
            "shards": [["001_1"]],
            "pending_shard_ids": [0],
        }
        with (
            patch.object(
                run_inference.preflight_inference_environment,
                "remote",
                return_value=plan,
            ) as preflight,
            patch.object(run_inference.run_inference_shard, "starmap") as starmap,
            patch.object(run_inference.finalize_inference_run, "remote") as finalize,
        ):
            result = _raw(run_inference.main)(
                split="test",
                preflight_only=True,
                deep_verify_images=True,
            )
        self.assertIs(result, plan)
        self.assertTrue(preflight.call_args.kwargs["deep_verify_images"])
        starmap.assert_not_called()
        finalize.assert_not_called()


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
                patch.object(train_modal, "atomic_write_json") as write_json,
            ):
                result = _raw(train_modal.preflight_training_environment)(
                    annotation_run_id="annot_round1",
                    smoke_test=True,
                )
            self.assertIs(result, plan)
            write_json.assert_not_called()
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


class InferencePlanSafetyTests(unittest.TestCase):
    @staticmethod
    def _dataset(count: int = 2) -> dict:
        return {
            f"{index:03d}_1": {
                "visible": f"Test/Images/visible/{index:03d}.png",
                "infrared": f"Test/Images/infrared/{index:03d}.png",
                "depth": f"Processed/Test/depth_jet/{index:03d}.png",
                "query": f"The target object number {index} near the center",
            }
            for index in range(1, count + 1)
        }

    @staticmethod
    def _write_base_checkpoint(root: Path, plan: dict) -> None:
        path = (
            root
            / "outputs"
            / "inference"
            / plan["metadata"]["run_id"]
            / "checkpoint.json"
        )
        atomic_write_json(
            path,
            {
                "metadata": plan["metadata"],
                "predictions": {key: None for key in plan["result_keys"]},
            },
        )

    @staticmethod
    def _write_images(root: Path, dataset: dict, *, depth_value: int = 0) -> None:
        import cv2
        import numpy as np

        for item in dataset.values():
            for field in ("visible", "infrared", "depth"):
                path = root / item[field]
                path.parent.mkdir(parents=True, exist_ok=True)
                value = depth_value if field == "depth" else 0
                image = np.full((4, 6, 3), value, dtype=np.uint8)
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"Failed to write test image {path}")

    def test_retry_of_limited_base_is_never_marked_as_full_test(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "test.json", self._dataset())
            base = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                limit=1,
                verify_images=False,
            )
            self._write_base_checkpoint(root, base)
            retry = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                retry_failed=True,
                base_run_id=base["metadata"]["run_id"],
                verify_images=False,
            )
            self.assertFalse(retry["result_is_full_split"])
            self.assertEqual(len(retry["result_keys"]), 1)

    def test_full_base_is_rejected_if_dataset_gains_a_key(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "test.json", self._dataset())
            base = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                verify_images=False,
            )
            self._write_base_checkpoint(root, base)
            atomic_write_json(root / "test.json", self._dataset(count=3))
            with self.assertRaisesRegex(ValueError, "declared dataset selection"):
                run_inference.prepare_inference_plan(
                    data_root=root,
                    split="test",
                    retry_failed=True,
                    base_run_id=base["metadata"]["run_id"],
                    verify_images=False,
                )

    def test_cpu_preflight_skips_complete_shards_without_starting_gpu_workers(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "test.json", self._dataset())
            initial = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                num_shards=2,
                resume=False,
                verify_images=False,
            )
            for shard_id, assigned in enumerate(initial["shards"]):
                checkpoint = (
                    root
                    / "outputs"
                    / "inference"
                    / initial["metadata"]["run_id"]
                    / "shards"
                    / f"shard_{shard_id:02d}.json"
                )
                atomic_write_json(
                    checkpoint,
                    {
                        "metadata": run_inference.build_shard_metadata(
                            initial["metadata"], shard_id, assigned
                        ),
                        "predictions": {key: None for key in assigned},
                    },
                )
            resumed = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                num_shards=2,
                resume=True,
                verify_images=False,
            )
            self.assertEqual(resumed["pending_shard_ids"], [])
            self.assertEqual(len(resumed["completed_payloads"]), 2)

    def test_resume_false_does_not_read_existing_shard_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "test.json", self._dataset(count=1))
            initial = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                resume=False,
                verify_images=False,
            )
            checkpoint = (
                root
                / "outputs"
                / "inference"
                / initial["metadata"]["run_id"]
                / "shards"
                / "shard_00.json"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("not json", encoding="utf-8")
            fresh = run_inference.prepare_inference_plan(
                data_root=root,
                split="test",
                resume=False,
                verify_images=False,
            )
            self.assertEqual(fresh["pending_shard_ids"], [0])
            self.assertEqual(fresh["completed_payloads"], [])

    def test_image_byte_change_changes_run_identity_at_same_paths(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self._dataset(count=1)
            atomic_write_json(root / "test.json", dataset)
            atomic_write_json(root / "Test" / "queries" / "queries.json", {})
            atomic_write_json(root / "split_manifest.json", {})
            self._write_images(root, dataset, depth_value=0)
            with (
                patch.object(run_inference, "validate_processed_test_index"),
                patch.object(run_inference, "validate_test_preparation_manifest"),
            ):
                first = run_inference.prepare_inference_plan(
                    data_root=root,
                    split="test",
                    verify_images=True,
                )
                self._write_images(root, dataset, depth_value=255)
                second = run_inference.prepare_inference_plan(
                    data_root=root,
                    split="test",
                    verify_images=True,
                )
            self.assertNotEqual(
                first["metadata"]["image_fingerprint"],
                second["metadata"]["image_fingerprint"],
            )
            self.assertNotEqual(first["metadata"]["run_id"], second["metadata"]["run_id"])

    def test_committed_test_preflight_keeps_contract_checks_without_image_io(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset = self._dataset(count=1)
            atomic_write_json(root / "test.json", dataset)
            atomic_write_json(root / "Test" / "queries" / "queries.json", {})
            atomic_write_json(root / "split_manifest.json", {})
            with (
                patch.object(run_inference, "validate_processed_test_index") as index_gate,
                patch.object(
                    run_inference,
                    "validate_test_preparation_manifest",
                ) as manifest_gate,
                patch.object(
                    run_inference,
                    "_verify_selected_images",
                    side_effect=AssertionError("deep verification must remain disabled"),
                ),
            ):
                plan = run_inference.prepare_inference_plan(
                    data_root=root,
                    split="test",
                    verify_images=False,
                    validate_committed_volume=True,
                )
            index_gate.assert_called_once()
            manifest_gate.assert_called_once()
            self.assertFalse(manifest_gate.call_args.kwargs["verify_file_bytes"])
            self.assertTrue(
                is_trusted_image_fingerprint(plan["metadata"]["image_fingerprint"])
            )

    def test_test_preflight_enforces_official_template_contract(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "test.json", self._dataset(count=1))
            atomic_write_json(root / "Test" / "queries" / "queries.json", {})
            with patch.object(
                run_inference,
                "validate_processed_test_index",
                side_effect=ValueError("contract mismatch"),
            ) as validate:
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    run_inference.prepare_inference_plan(
                        data_root=root,
                        split="test",
                        verify_images=True,
                    )
            validate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
