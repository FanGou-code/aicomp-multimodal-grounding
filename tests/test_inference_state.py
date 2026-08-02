"""Behavior tests for strict inference planning, Resume, Retry, and evaluation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aicomp_grounding.config import INFERENCE_COMPUTE_DTYPE, TRAINING_PROTOCOL_VERSION
from aicomp_grounding.inference_state import (
    build_run_metadata,
    build_shard_metadata,
    evaluate_predictions,
    fingerprint_inputs,
    fingerprint_lora,
    merge_retry_predictions,
    merge_shard_payloads,
    pending_keys,
    predictions_are_submission_ready,
    resolve_lora_path,
    validate_checkpoint_payload,
)
from aicomp_grounding.io import atomic_write_json


def _write_adapter_manifest(path: Path) -> None:
    atomic_write_json(
        path / "adapter_manifest.json",
        {
            "metadata": {
                "protocol_version": TRAINING_PROTOCOL_VERSION,
                "training_run_id": "train_0123456789abcdef",
                "annotation_run_id": "annot_round1",
                "train_dataset_fingerprint": "train-data",
                "val_dataset_fingerprint": "val-data",
                "train_image_fingerprint": "train-images",
                "val_image_fingerprint": "val-images",
                "annotation_prompt_hash": "annotation-prompt",
                "model_name": "model",
                "model_revision": "revision",
                "grounding_prompt_hash": "grounding-prompt",
                "hyperparameters": {"epochs": 3},
                "seed": 42,
                "run_tag": "round1",
                "train_samples": 2,
                "val_samples": 1,
            },
            "epoch": 1,
            "val_loss": 0.5,
        },
    )


def _dataset() -> dict:
    return {
        "001_1": {
            "visible": "Train/001/color/1.png",
            "infrared": "Train/001/infrared/1.png",
            "depth": "Processed/Train/001/depth/1.png",
            "query": "The red car beside the curb",
            "bbox": [0.1, 0.1, 0.5, 0.5],
        },
        "002_1": {
            "visible": "Train/002/color/1.png",
            "infrared": "Train/002/infrared/1.png",
            "depth": "Processed/Train/002/depth/1.png",
            "query": "The person wearing a blue shirt",
            "bbox": [0.2, 0.2, 0.6, 0.6],
        },
        "003_1": {
            "visible": "Train/003/color/1.png",
            "infrared": "Train/003/infrared/1.png",
            "depth": "Processed/Train/003/depth/1.png",
            "query": "The bicycle near the tree",
            "bbox": [0.3, 0.3, 0.7, 0.7],
        },
    }


def _metadata(keys: list[str], **overrides) -> dict:
    data = _dataset()
    values = {
        "mode": "base",
        "split": "val",
        "annotation_run_id": "annot_round1",
        "model": "model",
        "model_revision": "revision",
        "lora_path": None,
        "adapter_fingerprint": "base",
        "prompt_hash": "prompt",
        "generation_config": {"do_sample": False, "max_new_tokens": 32},
        "run_tag": "",
        "limit": None,
        "selected_keys": keys,
        "input_fingerprint": fingerprint_inputs(data, keys),
        "image_fingerprint": "images",
        "num_shards": 1,
        "base_run_id": "",
    }
    values.update(overrides)
    return build_run_metadata(**values)


class LoraIdentityTests(unittest.TestCase):
    def test_relative_path_is_canonical_and_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            adapter = root / "output_lora" / "best"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            _write_adapter_manifest(adapter)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"first")

            resolved = resolve_lora_path("output_lora/best", root)
            self.assertEqual(resolved, adapter.resolve())
            first = fingerprint_lora(resolved)
            weights.write_bytes(b"second")
            self.assertNotEqual(first, fingerprint_lora(resolved))

    def test_lora_must_stay_inside_data_root_and_have_required_files(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as outside:
            root = Path(td)
            with self.assertRaises(ValueError):
                resolve_lora_path(outside, root)
            incomplete = root / "adapter"
            incomplete.mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_lora_path(incomplete, root)
            (incomplete / "adapter_config.json").write_text("{}", encoding="utf-8")
            _write_adapter_manifest(incomplete)
            (incomplete / "training_args.bin").write_bytes(b"not adapter weights")
            with self.assertRaisesRegex(FileNotFoundError, "adapter_model"):
                resolve_lora_path(incomplete, root)


class RunIdentityTests(unittest.TestCase):
    def test_compute_dtype_is_part_of_run_identity(self):
        bf16_metadata = _metadata(["001_1"])
        with patch(
            "aicomp_grounding.inference_state.INFERENCE_COMPUTE_DTYPE", "float16"
        ):
            fp16_metadata = _metadata(["001_1"])

        self.assertEqual(bf16_metadata["compute_dtype"], INFERENCE_COMPUTE_DTYPE)
        self.assertNotEqual(bf16_metadata["run_id"], fp16_metadata["run_id"])

    def test_limit_run_tag_and_input_change_run_identity(self):
        keys = ["001_1", "002_1"]
        base = _metadata(keys)
        tagged = _metadata(keys, run_tag="experiment/one")
        limited = _metadata(keys[:1], limit=1)
        self.assertNotEqual(base["run_id"], tagged["run_id"])
        self.assertNotEqual(base["run_id"], limited["run_id"])
        self.assertNotIn("experiment", tagged["run_id"])

    def test_annotation_run_changes_run_identity(self):
        keys = ["001_1", "002_1"]
        first = _metadata(keys, annotation_run_id="annot_round1")
        second = _metadata(keys, annotation_run_id="annot_round2")
        self.assertNotEqual(first["run_id"], second["run_id"])

    def test_test_metadata_forbids_annotation_run(self):
        with self.assertRaisesRegex(ValueError, "forbids annotation_run_id"):
            _metadata(["001_1"], split="test", annotation_run_id="annot_round1")


class CheckpointTests(unittest.TestCase):
    def test_none_is_completed_failure_but_missing_key_is_pending(self):
        self.assertEqual(pending_keys(["a", "b", "c"], {"a": [0, 0, 1, 1], "b": None}), ["c"])

    def test_checkpoint_rejects_missing_metadata_extra_key_and_invalid_bbox(self):
        keys = ["001_1"]
        run = _metadata(keys)
        shard = build_shard_metadata(run, 0, keys)
        valid = {"metadata": shard, "predictions": {"001_1": [0.1, 0.1, 0.5, 0.5]}}
        self.assertEqual(
            validate_checkpoint_payload(valid, shard, keys, require_complete=True)["001_1"],
            [0.1, 0.1, 0.5, 0.5],
        )

        missing_meta = {"metadata": dict(shard), "predictions": valid["predictions"]}
        del missing_meta["metadata"]["prompt_hash"]
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(missing_meta, shard, keys, require_complete=True)
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(
                {"metadata": shard, "predictions": {"other": [0, 0, 1, 1]}},
                shard,
                keys,
                require_complete=True,
            )
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(
                {"metadata": shard, "predictions": {"001_1": [0.8, 0.8, 0.2, 0.2]}},
                shard,
                keys,
                require_complete=True,
            )

    def test_shard_merge_requires_exact_assignments(self):
        assignments = [["001_1", "002_1"], ["003_1"]]
        run = _metadata([key for shard in assignments for key in shard], num_shards=2)
        payloads = [
            {
                "metadata": build_shard_metadata(run, shard_id, keys),
                "predictions": {key: None for key in keys},
            }
            for shard_id, keys in enumerate(assignments)
        ]
        merged = merge_shard_payloads(run, assignments, list(reversed(payloads)))
        self.assertEqual(set(merged), {"001_1", "002_1", "003_1"})
        with self.assertRaises(ValueError):
            merge_shard_payloads(run, assignments, payloads[:1])

        with self.assertRaisesRegex(ValueError, "num_shards"):
            merge_shard_payloads(run, assignments[:1], payloads[:1])
        duplicate_assignments = [["001_1", "002_1"], ["002_1", "003_1"]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            merge_shard_payloads(run, duplicate_assignments, payloads)
        incomplete_assignments = [["001_1"], ["003_1"]]
        incomplete_payloads = [
            {
                "metadata": build_shard_metadata(run, shard_id, keys),
                "predictions": {key: None for key in keys},
            }
            for shard_id, keys in enumerate(incomplete_assignments)
        ]
        with self.assertRaisesRegex(ValueError, "cover"):
            merge_shard_payloads(run, incomplete_assignments, incomplete_payloads)


class RetryAndEvaluationTests(unittest.TestCase):
    def test_submission_ready_requires_full_valid_coverage(self):
        keys = ["a", "b"]
        self.assertTrue(
            predictions_are_submission_ready(
                keys,
                {"a": [0.1, 0.1, 0.5, 0.5], "b": [0.2, 0.2, 0.6, 0.6]},
            )
        )
        self.assertFalse(
            predictions_are_submission_ready(
                keys,
                {"a": [0.1, 0.1, 0.5, 0.5], "b": None},
            )
        )
        self.assertFalse(
            predictions_are_submission_ready(keys, {"a": [0.1, 0.1, 0.5, 0.5]})
        )

    def test_retry_only_replaces_failures_with_valid_overlay(self):
        base = {"a": [0.1, 0.1, 0.5, 0.5], "b": None, "c": None}
        original = dict(base)
        merged = merge_retry_predictions(
            base,
            {"b": [0.2, 0.2, 0.6, 0.6], "c": None},
            ["b", "c"],
        )
        self.assertEqual(base, original)
        self.assertEqual(merged["a"], base["a"])
        self.assertEqual(merged["b"], [0.2, 0.2, 0.6, 0.6])
        self.assertIsNone(merged["c"])
        with self.assertRaises(ValueError):
            merge_retry_predictions(base, {"a": [0, 0, 1, 1]}, ["b", "c"])

    def test_val_metrics_include_failed_predictions_in_denominator(self):
        data = _dataset()
        keys = list(data)
        predictions = {
            "001_1": [0.1, 0.1, 0.5, 0.5],
            "002_1": [0.0, 0.0, 0.1, 0.1],
            "003_1": None,
        }
        metrics = evaluate_predictions(data, keys, predictions)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["acc_at_0_5"], 1 / 3)
        self.assertEqual(metrics["failures"], 1)


if __name__ == "__main__":
    unittest.main()
