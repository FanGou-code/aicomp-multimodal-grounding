"""Behavior tests for strict inference planning, Resume, Retry, and evaluation."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aicomp_grounding.config import INFERENCE_COMPUTE_DTYPE, TRAINING_PROTOCOL_VERSION
from aicomp_grounding.inference_state import (
    assign_pending_shards,
    build_run_metadata,
    build_shard_metadata,
    evaluate_dataset_predictions,
    fingerprint_inputs,
    fingerprint_lora,
    load_resume_predictions,
    merge_retry_predictions,
    merge_shard_payloads,
    pending_keys,
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
            adapter = root / "training" / "best"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text('{"r": 16}', encoding="utf-8")
            _write_adapter_manifest(adapter)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"first")

            resolved = resolve_lora_path("training/best", root)
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

    def test_run_id_format_and_retry_suffix(self):
        keys = ["001_1"]
        base = _metadata(keys, mode="base")
        self.assertTrue(re.fullmatch(r"infer_[0-9a-f]{16}", base["run_id"]))
        retry = _metadata(keys, mode="retry")
        self.assertTrue(re.fullmatch(r"infer_[0-9a-f]{16}_retry", retry["run_id"]))


class PendingShardTests(unittest.TestCase):
    def test_pending_shards_exclude_existing_predictions(self):
        items = [{"key": f"k{i}"} for i in range(6)]
        shards = assign_pending_shards(
            items,
            num_shards=2,
            existing_predictions={"k1", "k3"},
        )
        keys = [item["key"] for shard in shards for item in shard]
        self.assertEqual(set(keys), {"k0", "k2", "k4", "k5"})
        self.assertNotIn("k1", keys)
        self.assertNotIn("k3", keys)


class ResumeFilesTests(unittest.TestCase):
    def test_foreign_checkpoint_rejected_without_rewriting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            metadata = _metadata(["001_1"])
            path = root / "checkpoint.json"
            atomic_write_json(path, {"metadata": {**metadata, "model": "foreign"},
                                     "predictions": {"001_1": [0, 0, 1, 1]}})
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                load_resume_predictions(root, metadata, ["001_1"])
            self.assertEqual(before, path.read_bytes())

    def test_flat_predictions_require_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "predictions.json", {"001_1": None})
            with self.assertRaisesRegex(ValueError, "without recorded metadata"):
                load_resume_predictions(root, _metadata(["001_1"]), ["001_1"])

    def test_base_and_legacy_shard_progress_survive_repeated_resume(self):
        keys = ["001_1", "002_1", "003_1"]
        metadata = _metadata(keys, num_shards=2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "checkpoint.json", {
                "metadata": metadata, "predictions": {keys[0]: None},
            })
            shard_path = root / "shard_checkpoints/shard_00.checkpoint.json"
            atomic_write_json(shard_path, {
                "metadata": build_shard_metadata(metadata, 0, keys[1:]),
                "predictions": {keys[1]: [0, 0, 1, 1]},
            })
            predictions = load_resume_predictions(root, metadata, keys)
            self.assertEqual(set(predictions), set(keys[:2]))
            atomic_write_json(root / "checkpoint.json", {
                "metadata": metadata, "predictions": predictions,
            })
            atomic_write_json(shard_path, {
                "metadata": build_shard_metadata(metadata, 0, keys[2:]),
                "assigned_keys": keys[2:], "predictions": {keys[2]: [0.1, 0.1, 0.5, 0.5]},
            })
            final = load_resume_predictions(root, metadata, keys)
            self.assertEqual(set(final), set(keys))
            self.assertEqual(final[keys[1]], [0, 0, 1, 1])

    def test_conflicting_results_and_wrong_shard_assignment_are_rejected(self):
        keys = ["001_1", "002_1"]
        metadata = _metadata(keys, num_shards=2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            atomic_write_json(root / "checkpoint.json", {
                "metadata": metadata, "predictions": {keys[0]: None},
            })
            path = root / "shard_checkpoints/shard_00.checkpoint.json"
            atomic_write_json(path, {
                "metadata": build_shard_metadata(metadata, 0, keys),
                "assigned_keys": keys, "predictions": {keys[0]: [0, 0, 1, 1]},
            })
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_resume_predictions(root, metadata, keys)
            atomic_write_json(path, {
                "metadata": build_shard_metadata(metadata, 0, keys),
                "assigned_keys": keys[1:], "predictions": {keys[1]: None},
            })
            with self.assertRaises(ValueError):
                load_resume_predictions(root, metadata, keys)

    def test_identical_lora_content_allows_different_path_in_trace(self):
        keys = ["001_1"]
        metadata = _metadata(keys, lora_path=Path("/new/adapter"), adapter_fingerprint="lora_same")
        payload = {"metadata": {**metadata, "lora_path": "/old/adapter"},
                   "predictions": {keys[0]: None}}
        self.assertEqual(validate_checkpoint_payload(payload, metadata, keys, require_complete=True),
                         {keys[0]: None})
        payload["metadata"]["adapter_fingerprint"] = "lora_foreign"
        with self.assertRaises(ValueError):
            validate_checkpoint_payload(payload, metadata, keys, require_complete=True)

    def test_multishard_with_loader_workers_and_completed_resume(self):
        import argparse
        import contextlib
        import io
        from PIL import Image
        from tools.infer import run_cli
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            rows = {}
            for i in range(4):
                row = {"query": f"object {i}", "bbox": [0, 0, 1, 1]}
                for field in ("visible", "infrared", "depth"):
                    name = f"{i}_{field}.png"
                    Image.new("RGB", (16, 16)).save(data / name)
                    row[field] = name
                rows[str(i)] = row
            atomic_write_json(root / "index.json", rows)
            args = argparse.Namespace(project_root=root, data_dir=data, test_json=root / "index.json",
                output_dir=root / "outputs", model="mock", model_path=None, lora_path=None,
                max_pixels=2408448, annotation_run_id="", run_tag="regression", limit=0,
                num_shards=2, num_workers=1, batch_size=2, batch_save=1, resume=True)
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_cli(args)
                second = run_cli(args)
            self.assertEqual(first["total_predictions"], 4)
            self.assertEqual(second["total_predictions"], 4)
            self.assertEqual(first["metadata"], second["metadata"])


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
        metrics = evaluate_dataset_predictions(data, keys, predictions)
        self.assertEqual(metrics["hits"], 1)
        self.assertEqual(metrics["total"], 3)
        self.assertAlmostEqual(metrics["acc_at_0_5"], 1 / 3)
        self.assertEqual(metrics["failures"], 1)


if __name__ == "__main__":
    unittest.main()
