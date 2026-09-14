"""Offline tests for approved-data gating and training schedule helpers."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from aicomp_grounding.annotation_state import (
    ANNOTATION_MODE,
    ASSIGNMENT_POLICY,
    approved_dataset_fingerprint,
)
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.sequence import source_fingerprint
from aicomp_grounding.sharding import group_keys_by_scene
from aicomp_grounding.training_state import (
    accumulation_window_size,
    assert_single_cuda_device_map,
    build_epoch_adapter_manifest,
    build_training_metadata,
    move_batch_to_device,
    optimizer_steps_per_epoch,
    should_optimizer_step,
    validate_adapter_manifest,
    validate_completed_training_state,
    validate_loaded_training_state,
    validate_resume_checkpoint,
    validate_training_artifacts,
    validated_prompt_length,
)
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.images import trusted_dataset_image_fingerprint
from aicomp_grounding.training_core import (
    _load_training_state,
    persist_training_plan,
    prepare_training_plan,
)


class _RecordingTorch:
    def __init__(self) -> None:
        self.calls = []

    def load(self, path, **kwargs):
        self.calls.append((path, kwargs))
        return {"optimizer": {}}


class TrainingStateLoadingTests(unittest.TestCase):
    def test_checkpoint_load_uses_restricted_weights_only_mode(self):
        torch_module = _RecordingTorch()
        path = Path("checkpoint/training_state.pt")

        result = _load_training_state(torch_module, path)

        self.assertEqual(result, {"optimizer": {}})
        self.assertEqual(
            torch_module.calls,
            [(path, {"map_location": "cpu", "weights_only": True})],
        )


def _data(scene: str) -> dict:
    return {
        f"{scene}_00000001": {
            "visible": f"Train/{scene}/color/1.png",
            "infrared": f"Train/{scene}/infrared/1.png",
            "depth": f"Processed/Train/{scene}/depth/1.png",
            "query": "The person wearing a bright yellow waterproof jacket",
            "bbox": [0.1, 0.1, 0.4, 0.5],
            "width": 1920,
            "height": 1080,
        }
    }


def _artifact(split: str, scene: str, run_id: str = "annot_round1") -> dict:
    data = _data(scene)
    return {
        "metadata": {
            "status": "approved",
            "protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "run_id": run_id,
            "split": split,
            "source_fingerprint": source_fingerprint(data),
            "preparation_fingerprint": "preparation-bytes",
            "image_fingerprint": f"{split}-{scene}-image-bytes",
            "dataset_fingerprint": approved_dataset_fingerprint(data),
            "sample_count": len(data),
            "sequence_count": len(group_keys_by_scene(list(data), data)),
            "prompt_hash": "annotation-prompt",
            "provenance": {
                "source_type": "hosted_open_weights",
                "provider": "zhipu",
                "api_base_url": "https://api.siliconflow.cn/v1",
                "annotator_model": "open-model",
                "annotator_revision": "revision",
                "model_weights_url": "https://example.com/open-model",
                "model_license": "Apache-2.0",
                "mode": ANNOTATION_MODE,
                "assignment_policy": ASSIGNMENT_POLICY,
                "render_protocol": "clean-views-v1",
                "generation_config": {"max_tokens": 256, "enable_thinking": False},
            },
            "qc": {
                "complete": True,
                "failed_sequences": 0,
                "failed_frames": 0,
                "invalid_queries": 0,
                "generated_samples": len(data),
            },
        },
        "data": data,
    }


class ApprovedDataGateTests(unittest.TestCase):
    def test_only_matching_open_weights_approved_train_and_val_are_accepted(self):
        train = _artifact("train", "001")
        val = _artifact("val", "002")
        validated_train, validated_val = validate_training_artifacts(
            train,
            val,
            annotation_run_id="annot_round1",
        )
        self.assertIs(validated_train, train)
        self.assertIs(validated_val, val)

        closed = copy.deepcopy(train)
        closed["metadata"]["provenance"]["source_type"] = "closed_api"
        with self.assertRaises(ValueError):
            validate_training_artifacts(closed, val, annotation_run_id="annot_round1")

    def test_train_val_sequence_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_training_artifacts(
                _artifact("train", "001"),
                _artifact("val", "001"),
                annotation_run_id="annot_round1",
            )

    def test_training_identity_changes_with_data_or_hyperparameters(self):
        train = _artifact("train", "001")
        val = _artifact("val", "002")
        common = {
            "annotation_run_id": "annot_round1",
            "train_artifact": train,
            "val_artifact": val,
            "model_name": "model",
            "model_revision": "revision",
            "grounding_prompt_hash": "grounding",
            "seed": 42,
            "run_tag": "round1",
        }
        first = build_training_metadata(hyperparameters={"lr": 2e-4}, **common)
        second = build_training_metadata(hyperparameters={"lr": 1e-4}, **common)
        self.assertNotEqual(first["training_run_id"], second["training_run_id"])
        changed_images = copy.deepcopy(train)
        changed_images["metadata"]["image_fingerprint"] = "changed-image-bytes"
        third = build_training_metadata(
            hyperparameters={"lr": 2e-4},
            **{**common, "train_artifact": changed_images},
        )
        self.assertNotEqual(first["training_run_id"], third["training_run_id"])
        self.assertIs(
            build_epoch_adapter_manifest(first, epoch=1, val_loss=0.5)["metadata"],
            first,
        )
        reordered = dict(reversed(list(first.items())))
        self.assertEqual(
            validate_adapter_manifest(
                {"metadata": reordered, "epoch": 1, "val_loss": 0.5}
            )["metadata"],
            reordered,
        )

    def test_training_plan_trusts_approved_image_references_without_file_io(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round1"
            for split, scene in (("train", "001"), ("val", "002")):
                artifact = _artifact(split, scene, run_id)
                artifact["metadata"]["image_fingerprint"] = (
                    trusted_dataset_image_fingerprint(
                        artifact["data"],
                        artifact["data"],
                        require_recorded_size=True,
                    )
                )
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    artifact,
                )

            plan = prepare_training_plan(
                data_root=root,
                annotation_root=root / "outputs" / "annotations",
                output_root=root / "outputs",
                annotation_run_id=run_id,
                run_tag="qlora-r1",
                seed=42,
                resume=True,
            )
            self.assertFalse(plan["skip_training"])
            self.assertIsNone(plan["resume_checkpoint"])
            self.assertEqual(
                Path(plan["run_dir"]),
                root / "outputs" / "output_lora" / plan["metadata"]["training_run_id"],
            )

    def test_training_plan_default_output_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_test_run"
            for split, scene in (("train", "001"), ("val", "002")):
                artifact = _artifact(split, scene, run_id)
                artifact["metadata"]["image_fingerprint"] = (
                    trusted_dataset_image_fingerprint(
                        artifact["data"],
                        artifact["data"],
                        require_recorded_size=True,
                    )
                )
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    artifact,
                )

            plan = prepare_training_plan(
                data_root=root / "data",
                annotation_root=root / "outputs" / "annotations",
                annotation_run_id=run_id,
                run_tag="exp-layout",
                seed=42,
                resume=True,
            )

            self.assertEqual(
                Path(plan["run_dir"]),
                root / "data" / "output_lora" / plan["metadata"]["training_run_id"],
            )

    def test_training_plan_accepts_internvl_model(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_internvl"
            for split, scene in (("train", "001"), ("val", "002")):
                artifact = _artifact(split, scene, run_id)
                artifact["metadata"]["image_fingerprint"] = (
                    trusted_dataset_image_fingerprint(
                        artifact["data"],
                        artifact["data"],
                        require_recorded_size=True,
                    )
                )
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    artifact,
                )

            plan = prepare_training_plan(
                data_root=root,
                annotation_root=root / "outputs" / "annotations",
                output_root=root / "outputs",
                annotation_run_id=run_id,
                model="internvl35",
                run_tag="internvl-smoke",
                seed=42,
                resume=True,
                smoke_test=True,
            )
            self.assertEqual(plan["model"], "internvl35")
            self.assertEqual(
                plan["metadata"]["model_name"], "OpenGVLab/InternVL3_5-8B-HF"
            )


class TrainingScheduleTests(unittest.TestCase):
    def test_binary_resume_state_is_bound_to_json_and_run(self):
        class FakeTensor:
            def __init__(self, value=None):
                self.value = value
            def numel(self):
                return 1

        metadata = {"training_run_id": "train_0123456789abcdef"}
        state = {"completed_epoch": 1, "global_step": 10}
        binary = {
            "optimizer": {"state": {}, "param_groups": [{"params": [0]}]},
            "scheduler": {"last_epoch": 10, "_step_count": 11, "base_lrs": [2e-4], "_last_lr": [1e-4]},
            "torch_rng_state": FakeTensor(),
            "cuda_rng_state": [FakeTensor()],
            "completed_epoch": 1,
            "global_step": 10,
            "training_run_id": metadata["training_run_id"],
        }
        self.assertIs(validate_loaded_training_state(binary, state, metadata), binary)
        changed = dict(binary, global_step=11)
        with self.assertRaisesRegex(ValueError, "global_step"):
            validate_loaded_training_state(changed, state, metadata)
        wrong_run = dict(binary, training_run_id="different_run_id")
        with self.assertRaisesRegex(ValueError, "different training run"):
            validate_loaded_training_state(wrong_run, state, metadata)

    def test_tail_accumulation_gets_an_optimizer_step_and_true_divisor(self):
        self.assertEqual(optimizer_steps_per_epoch(5, 4), 2)
        self.assertFalse(should_optimizer_step(0, 5, 4))
        self.assertTrue(should_optimizer_step(3, 5, 4))
        self.assertTrue(should_optimizer_step(4, 5, 4))
        self.assertEqual(accumulation_window_size(3, 5, 4), 4)
        self.assertEqual(accumulation_window_size(4, 5, 4), 1)

    def test_batch_device_move_is_explicit_and_preserves_primitives(self):
        class FakeTensor:
            def __init__(self):
                self.calls = []

            def to(self, device, non_blocking=False):
                self.calls.append((device, non_blocking))
                return self

        tensor = FakeTensor()
        moved = move_batch_to_device({"tensor": tensor, "name": "sample"}, "cuda")
        self.assertIs(moved["tensor"], tensor)
        self.assertEqual(tensor.calls, [("cuda", True)])
        self.assertEqual(moved["name"], "sample")

    def test_cpu_or_disk_model_offload_is_rejected(self):
        assert_single_cuda_device_map({"model": 0})
        with self.assertRaises(RuntimeError):
            assert_single_cuda_device_map({"vision": 0, "language": "cpu"})

    def test_training_prompt_must_be_an_exact_token_prefix(self):
        import numpy as np

        full = np.array([[10, 20, 30, 40]], dtype=np.int64)
        prompt = np.array([[10, 20, 30]], dtype=np.int64)
        self.assertEqual(
            validated_prompt_length(full, prompt, sample_id="sample"),
            3,
        )
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            validated_prompt_length(
                full,
                np.array([[10, 99, 30]], dtype=np.int64),
                sample_id="sample",
            )


_TRAIN_PARAMS = {"batch_size": 2, "grad_accum_steps": 4, "num_epochs": 3}


class CompletedTrainingStateTests(unittest.TestCase):
    @staticmethod
    def _write_adapter(path: Path, metadata: dict, *, best: bool) -> None:
        path.mkdir(parents=True)
        atomic_write_json(path / "adapter_config.json", {"r": 16})
        (path / "adapter_model.safetensors").write_bytes(b"weights")
        manifest = {"metadata": metadata}
        if best:
            manifest.update({"epoch": 1, "val_loss": 0.5})
        else:
            manifest["completed_epochs"] = 3
        atomic_write_json(path / "adapter_manifest.json", manifest)

    def test_completed_state_requires_real_best_and_last_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "train_run"
            metadata = {"training_run_id": "train_run", "train_samples": 9}
            best = run_dir / "best" / "epoch_01"
            last = run_dir / "last"
            self._write_adapter(best, metadata, best=True)
            self._write_adapter(last, metadata, best=False)
            completed = {
                "metadata": metadata,
                "status": "completed",
                "global_step": 6,
                "best_val_loss": 0.5,
                "best_path": str(best),
                "last_path": str(last),
                "best_metric": "acc_at_0_5",
                "best_metric_value": 0.9,
            }
            self.assertIs(
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS),
                completed,
            )

            (best / "adapter_model.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "weights"):
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS)

    def test_completed_state_requires_exact_steps_and_matching_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "train_run"
            metadata = {"training_run_id": "train_run", "train_samples": 9}
            best = run_dir / "best" / "epoch_01"
            last = run_dir / "last"
            self._write_adapter(best, metadata, best=True)
            self._write_adapter(last, metadata, best=False)
            completed = {
                "metadata": metadata,
                "status": "completed",
                "global_step": 5,
                "best_val_loss": 0.5,
                "best_path": str(best),
                "last_path": str(last),
                "best_metric": "acc_at_0_5",
                "best_metric_value": 0.9,
            }
            with self.assertRaisesRegex(ValueError, "global_step"):
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS)

            completed["global_step"] = 6
            atomic_write_json(
                best / "adapter_manifest.json",
                {"metadata": metadata, "epoch": 1, "val_loss": 0.6},
            )
            with self.assertRaisesRegex(ValueError, "val_loss disagree"):
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS)

            atomic_write_json(
                best / "adapter_manifest.json",
                {"metadata": metadata, "epoch": 1, "val_loss": 0.5},
            )
            atomic_write_json(
                last / "adapter_manifest.json",
                {"metadata": metadata, "completed_epochs": 2},
            )
            with self.assertRaisesRegex(ValueError, "every training epoch"):
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS)

    def test_completed_state_rejects_adapter_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "train_run"
            metadata = {"training_run_id": "train_run", "train_samples": 9}
            outside = root / "outside" / "epoch_01"
            last = run_dir / "last"
            self._write_adapter(outside, metadata, best=True)
            self._write_adapter(last, metadata, best=False)
            completed = {
                "metadata": metadata,
                "status": "completed",
                "global_step": 6,
                "best_val_loss": 0.5,
                "best_path": str(outside),
                "last_path": str(last),
                "best_metric": "acc_at_0_5",
                "best_metric_value": 0.9,
            }
            with self.assertRaisesRegex(ValueError, "escapes"):
                validate_completed_training_state(completed, metadata, run_dir, **_TRAIN_PARAMS)

    def test_resume_checkpoint_state_is_checked_before_model_loading(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "train_run"
            metadata = {"training_run_id": "train_run", "train_samples": 9}
            best = run_dir / "best" / "epoch_01"
            checkpoint = run_dir / "checkpoints" / "epoch_01"
            self._write_adapter(best, metadata, best=True)
            self._write_adapter(checkpoint, metadata, best=True)
            (checkpoint / "training_state.pt").write_bytes(b"state")
            state = {
                "metadata": metadata,
                "completed_epoch": 1,
                "global_step": 2,
                "best_val_loss": 0.5,
                "best_path": str(best),
                "train_loss": 0.7,
                "val_loss": 0.5,
                "best_metric": "acc_at_0_5",
                "best_metric_value": 0.9,
                "epoch_metrics": {
                    "hits": 5,
                    "total": 6,
                    "acc_at_0_5": 0.8333333333333333,
                    "mean_iou": 0.7,
                    "failures": 0,
                },
            }
            atomic_write_json(checkpoint / "state.json", state)
            self.assertEqual(
                validate_resume_checkpoint(
                    checkpoint,
                    run_dir=run_dir,
                    metadata=metadata,
                    **_TRAIN_PARAMS,
                ),
                state,
            )
            atomic_write_json(
                checkpoint / "adapter_manifest.json",
                {"metadata": metadata, "epoch": 1, "val_loss": 0.6},
            )
            with self.assertRaisesRegex(ValueError, "manifest val_loss disagree"):
                validate_resume_checkpoint(
                    checkpoint,
                    run_dir=run_dir,
                    metadata=metadata,
                    **_TRAIN_PARAMS,
                )
            atomic_write_json(
                checkpoint / "adapter_manifest.json",
                {"metadata": metadata, "epoch": 1, "val_loss": 0.5},
            )
            state["global_step"] = 1
            atomic_write_json(checkpoint / "state.json", state)
            with self.assertRaisesRegex(ValueError, "global_step"):
                validate_resume_checkpoint(
                    checkpoint,
                    run_dir=run_dir,
                    metadata=metadata,
                    **_TRAIN_PARAMS,
                )


class PersistTrainingPlanTests(unittest.TestCase):
    @staticmethod
    def _plan(root: Path, *, smoke_test: bool = False) -> dict:
        run_dir = root / "output_lora" / "train_persist"
        return {
            "metadata": {"training_run_id": "train_persist"},
            "train_artifact_path": str(root / "train.json"),
            "val_artifact_path": str(root / "val.json"),
            "run_dir": str(run_dir),
            "smoke_test": smoke_test,
        }

    def test_smoke_plan_never_persists_or_commits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = self._plan(root, smoke_test=True)
            commits = []
            persist_training_plan(plan, commit_hook=lambda: commits.append(1))
            self.assertFalse((Path(plan["run_dir"]) / "plan.json").exists())
            self.assertEqual(commits, [])

    def test_formal_plan_persists_and_commits(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = self._plan(root)
            commits = []
            persist_training_plan(plan, commit_hook=lambda: commits.append(1))
            plan_path = Path(plan["run_dir"]) / "plan.json"
            self.assertTrue(plan_path.is_file())
            self.assertEqual(commits, [1])

    def test_matching_existing_plan_is_accepted_without_commit(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = self._plan(root)
            persist_training_plan(plan)
            commits = []
            persist_training_plan(plan, commit_hook=lambda: commits.append(1))
            self.assertEqual(commits, [])

    def test_diverging_existing_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = self._plan(root)
            persist_training_plan(plan)
            plan["train_artifact_path"] = str(root / "other_train.json")
            with self.assertRaisesRegex(ValueError, "Training plan mismatch"):
                persist_training_plan(plan)


if __name__ == "__main__":
    unittest.main()
