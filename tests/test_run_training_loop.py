"""Behavior tests for the training main loop via a fake LoRA-free model.

``run_training`` normally loads real weights and PEFT adapters. These tests
swap the model/adapter call surface for deterministic fakes so the loop
itself — batching, step checkpoints, epoch metrics, best-path selection — is
exercised without a GPU. Plans are built through ``prepare_training_plan``
against on-disk approved artifacts, so the entry path is the real one.
"""

from __future__ import annotations

import re
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from aicomp_grounding.annotation_state import (
    ANNOTATION_MODE,
    ASSIGNMENT_POLICY,
    approved_dataset_fingerprint,
)
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json, load_json
from aicomp_grounding.sequence import source_fingerprint
from aicomp_grounding.sharding import group_keys_by_scene
from aicomp_grounding.training_core import (
    candidate_metric_value,
    initial_best_metric_value,
    metric_improved,
    persist_training_plan,
    prepare_training_plan,
    run_training,
)
from aicomp_grounding.training_state import validate_approved_artifact  # noqa: F401
from aicomp_grounding.images import trusted_dataset_image_fingerprint
from aicomp_grounding.models.base import DEFAULT_LORA_PROJECTIONS, language_model_lora_targets
from aicomp_grounding.prompts import GROUNDING_SYSTEM_PROMPT, grounding_prompt_hash


def _data(scene: str) -> dict:
    return {
        f"{scene}_0000000{i}": {
            "visible": f"Train/{scene}/color/{i}.png",
            "infrared": f"Train/{scene}/infrared/{i}.png",
            "depth": f"Processed/Train/{scene}/depth/{i}.png",
            "query": f"the {scene} object number {i}",
            "bbox": [0.1 * i, 0.1, 0.1 * i + 0.2, 0.4],
            "width": 1920,
            "height": 1080,
        }
        for i in range(1, 5)
    }


def _artifact(split: str, scene: str, run_id: str) -> dict:
    data = _data(scene)
    return {
        "metadata": {
            "status": "approved",
            "protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "run_id": run_id,
            "split": split,
            "source_fingerprint": source_fingerprint(data),
            "preparation_fingerprint": "preparation-bytes",
            "image_fingerprint": trusted_dataset_image_fingerprint(
                data, data, require_recorded_size=True
            ),
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




class _CpuTorchShim(types.ModuleType):
    """Real torch module with cuda mapped off and device() pinned to CPU.

    Subclassing the real module keeps ``from torch import X`` working inside
    peft/torch's own import chain; only the members run_training touches are
    overridden.
    """

    def __init__(self, real):
        super().__init__("torch")
        self.__dict__.update(real.__dict__)
        self._real = real
        self.device = lambda spec=None: self._real.device("cpu")

    class cuda:  # noqa: N801 - mirrors torch.cuda
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def is_bf16_supported():
            return True

        @staticmethod
        def manual_seed_all(seed):
            return None


def _write_fixture_images(root: Path) -> None:
    from PIL import Image

    for scene in ("001", "002"):
        for i in range(1, 5):
            for relative in (
                f"Train/{scene}/color/{i}.png",
                f"Train/{scene}/infrared/{i}.png",
                f"Processed/Train/{scene}/depth/{i}.png",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (8, 8), color=(10, 20, 30)).save(path)


class _DictConfig(dict):
    """dict with attribute access: PEFT wants both membership and .model_type."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.config = _DictConfig(
            use_cache=True, model_type="qwen2_5_vl", _name_or_path=""
        )
        self.generation_config = {"use_cache": False}
        self.generate_calls = 0

    def enable_input_require_grads(self):
        pass

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, input_ids=None, labels=None, **kwargs):
        loss = self.embed(input_ids).sum()
        if labels is not None:
            loss = loss + self.embed(labels.clamp(min=0)).sum() * 0.0
        return type("Out", (), {"loss": loss})()

    @torch.no_grad()
    def generate(self, **kwargs):
        self.generate_calls += 1
        batch = kwargs["input_ids"].shape[0]
        return torch.tensor([[5, 6, 7]] * batch, device=kwargs["input_ids"].device)


def _fake_adapter(calls: dict):
    class _Adapter:
        name = "mimo_vl"
        model_name = "fake/model"
        model_revision = "rev"
        supports_lora = True
        supports_prepared_inputs = True

        def __init__(self):
            self.generation_config = {"max_new_tokens": 8, "do_sample": False}

        def prompt_hash(self):
            return grounding_prompt_hash(GROUNDING_SYSTEM_PROMPT)

        def identity(self):
            return {"model_name": self.model_name, "prompt_hash": self.prompt_hash()}

        def training_hyperparameters(self):
            return {
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "learning_rate": 1e-4,
                "epochs": 1,
                "warmup_ratio": 0.0,
                "lr_scheduler_type": "cosine",
                "max_grad_norm": 1.0,
                "weight_decay": 0.0,
                "lora_rank": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "compute_dtype": "float32",
                "autocast": False,
                "eval_batch_size": 1,
                "best_epoch_primary_metric": "acc_at_0_5",
                "lora_targets": language_model_lora_targets(*DEFAULT_LORA_PROJECTIONS),
            }

        def lora_target_modules(self):
            return language_model_lora_targets(*DEFAULT_LORA_PROJECTIONS)

        def load_for_training(self, *, device="cuda", lora_path=None, model_path=None):
            calls["load_for_training"] = True
            processor = mock.Mock()
            processor.tokenizer.pad_token_id = 0
            return _FakeModel(), processor

        def build_peft_model(self, base_model, hyperparameters):
            # The loop's LoRA injection is bypassed: the fake model has no
            # qwen-style projections. Return a trainable wrapper instead.
            import peft

            return peft.get_peft_model(base_model, peft.LoraConfig(
                r=hyperparameters["lora_rank"],
                lora_alpha=hyperparameters["lora_alpha"],
                lora_dropout=hyperparameters["lora_dropout"],
                target_modules=["embed"],
                task_type="CAUSAL_LM",
            ))

        def build_training_batch(self, item, *, data_root, processor):
            calls.setdefault("training_keys", []).append(item.get("key", ""))
            input_ids = torch.tensor([[1, 2, 3]])
            return {"input_ids": input_ids, "labels": input_ids.clone()}

        def collate_training_batch(self, batch, *, processor):
            return {
                "input_ids": torch.cat([item["input_ids"] for item in batch]),
                "labels": torch.cat([item["labels"] for item in batch]),
            }

        def build_grounding_batch(self, samples, *, processor):
            calls.setdefault("eval_keys", []).extend(s.key for s in samples)
            return {"input_ids": torch.ones(len(samples), 3, dtype=torch.long)}

        def decode_grounding_outputs(self, processor, generated_ids, prompt_len):
            return ['{"bbox_2d": [0.15, 0.1, 0.35, 0.4]}'] * generated_ids.shape[0]

        def parse_grounding_text(self, text):
            match = re.search(r'"bbox_2d":\s*\[([^\]]+)\]', text)
            if not match:
                return None
            return [float(v) for v in match.group(1).split(",")]

    return _Adapter()


class RunTrainingLoopTests(unittest.TestCase):
    def test_smoke_path_runs_one_batch_and_returns_losses(self):
        calls: dict = {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round1"
            _write_fixture_images(root)
            for split, scene in (("train", "001"), ("val", "002")):
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    _artifact(split, scene, run_id),
                )
            import sys as _sys
            import peft as _peft  # noqa: F401 - import before the torch shim
            import transformers as _transformers  # noqa: F401

            with mock.patch(
                "aicomp_grounding.training_core.get_adapter",
                return_value=_fake_adapter(calls),
            ) as adapter_patch, mock.patch(
                "aicomp_grounding.models.get_adapter", adapter_patch
            ), mock.patch.dict(
                _sys.modules, {"torch": _CpuTorchShim(torch)}
            ):
                plan = prepare_training_plan(
                    data_root=root,
                    annotation_root=root / "outputs" / "annotations",
                    output_root=root / "outputs",
                    annotation_run_id=run_id,
                    model="mimo_vl",
                    run_tag="fake-r1",
                    seed=42,
                    resume=True,
                    smoke_test=True,
                )
                self.assertTrue(plan["smoke_test"])
                result = run_training(plan, data_root=root)
            self.assertEqual(result["status"], "smoke_passed")
            self.assertTrue(calls.get("load_for_training"))
            # At least one train micro-batch ran; the smoke path stops after
            # a single optimizer step.
            self.assertGreaterEqual(len(calls.get("training_keys", [])), 2)
            # The smoke path computes a val loss batch only; the grounding
            # eval runs at epoch end in full runs.
            self.assertEqual(len(calls.get("eval_keys", [])), 0)
            self.assertIsInstance(result["train_loss"], float)
            self.assertIsInstance(result["val_loss"], float)
            # Smoke runs persist no plan.json.
            self.assertFalse((Path(plan["run_dir"]) / "plan.json").exists())

    def test_full_run_writes_plan_best_and_last_checkpoints(self):
        calls: dict = {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round2"
            _write_fixture_images(root)
            for split, scene in (("train", "001"), ("val", "002")):
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    _artifact(split, scene, run_id),
                )
            import sys as _sys
            import peft as _peft  # noqa: F401 - import before the torch shim
            import transformers as _transformers  # noqa: F401

            with mock.patch(
                "aicomp_grounding.training_core.get_adapter",
                return_value=_fake_adapter(calls),
            ) as adapter_patch, mock.patch(
                "aicomp_grounding.models.get_adapter", adapter_patch
            ), mock.patch.dict(
                _sys.modules, {"torch": _CpuTorchShim(torch)}
            ):
                plan = prepare_training_plan(
                    data_root=root,
                    annotation_root=root / "outputs" / "annotations",
                    output_root=root / "outputs",
                    annotation_run_id=run_id,
                    model="mimo_vl",
                    run_tag="fake-full",
                    seed=42,
                    resume=True,
                )
                result = run_training(plan, data_root=root)
                # plan.json is persisted by the CLI shell, not run_training.
                persist_training_plan(plan)
                replan = prepare_training_plan(
                    data_root=root,
                    annotation_root=root / "outputs" / "annotations",
                    output_root=root / "outputs",
                    annotation_run_id=run_id,
                    model="mimo_vl",
                    run_tag="fake-full",
                    seed=42,
                    resume=True,
                )
            run_dir = Path(plan["run_dir"])
            self.assertTrue((run_dir / "plan.json").exists())
            self.assertTrue((run_dir / "completed.json").exists())
            # One epoch: best/epoch_01 and last/; step checkpoints are
            # pruned by the epoch checkpoint that supersedes them.
            self.assertTrue((run_dir / "best" / "epoch_01").exists())
            self.assertTrue((run_dir / "best" / "epoch_01" / "metrics.json").exists())
            self.assertTrue((run_dir / "last").exists())
            self.assertEqual(result["status"], "completed")
            # The whole train set ran once; val went through the grounding eval.
            # Batches of 2 over 4 samples = 2 micro-batches; each eval sample
            # appears once in the grounding eval batches.
            self.assertEqual(len(calls.get("training_keys", [])), 8)
            self.assertEqual(len(calls.get("eval_keys", [])), 4)
            self.assertTrue(replan["skip_training"])


class HyperparameterOverrideTests(unittest.TestCase):
    def test_overrides_update_hyperparameters_and_change_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_overrides"
            _write_fixture_images(root)
            for split, scene in (("train", "001"), ("val", "002")):
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    _artifact(split, scene, run_id),
                )

            base_plan = prepare_training_plan(
                data_root=root,
                annotation_root=root / "outputs" / "annotations",
                output_root=root / "outputs",
                annotation_run_id=run_id,
                model="mimo_vl",
                run_tag="override-test",
                seed=42,
                resume=True,
            )

            # None overrides keep exact identity
            none_plan = prepare_training_plan(
                data_root=root,
                annotation_root=root / "outputs" / "annotations",
                output_root=root / "outputs",
                annotation_run_id=run_id,
                model="mimo_vl",
                run_tag="override-test",
                seed=42,
                resume=True,
                hyperparameter_overrides={
                    "batch_size": None,
                    "learning_rate": None,
                },
            )
            self.assertEqual(
                base_plan["metadata"]["training_run_id"],
                none_plan["metadata"]["training_run_id"],
            )

            # Overrides change hyperparameters and drift run identity
            overridden_plan = prepare_training_plan(
                data_root=root,
                annotation_root=root / "outputs" / "annotations",
                output_root=root / "outputs",
                annotation_run_id=run_id,
                model="mimo_vl",
                run_tag="override-test",
                seed=42,
                resume=True,
                hyperparameter_overrides={
                    "batch_size": 2,
                    "gradient_accumulation_steps": 8,
                    "learning_rate": 5e-5,
                    "epochs": 5,
                    "eval_batch_size": 2,
                    "best_epoch_primary_metric": "mean_iou",
                },
            )
            hp = overridden_plan["metadata"]["hyperparameters"]
            self.assertEqual(hp["batch_size"], 2)
            self.assertEqual(hp["gradient_accumulation_steps"], 8)
            self.assertEqual(hp["learning_rate"], 5e-5)
            self.assertEqual(hp["epochs"], 5)
            self.assertEqual(hp["eval_batch_size"], 2)
            self.assertEqual(hp["best_epoch_primary_metric"], "mean_iou")
            self.assertNotEqual(
                base_plan["metadata"]["training_run_id"],
                overridden_plan["metadata"]["training_run_id"],
            )

    def test_invalid_overrides_raise_clear_value_error(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_invalid_override"
            _write_fixture_images(root)
            for split, scene in (("train", "001"), ("val", "002")):
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    _artifact(split, scene, run_id),
                )

            for key, val in [
                ("batch_size", 0),
                ("gradient_accumulation_steps", -1),
                ("learning_rate", -0.01),
                ("epochs", 0),
                ("eval_batch_size", 0),
                ("best_epoch_primary_metric", "invalid_metric"),
                ("unsupported_key", 123),
            ]:
                with self.subTest(key=key, val=val):
                    with self.assertRaises(ValueError):
                        prepare_training_plan(
                            data_root=root,
                            annotation_root=root / "outputs" / "annotations",
                            output_root=root / "outputs",
                            annotation_run_id=run_id,
                            model="mimo_vl",
                            run_tag="invalid",
                            seed=42,
                            resume=True,
                            hyperparameter_overrides={key: val},
                        )


class BestMetricTests(unittest.TestCase):
    """``--best-metric`` may be a minimized metric (``val_loss``) as well as a
    maximized one (accuracy / IoU); the two differ in source and direction."""

    def test_metric_direction_and_source(self):
        self.assertTrue(metric_improved("val_loss", 1.0, 2.0))
        self.assertFalse(metric_improved("val_loss", 2.0, 1.0))
        self.assertTrue(metric_improved("acc_at_0_5", 0.6, 0.5))
        self.assertFalse(metric_improved("acc_at_0_5", 0.4, 0.5))
        self.assertTrue(metric_improved("mean_iou", 0.4, 0.3))
        self.assertFalse(metric_improved("mean_iou", 0.3, 0.4))
        # A neutral incumbent must be worse than any real value in each direction.
        self.assertEqual(initial_best_metric_value("val_loss"), float("inf"))
        self.assertEqual(initial_best_metric_value("acc_at_0_5"), -1.0)
        self.assertEqual(initial_best_metric_value("mean_iou"), -1.0)
        epoch_metrics = {"acc_at_0_5": 0.5, "mean_iou": 0.3}
        self.assertEqual(
            candidate_metric_value("val_loss", epoch_metrics=epoch_metrics, val_loss=1.25),
            1.25,
        )
        self.assertEqual(
            candidate_metric_value("mean_iou", epoch_metrics=epoch_metrics, val_loss=1.25),
            0.3,
        )

    def test_val_loss_best_metric_runs_every_epoch_and_selects_a_best_adapter(self):
        """The grounding metric dict has no ``val_loss`` key, so this path used
        to raise KeyError once the first epoch finished."""
        calls: dict = {}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_best_metric"
            _write_fixture_images(root)
            for split, scene in (("train", "001"), ("val", "002")):
                atomic_write_json(
                    root / "outputs" / "annotations" / run_id / split / "approved.json",
                    _artifact(split, scene, run_id),
                )
            import sys as _sys
            import peft as _peft  # noqa: F401 - import before the torch shim
            import transformers as _transformers  # noqa: F401

            with mock.patch(
                "aicomp_grounding.training_core.get_adapter",
                return_value=_fake_adapter(calls),
            ) as adapter_patch, mock.patch(
                "aicomp_grounding.models.get_adapter", adapter_patch
            ), mock.patch.dict(
                _sys.modules, {"torch": _CpuTorchShim(torch)}
            ):
                plan = prepare_training_plan(
                    data_root=root,
                    annotation_root=root / "outputs" / "annotations",
                    output_root=root / "outputs",
                    annotation_run_id=run_id,
                    model="mimo_vl",
                    run_tag="best-metric-val-loss",
                    seed=42,
                    resume=True,
                    hyperparameter_overrides={
                        "epochs": 2,
                        "best_epoch_primary_metric": "val_loss",
                    },
                )
                completed = run_training(plan, data_root=root)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["best_metric"], "val_loss")
            # By construction the selected adapters's val_loss is the winner.
            self.assertEqual(completed["best_metric_value"], completed["best_val_loss"])
            best_dir = Path(completed["best_path"])
            self.assertTrue(best_dir.is_dir(), completed["best_path"])
            self.assertEqual(best_dir.parent.name, "best")
            manifest = load_json(best_dir / "adapter_manifest.json")
            self.assertEqual(manifest["val_loss"], completed["best_val_loss"])


if __name__ == "__main__":
    unittest.main()
