"""Inference must bind train/val runs to approved open-weights annotations."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import run_inference
from aicomp_grounding.annotation_state import (
    ANNOTATION_MODE,
    ASSIGNMENT_POLICY,
    approved_dataset_fingerprint,
)
from aicomp_grounding.config import ANNOTATION_PROTOCOL_VERSION
from aicomp_grounding.io import atomic_write_json
from aicomp_grounding.sequence import source_fingerprint
from aicomp_grounding.training_state import build_training_metadata


def _approved_artifact(
    run_id: str,
    *,
    split: str = "val",
    scene: str = "001",
) -> dict:
    data = {
        f"{scene}_00000001": {
            "visible": f"Train/{scene}/color/00000001.png",
            "infrared": f"Train/{scene}/infrared/00000001.png",
            "depth": f"Processed/Train/{scene}/depth_jet/00000001.png",
            "query": "The pedestrian wearing a bright yellow jacket beside the railing",
            "bbox": [0.1, 0.1, 0.4, 0.6],
            "width": 1920,
            "height": 1080,
        }
    }
    return {
        "metadata": {
            "status": "approved",
            "protocol_version": ANNOTATION_PROTOCOL_VERSION,
            "run_id": run_id,
            "split": split,
            "source_fingerprint": source_fingerprint(data),
            "preparation_fingerprint": "preparation-bytes",
            "image_fingerprint": f"{split}-image-bytes",
            "dataset_fingerprint": approved_dataset_fingerprint(data),
            "sample_count": 1,
            "sequence_count": 1,
            "prompt_hash": "annotation-prompt",
            "provenance": {
                "source_type": "hosted_open_weights",
                "provider": "zhipu",
                "api_base_url": "https://api.siliconflow.cn/v1",
                "annotator_model": "open-model",
                "annotator_revision": "fixed-revision",
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
                "generated_samples": 1,
            },
        },
        "data": data,
    }


def _write_approved(
    root: Path,
    run_id: str,
    artifact: dict,
    *,
    destination_split: str | None = None,
) -> Path:
    path = (
        root
        / "outputs"
        / "annotations"
        / run_id
        / (destination_split or artifact["metadata"]["split"])
        / "approved.json"
    )
    atomic_write_json(path, artifact)
    return path


class ApprovedInferenceSourceTests(unittest.TestCase):
    def test_lora_provenance_requires_matching_approved_open_weights_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round1"
            train = _approved_artifact(run_id, split="train", scene="001")
            val = _approved_artifact(run_id, split="val", scene="002")
            _write_approved(root, run_id, train)
            _write_approved(root, run_id, val)
            metadata = build_training_metadata(
                annotation_run_id=run_id,
                train_artifact=train,
                val_artifact=val,
                model_name=run_inference.MODEL_NAME,
                model_revision=run_inference.MODEL_REVISION,
                grounding_prompt_hash=run_inference._prompt_hash(False),
                hyperparameters={"epochs": 3},
                seed=42,
                run_tag="round1",
            )
            adapter = (
                root
                / "output_lora"
                / metadata["training_run_id"]
                / "checkpoints"
                / "epoch_01"
            )
            adapter.mkdir(parents=True)
            atomic_write_json(adapter / "adapter_config.json", {"r": 16})
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            atomic_write_json(
                adapter / "adapter_manifest.json",
                {"metadata": metadata, "epoch": 1, "val_loss": 0.5},
            )

            resolved = run_inference.resolve_lora_path(adapter, root)
            self.assertEqual(
                run_inference._validate_adapter_provenance(
                    root,
                    resolved,
                    split="test",
                    annotation_run_id="",
                ),
                metadata,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_inference._validate_adapter_provenance(
                    root,
                    resolved,
                    split="val",
                    annotation_run_id="annot_round2",
                )

    def test_val_loads_and_validates_approved_artifact_without_root_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round1"
            artifact = _approved_artifact(run_id)
            _write_approved(root, run_id, artifact)
            atomic_write_json(root / "val.json", {"legacy": {"query": "closed API label"}})

            loaded = run_inference._load_dataset(root, "val", run_id)

            self.assertEqual(loaded, artifact["data"])
            self.assertNotIn("legacy", loaded)

    def test_wrong_split_run_or_tampered_approved_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_id = "annot_round1"

            wrong_split = _approved_artifact(run_id, split="train")
            path = _write_approved(
                root,
                run_id,
                wrong_split,
                destination_split="val",
            )
            with self.assertRaisesRegex(ValueError, "split or run ID"):
                run_inference._load_dataset(root, "val", run_id)

            wrong_run = _approved_artifact("annot_other")
            atomic_write_json(path, wrong_run)
            with self.assertRaisesRegex(ValueError, "split or run ID"):
                run_inference._load_dataset(root, "val", run_id)

            tampered = copy.deepcopy(_approved_artifact(run_id))
            tampered["data"]["001_00000001"]["query"] = (
                "The pedestrian wearing a vivid orange jacket beside the railing"
            )
            atomic_write_json(path, tampered)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_inference._load_dataset(root, "val", run_id)

    def test_annotation_run_changes_plan_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for run_id in ("annot_round1", "annot_round2"):
                _write_approved(root, run_id, _approved_artifact(run_id))

            first = run_inference.prepare_inference_plan(
                data_root=root,
                split="val",
                annotation_run_id="annot_round1",
                verify_images=False,
            )
            second = run_inference.prepare_inference_plan(
                data_root=root,
                split="val",
                annotation_run_id="annot_round2",
                verify_images=False,
            )

            self.assertNotEqual(first["metadata"]["run_id"], second["metadata"]["run_id"])

    def test_retry_rejects_a_different_annotation_run(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for run_id in ("annot_round1", "annot_round2"):
                _write_approved(root, run_id, _approved_artifact(run_id))
            base = run_inference.prepare_inference_plan(
                data_root=root,
                split="val",
                annotation_run_id="annot_round1",
                verify_images=False,
            )
            checkpoint = (
                root
                / "outputs"
                / "inference"
                / base["metadata"]["run_id"]
                / "checkpoint.json"
            )
            atomic_write_json(
                checkpoint,
                {
                    "metadata": base["metadata"],
                    "predictions": {base["result_keys"][0]: None},
                },
            )

            with self.assertRaisesRegex(ValueError, "annotation run"):
                run_inference.prepare_inference_plan(
                    data_root=root,
                    split="val",
                    annotation_run_id="annot_round2",
                    retry_failed=True,
                    base_run_id=base["metadata"]["run_id"],
                    verify_images=False,
                )


if __name__ == "__main__":
    unittest.main()
