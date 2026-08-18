"""Cloud (Modal) H100 LoRA training shell around the shared training core."""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# This entrypoint lives in cloud/; make the repository root importable so the
# shared library resolves both under `modal run` and in offline unit tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_grounding.config import DATA_ROOT, MODAL_GPU_PACKAGES
from aicomp_grounding.io import load_json
from aicomp_grounding.training_core import (
    SEED,
    _load_training_state,
    persist_training_plan,
    prepare_training_plan,
    run_training,
)
from aicomp_grounding.training_state import validate_loaded_training_state

dataset_volume = modal.Volume.from_name("rgbdt-dataset")
model_volume = modal.Volume.from_name("hf-model-cache")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(*MODAL_GPU_PACKAGES)
    .add_local_python_source("aicomp_grounding")
)

app = modal.App("rgbdt-visual-grounding", image=image)


@app.function(cpu=4.0, memory=8192, timeout=3600, volumes={"/data": dataset_volume})
def preflight_training_environment(
    annotation_run_id: str,
    run_tag: str = "",
    seed: int = SEED,
    resume: bool = True,
    smoke_test: bool = False,
    deep_verify_images: bool = False,
    use_all_data: bool = False,
    val_scenes: int = 40,
) -> dict:
    dataset_volume.reload()
    plan = prepare_training_plan(
        data_root=DATA_ROOT,
        annotation_run_id=annotation_run_id,
        run_tag=run_tag,
        seed=seed,
        resume=resume,
        smoke_test=smoke_test,
        verify_images=deep_verify_images,
        use_all_data=use_all_data,
        val_scenes=val_scenes,
    )
    if plan["resume_checkpoint"]:
        import torch

        checkpoint_path = Path(plan["resume_checkpoint"])
        state = load_json(checkpoint_path / "state.json")
        binary_state = _load_training_state(
            torch, checkpoint_path / "training_state.pt"
        )
        validate_loaded_training_state(binary_state, state, plan["metadata"])
    if not smoke_test:
        persist_training_plan(plan, commit_hook=dataset_volume.commit)
    return plan


@app.function(
    gpu="H100",
    cpu=4.0,
    memory=32768,
    timeout=43200,
    volumes={
        "/data": dataset_volume,
        "/root/.cache/huggingface": model_volume,
    },
)
def train(training_plan: dict) -> dict:
    dataset_volume.reload()
    return run_training(
        training_plan,
        data_root=DATA_ROOT,
        commit_hook=dataset_volume.commit,
    )


@app.local_entrypoint()
def main(
    annotation_run_id: str = "",
    run_tag: str = "",
    seed: int = SEED,
    resume: bool = True,
    preflight_only: bool = False,
    smoke_test: bool = False,
    deep_verify_images: bool = False,
    use_all_data: bool = False,
    val_scenes: int = 40,
) -> dict:
    if not annotation_run_id:
        raise ValueError("annotation_run_id is required; bare train.json/val.json are prohibited")
    plan = preflight_training_environment.remote(
        annotation_run_id=annotation_run_id,
        run_tag=run_tag,
        seed=seed,
        resume=resume,
        smoke_test=smoke_test,
        deep_verify_images=deep_verify_images,
        use_all_data=use_all_data,
        val_scenes=val_scenes,
    )
    if preflight_only:
        print(
            f"Training preflight passed: {plan['metadata']['training_run_id']} | "
            f"already completed: {plan['skip_training']}",
            flush=True,
        )
        return plan

    if plan["skip_training"] and not smoke_test:
        print(f"Training run already completed: {plan['metadata']['training_run_id']}")
        return plan["completed"]

    result = train.remote(plan)
    if smoke_test:
        if result.get("status") != "smoke_passed":
            raise RuntimeError("Training smoke test did not return a passing result")
        print(
            f"Training smoke passed: train_loss={result['train_loss']:.4f}, "
            f"val_loss={result['val_loss']:.4f}"
        )
        return result
    print(f"Training run_id: {result['metadata']['training_run_id']}")
    print(f"Best adapter: {result['best_path']}")
    print(f"Last adapter: {result['last_path']}")
    return result
