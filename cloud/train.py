"""Modal training shell — single H100, volume-mounted, offline-identical semantics.

Same hardcoded envelope as cloud/infer.py (H100 / 8 cpu / 32GiB RAM — the
proven Iteration-02 configuration). Hyperparameters are NOT hardcoded here:
they live in each adapter's ``training_hyperparameters()``. The volume commit
hook is invoked after every durable write (run plan, checkpoints) so a
preempted container resumes exactly where it stopped.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tomllib

import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VOLUME_ROOT = "/mnt/workspace"
REPO_MOUNT = "/root/aicomp"


def _project_dependencies() -> list[str]:
    """Read the single dependency source (pyproject.toml) for the image build."""
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as file:
        return tomllib.load(file)["project"]["dependencies"]


# Dependencies come from pyproject.toml (single source); torch is a range there.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_project_dependencies())
    .add_local_dir(
        PROJECT_ROOT / "aicomp_grounding", remote_path=f"{REPO_MOUNT}/aicomp_grounding"
    )
    .add_local_dir(PROJECT_ROOT / "offline", remote_path=f"{REPO_MOUNT}/offline")
    .add_local_file(PROJECT_ROOT / "pyproject.toml", remote_path=f"{REPO_MOUNT}/pyproject.toml")
)

volume = modal.Volume.from_name("aicomp", create_if_missing=True)
app = modal.App("aicomp-train", image=image, volumes={VOLUME_ROOT: volume})

TRAINING_DEFAULTS: dict = {
    "model": "qwen3vl",
    "model_path": None,
    "data_dir": f"{VOLUME_ROOT}/data",
    "annotation_root": f"{VOLUME_ROOT}/outputs/annotations",
    "output_root": f"{VOLUME_ROOT}/outputs",
    "run_tag": "",
    "seed": 42,
    "resume": True,
    "smoke_test": False,
    "preflight_only": False,
    "deep_verify_images": False,
    "num_workers": 4,
    "checkpoint_interval": 20,
}


@app.function(gpu="H100", cpu=8, memory=32 * 1024, timeout=24 * 3600)
def train_job(args_dict: dict) -> dict:
    os.chdir(REPO_MOUNT)
    sys.path.insert(0, REPO_MOUNT)
    from offline.train import run_cli

    if not args_dict.get("annotation_run_id"):
        raise ValueError("annotation_run_id is required for training")
    merged = {
        **TRAINING_DEFAULTS,
        **{key: value for key, value in args_dict.items() if value is not None},
    }
    ns = argparse.Namespace(project_root=VOLUME_ROOT, **merged)
    result = run_cli(ns, commit_hook=volume.commit)
    volume.commit()
    return {
        "training_run_id": result["metadata"]["training_run_id"],
        "status": result.get("status", "completed"),
        "best_path": result.get("best_path"),
        "last_path": result.get("last_path"),
    }


@app.local_entrypoint()
def train(
    annotation_run_id: str,
    model: str = "qwen3vl",
    model_path: str | None = None,
    run_tag: str = "",
    seed: int = 42,
    resume: bool = True,
    smoke_test: bool = False,
    preflight_only: bool = False,
    deep_verify_images: bool = False,
    num_workers: int = 4,
    checkpoint_interval: int = 20,
):
    """modal run cloud/train.py --annotation-run-id annot_dc189f029d962b27 --smoke-test"""
    train_job.remote(dict(locals()))
