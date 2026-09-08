"""Modal inference shell — single H100, volume-mounted, offline-identical semantics.

Hardcoded per campaign decision (2026-09-01): gpu="H100", cpu=8,
memory=32GiB — the envelope proven for 8B training/inference on this stack
(handoff Iteration 02). Everything model-specific (weights, LoRA, params)
arrives via `modal run` flags; the volume layout mirrors the DSW
`/mnt/workspace` contract so CLI arguments are identical to offline/.
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


# Dependencies come from pyproject.toml (single source). torch is declared as
# a range there, so Modal installs the latest stable CUDA build from PyPI while
# DSW keeps its ROCm build — the same file, platform-flavoured at install time.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*_project_dependencies())
    # Runtime-uploaded on every `modal run` (not baked into the image), so
    # code edits propagate without a rebuild. data/weights/outputs live on
    # the volume, never inside this upload.
    .add_local_dir(
        PROJECT_ROOT / "aicomp_grounding", remote_path=f"{REPO_MOUNT}/aicomp_grounding"
    )
    .add_local_dir(PROJECT_ROOT / "offline", remote_path=f"{REPO_MOUNT}/offline")
    .add_local_dir(PROJECT_ROOT / "envs", remote_path=f"{REPO_MOUNT}/envs")
    .add_local_file(PROJECT_ROOT / "pyproject.toml", remote_path=f"{REPO_MOUNT}/pyproject.toml")
)

volume = modal.Volume.from_name("aicomp", create_if_missing=True)
app = modal.App("aicomp", image=image, volumes={VOLUME_ROOT: volume})

INFERENCE_DEFAULTS: dict = {
    "model": "qwen3vl",
    "model_path": None,
    "test_json": f"{VOLUME_ROOT}/data/Test/queries/queries.json",
    "data_dir": f"{VOLUME_ROOT}/data",
    "output_dir": f"{VOLUME_ROOT}/outputs/inference",
    "lora_path": None,
    "annotation_run_id": "",
    "run_tag": "",
    "limit": 0,
    "resume": True,
    "batch_save": 50,
    "num_workers": 2,
    "batch_size": 4,
}


@app.function(gpu="H100", cpu=8, memory=32 * 1024, timeout=8 * 3600)
def infer_job(args_dict: dict) -> dict:
    os.chdir(REPO_MOUNT)
    sys.path.insert(0, REPO_MOUNT)
    from offline.infer import run_cli

    merged = {
        **INFERENCE_DEFAULTS,
        **{key: value for key, value in args_dict.items() if value is not None},
    }
    ns = argparse.Namespace(project_root=VOLUME_ROOT, **merged)
    summary = run_cli(ns)
    volume.commit()
    return {
        "run_id": summary["metadata"]["run_id"],
        "total_predictions": summary["total_predictions"],
        "valid_predictions": summary["valid_predictions"],
    }


@app.local_entrypoint()
def infer(
    model: str = "qwen3vl",
    model_path: str | None = None,
    test_json: str | None = None,
    data_dir: str | None = None,
    output_dir: str | None = None,
    lora_path: str | None = None,
    annotation_run_id: str = "",
    run_tag: str = "",
    limit: int = 0,
    resume: bool = True,
    num_workers: int = 2,
    batch_size: int = 4,
    batch_save: int = 50,
):
    """modal run cloud/infer.py --model qwen3vl --lora-path /mnt/workspace/outputs/..."""
    infer_job.remote(dict(locals()))
