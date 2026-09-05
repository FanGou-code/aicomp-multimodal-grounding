"""Shared configuration for local checks and Modal jobs.

Model-specific identity (model ids, revisions, pixel budgets) lives in
``aicomp_grounding.models`` adapters; this module keeps only cross-model
constants.
"""

from __future__ import annotations

import platform
from importlib import metadata as importlib_metadata

# Annotation generation constants (teacher model identity, API endpoint,
# rate budgets) live in the external query-foundry repository. This module
# retains only ANNOTATION_PROTOCOL_VERSION — the product contract version the
# training side enforces when loading approved.json artifacts.

DATA_ROOT = "/data/data"
INFERENCE_COMPUTE_DTYPE = "bfloat16"
RUNTIME_PYTHON_VERSION = platform.python_version()

# Generic inference pixel-budget defaults recorded in run metadata (Qwen
# processor units, 28x28 per patch). Model adapters may override via kwargs.
INFERENCE_DEFAULT_MIN_PIXELS = 256 * 28 * 28
INFERENCE_DEFAULT_MAX_PIXELS = 3072 * 28 * 28

# Local development/validation dependencies are pinned in requirements-lock.txt.
# MODAL_GPU_PACKAGES is the separate Modal GPU runtime package set; pins are
# fallbacks only — current_runtime_packages() records actually-installed versions.
MODAL_GPU_PACKAGES = (
    "transformers==5.14.1",
    "accelerate==1.14.0",
    "peft==0.19.1",
    "qwen-vl-utils==0.0.14",
    "pillow==12.1.0",
    "torch==2.13.0",
    "torchvision==0.28.0",
)


def current_runtime_packages() -> list[str]:
    """Return installed runtime versions, with pinned fallbacks."""
    packages = list(MODAL_GPU_PACKAGES)

    for index, package in enumerate(packages):
        distribution, separator, _ = package.partition("==")
        if not separator:
            continue
        try:
            version = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            continue
        packages[index] = f"{distribution}=={version}"

    try:
        import torch
    except Exception:
        torch = None
    if torch is not None:
        packages = [
            f"torch=={torch.__version__}" if package.startswith("torch==") else package
            for package in packages
        ]
        try:
            hip_version = torch.version.hip
        except AttributeError:
            hip_version = None
        if hip_version:
            packages.append(f"hip=={hip_version}")

    return packages

CHECKPOINT_VERSION = 5
ANNOTATION_PROTOCOL_VERSION = 12
PREPARATION_PROTOCOL_VERSION = 2
TRAINING_PROTOCOL_VERSION = 2
MAX_MODAL_CONTAINERS = 10

INFERENCE_SPLITS = frozenset({"train", "val", "test"})
