"""Shared configuration for local runs and run-metadata recording.

Model-specific identity (model ids, revisions, pixel budgets) lives in
``aicomp_grounding.serving.models`` adapters; this module keeps only cross-model
constants.
"""

from __future__ import annotations

import platform
from importlib import metadata as importlib_metadata

INFERENCE_COMPUTE_DTYPE = "bfloat16"
RUNTIME_PYTHON_VERSION = platform.python_version()

# Generic inference pixel-budget defaults recorded in run metadata (Qwen
# processor units, 28x28 per patch). Model adapters may override via kwargs.
INFERENCE_DEFAULT_MIN_PIXELS = 256 * 28 * 28
INFERENCE_DEFAULT_MAX_PIXELS = 3072 * 28 * 28

# Dependency declarations live in pyproject.toml (single source). This tuple is
# only the fallback name/version set recorded into run metadata; the four model
# libraries are kept identical to pyproject by tests/test_env_contract.py.
# current_runtime_packages() overwrites each entry with the actually-installed
# version, so the pinned values below matter only when a package is absent.
RUNTIME_PACKAGES = (
    "transformers==5.15.1",
    "accelerate==1.14.0",
    "peft==0.20.0",
    "qwen-vl-utils==0.0.14",
    "pillow==12.1.0",
    "torch==2.13.0",
    "torchvision==0.28.0",
)


def current_runtime_packages() -> list[str]:
    """Return installed runtime versions, with pinned fallbacks."""
    packages = list(RUNTIME_PACKAGES)

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

INFERENCE_SPLITS = frozenset({"train", "val", "test"})
