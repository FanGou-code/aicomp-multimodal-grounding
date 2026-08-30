"""Shared configuration for local checks and Modal jobs.

Model-specific identity (model ids, revisions, pixel budgets) lives in
``aicomp_grounding.models`` adapters; this module keeps only cross-model
constants.
"""

from __future__ import annotations

import platform
from importlib import metadata as importlib_metadata

ANNOTATION_PROVIDER = "zhipu"
ANNOTATION_MODEL_NAME = "glm-4.6v"
ANNOTATION_MODEL_REVISION = "2025-12-08"
ANNOTATION_MODEL_WEIGHTS_URL = "https://huggingface.co/zai-org/GLM-4.6V"
ANNOTATION_MODEL_LICENSE = "MIT"
ANNOTATION_API_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
ANNOTATION_MAX_TOKENS = 256
ANNOTATION_TEMPERATURE = 0.2
# Zhipu primarily enforces per-account concurrency, not RPM/TPM. These defaults
# keep the client-side limiter from throttling below concurrency; tune via CLI
# (--requests-per-minute / --tokens-per-minute) against your account's limits
# shown at https://bigmodel.cn/usercenter/proj-mgmt/rate-limits.
ANNOTATION_REQUESTS_PER_MINUTE = 600
ANNOTATION_TOKENS_PER_MINUTE = 500_000
ANNOTATION_ESTIMATED_TOKENS_PER_REQUEST = 1_800

DATA_ROOT = "/data/data"
INFERENCE_COMPUTE_DTYPE = "bfloat16"
RUNTIME_PYTHON_VERSION = platform.python_version()

# Local development/validation dependencies are pinned in requirements-lock.txt.
# MODAL_GPU_PACKAGES is the separate Modal GPU runtime package set.
MODAL_GPU_PACKAGES = (
    "transformers==4.57.3",
    "accelerate==1.14.0",
    "peft==0.19.1",
    "qwen-vl-utils==0.0.14",
    "pillow==12.1.0",
    "torch==2.13.0",
    "torchvision==0.28.0",
)


def current_runtime_packages(model_name: str | None = None) -> list[str]:
    """Return installed runtime versions, with model-specific pinned fallbacks."""
    packages = list(MODAL_GPU_PACKAGES)
    if model_name == "qwen3_8":
        packages = [
            "transformers==5.8.0" if package.startswith("transformers==") else package
            for package in packages
        ]

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
ANNOTATION_SPLITS = frozenset({"train", "val"})
