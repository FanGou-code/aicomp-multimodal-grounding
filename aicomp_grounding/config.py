"""Shared configuration for local checks and Modal jobs."""

from __future__ import annotations

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

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
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 2560 * 28 * 28
INFERENCE_COMPUTE_DTYPE = "bfloat16"

# Keep in sync with requirements-lock.txt when adding or upgrading packages.
MODAL_GPU_PACKAGES = (
    "transformers==4.57.3",
    "accelerate==1.14.0",
    "peft==0.19.1",
    "qwen-vl-utils==0.0.14",
    "pillow==12.1.0",
    "torch==2.13.0",
    "torchvision==0.28.0",
)

CHECKPOINT_VERSION = 5
ANNOTATION_PROTOCOL_VERSION = 12
PREPARATION_PROTOCOL_VERSION = 2
TRAINING_PROTOCOL_VERSION = 2
MAX_MODAL_CONTAINERS = 10

INFERENCE_SPLITS = frozenset({"train", "val", "test"})
ANNOTATION_SPLITS = frozenset({"train", "val"})
