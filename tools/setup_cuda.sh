#!/usr/bin/env bash
# CUDA environment setup: install the pinned CUDA wheels, then the project.
#
# Usage:
#   bash tools/setup_cuda.sh
#
# Mirror choice: PyPI carries the CUDA builds of torch/torchvision directly
# (torch 2.14.0 on Linux pulls cuda-toolkit==13.0.3, nvidia-cudnn-cu13,
# nvidia-nccl-cu13), so no separate pytorch wheel index is needed. Package
# downloads go to mirrors.ustc.edu.cn because mirrors.aliyun.com throttles
# HTTP/1.1 to ~0.04 MB/s and pip speaks HTTP/1.1 only; USTC serves the same
# files at ~8 MB/s over HTTP/1.1. Override with PIP_INDEX_MIRROR.
#
# A managed image that already ships a matching torch short-circuits: pip
# reports the requirement as already satisfied and downloads nothing for it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MIRROR="${PIP_INDEX_MIRROR:-https://mirrors.ustc.edu.cn/pypi/simple/}"

echo "[setup] installing torch==2.14.0 torchvision==0.29.0 from ${MIRROR}"
pip install --index-url "$MIRROR" torch==2.14.0 torchvision==0.29.0
pip install --index-url "$MIRROR" -e .

python - <<'PY'
import sys

import torch
import transformers
import yaml  # noqa: F401

assert torch.__version__.split("+")[0] == "2.14.0", torch.__version__
assert torch.cuda.is_available(), "CUDA is not available"
assert transformers.__version__ == "5.17.0", transformers.__version__

name = torch.cuda.get_device_name(0)
total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"[setup] torch {torch.__version__} | CUDA {torch.version.cuda} | {name} ({total_gib:.1f} GiB)")
print(f"[setup] transformers {transformers.__version__} | python {sys.version.split()[0]}")
print("[setup] environment OK")
PY
