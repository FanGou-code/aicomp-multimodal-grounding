#!/usr/bin/env bash
# Install the pinned torch/torchvision, then the project.
#
# Usage:
#   bash tools/setup_cuda.sh
#
# Requires CUDA 13.x (driver >= R580). torch 2.14.0 on PyPI is the CUDA 13
# build: it pulls cuda-toolkit==13.0.3 and the nvidia-*-cu13 libraries.
#
# Packages come from mirrors.ustc.edu.cn, not mirrors.aliyun.com: aliyun
# throttles HTTP/1.1 to ~0.04 MB/s and pip has no HTTP/2, which turns the
# 553 MB nvidia-cudnn wheel into hours. Override with PIP_INDEX_MIRROR.

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
