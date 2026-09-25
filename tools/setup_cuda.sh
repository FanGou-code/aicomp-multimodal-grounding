#!/usr/bin/env bash
# CUDA environment setup: install the pinned CUDA wheel variants of
# torch/torchvision for this machine's driver, then install the project.
#
# Usage:
#   bash tools/setup_cuda.sh
#
# The wheel variant is selected from the driver's reported CUDA support:
#   driver >= 13.0 -> cu130 wheels
#   driver <  13.0 -> cu128 wheels
# Both variants install the same torch version declared in pyproject.toml.
#
# A managed image that already ships a matching torch (for example a
# CUDA 13.x / Python 3.12 / torch 2.14 base image) short-circuits: pip
# reports the requirement as already satisfied and downloads nothing for it.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] nvidia-smi not found: this script targets CUDA GPU runtimes." >&2
    exit 1
fi

CUDA_MAJOR="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -n1 || true)"
if [ -z "$CUDA_MAJOR" ]; then
    echo "[setup] Could not read the CUDA version from nvidia-smi output." >&2
    exit 1
fi

if [ "$CUDA_MAJOR" -ge 13 ]; then
    WHEEL_INDEX="https://download.pytorch.org/whl/cu130"
    echo "[setup] Driver supports CUDA ${CUDA_MAJOR}.x; installing cu130 wheels."
else
    WHEEL_INDEX="https://download.pytorch.org/whl/cu128"
    echo "[setup] Driver supports CUDA ${CUDA_MAJOR}.x; installing cu128 wheels."
fi

pip install --quiet torch==2.14.0 torchvision==0.29.0 --index-url "$WHEEL_INDEX"
pip install --quiet -e .

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
