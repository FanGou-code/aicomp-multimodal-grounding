#!/usr/bin/env bash
# CUDA environment setup: install the pinned CUDA wheel variants of
# torch/torchvision for this machine's driver, then install the project.
#
# Usage:
#   bash tools/setup_cuda.sh
#
# The wheel variant is selected from the driver's reported CUDA support:
#   driver >= 13.0 -> cu130 wheels
#   driver >= 12.6 -> cu126 wheels
# Both variants install the same torch version declared in pyproject.toml.
# torch 2.14.0 publishes only cu126 / cu130 / cu132; there is no cu128 or
# cu129 build, so drivers older than 12.6 have no matching wheel.
#
# torch/torchvision come from mirrors.aliyun.com as a flat wheel directory.
# They are fetched with curl rather than left to pip: pip resolves the same
# URL but has been observed at ~50 kB/s against ~10 MB/s for curl on the same
# host, which turns a 555 MB wheel into hours. curl also resumes, so an
# interrupted run continues instead of restarting.
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
    WHEEL_TAG="cu130"
elif [ "$CUDA_MAJOR" -ge 12 ]; then
    WHEEL_TAG="cu126"
else
    echo "[setup] CUDA ${CUDA_MAJOR}.x has no torch 2.14.0 wheel (cu126/cu130/cu132 only)." >&2
    exit 1
fi
WHEEL_DIR="${PYTORCH_WHEEL_DIR:-https://mirrors.aliyun.com/pytorch-wheels/${WHEEL_TAG}}"
CACHE_DIR="${TORCH_WHEEL_CACHE:-$HOME/.cache/torch-wheels}/${WHEEL_TAG}"
PY_TAG="cp$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
case "$(uname -m)" in
    x86_64 | amd64) WHEEL_ARCH="x86_64" ;;
    aarch64 | arm64) WHEEL_ARCH="aarch64" ;;
    *)
        echo "[setup] Unsupported architecture: $(uname -m)" >&2
        exit 1
        ;;
esac
WHEEL_PLATFORM="manylinux_2_28_${WHEEL_ARCH}"

mkdir -p "$CACHE_DIR"
echo "[setup] Driver supports CUDA ${CUDA_MAJOR}.x; caching ${WHEEL_TAG} wheels in ${CACHE_DIR}"

# Mirrors percent-encode the local version separator: torch-2.14.0%2Bcu130-...
# pip rejects '%2B' in a filename, so each wheel is stored under its real name.
fetch_wheel() {
    local name="$1"
    local dest="$CACHE_DIR/$name"
    if [ -s "$dest" ]; then
        echo "[setup] cached $(basename "$dest")"
        return 0
    fi
    local encoded="${name/+/%2B}"
    echo "[setup] downloading ${name}"
    # Download under the encoded name and rename on completion: an interrupted
    # transfer leaves the encoded file behind, so the `-s "$dest"` check above
    # only skips complete wheels, and -C - resumes the partial one.
    curl -fL -C - -o "$CACHE_DIR/$encoded" "$WHEEL_DIR/$encoded"
    mv "$CACHE_DIR/$encoded" "$dest"
}

fetch_wheel "torch-2.14.0+${WHEEL_TAG}-${PY_TAG}-${PY_TAG}-${WHEEL_PLATFORM}.whl"
fetch_wheel "torchvision-0.29.0+${WHEEL_TAG}-${PY_TAG}-${PY_TAG}-${WHEEL_PLATFORM}.whl"

# Dependencies (filelock, sympy, triton, nvidia-*) come from the PyPI mirror.
pip install \
    --find-links "$CACHE_DIR" \
    --index-url https://mirrors.aliyun.com/pypi/simple/ \
    torch==2.14.0 torchvision==0.29.0
pip install -e .

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
