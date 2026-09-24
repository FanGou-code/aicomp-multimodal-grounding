#!/usr/bin/env bash
# AMD ROCm performance environment for MI300X training/inference.
#
# Source this file before launching offline/train.py or offline/infer.py:
#   source offline/rocm_env.sh

export TORCH_BLAS_PREFER_HIPBLASLT=1
export GPU_MAX_HW_QUEUES=2
# Cache root for Triton JIT kernels and pip downloads. Override with
# AICOMP_CACHE_ROOT to put them on a persistent disk; the default keeps them in
# the user cache directory.
: "${AICOMP_CACHE_ROOT:=$HOME/.cache}"
export TRITON_CACHE_DIR="$AICOMP_CACHE_ROOT/.triton_cache"
export PIP_CACHE_DIR="$AICOMP_CACHE_ROOT/.pip_cache"
# TunableOp is intentionally disabled for this torch/ROCm stack: recent MI300X
# torch builds have reported memory-leak/OOM behavior when it is enabled at
# runtime. Re-enable only after an offline tuning run with a persistent result
# file, then benchmark the same workload before and after.
export PYTORCH_TUNABLE_OPS=0
export PYTORCH_TUNABLEOP_ENABLED=0
