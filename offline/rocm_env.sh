#!/usr/bin/env bash
# AMD ROCm performance environment for MI300X training/inference.
#
# Source this file before launching offline/train.py or offline/infer.py:
#   source offline/rocm_env.sh

export TORCH_BLAS_PREFER_HIPBLASLT=1
export PYTORCH_TUNABLE_OPS=1
export PYTORCH_TUNABLE_OPS_TUNING_ITER=200
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
