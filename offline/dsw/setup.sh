#!/bin/bash
# ==============================================================================
# ModelScope DSW 极速换源与环境初始化脚本
# ==============================================================================
set -e

echo "🚀 [ModelScope DSW] 正在配置国内加速源并安装推理依赖..."

# 版本与 Modal 端 aicomp_grounding/config.py 的 MODAL_GPU_PACKAGES 严格对齐，
# 保证双端推理行为一致、环境可复现（DSW 镜像已预装 torch/torchvision/pillow，
# 此处不重复安装，避免覆盖镜像自带版本）。
pip install \
    transformers==4.57.3 \
    peft==0.19.1 \
    accelerate==1.14.0 \
    qwen-vl-utils==0.0.14 \
    -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo "✅ [ModelScope DSW] 环境依赖安装完成！"
