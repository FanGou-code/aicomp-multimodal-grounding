#!/bin/bash
# ==============================================================================
# ModelScope DSW 极速换源与环境初始化脚本
# ==============================================================================
set -e

echo "🚀 [ModelScope DSW] 正在配置国内加速源并安装推理依赖..."

# 使用阿里云官方 PyPI 镜像加速安装核心依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 确保 qwen-vl-utils 与 peft 处于最新兼容版本
pip install -U qwen-vl-utils peft transformers accelerate -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

echo "✅ [ModelScope DSW] 环境依赖安装完成！"
