#!/bin/bash
# 魔搭 DSW 推理环境一键配置
# 目标镜像: ubuntu22.04-cuda12.1.0-py310-torch2.3.0-...
# 策略: 用镜像自带 torch(2.3.0)，不升级；其余推理库走国内源秒装
set -e

# 阿里云 PyPI 镜像（魔搭 DSW 在阿里云内网，速度最快）
MIRROR="https://mirrors.aliyun.com/pypi/simple/"

echo "=== 1. 检查 torch（镜像自带，不升级）==="
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU {torch.cuda.get_device_name(0)}')"
# 若这里报算子不兼容，再手动升级 torch（国内源）:
#   pip install -U torch==2.5.1 torchvision==0.20.1 -i "$MIRROR"

echo "=== 2. 安装推理核心库（国内源）==="
pip install -U -i "$MIRROR" \
  transformers==4.57.0 \
  peft \
  accelerate \
  pillow \
  safetensors \
  qwen-vl-utils

echo "=== 3. 清理冲突包 ==="
pip uninstall -y autoawq auto-awq 2>/dev/null || true

echo "=== 4. 验证 ==="
python3 - <<'EOF'
import torch
import transformers
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
print(f"OK | torch {torch.__version__} | transformers {transformers.__version__}")
print("peft / qwen-vl-utils / Qwen3VL 全部可用")
EOF

echo ""
echo "=== 环境配置完成 ==="
