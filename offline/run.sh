#!/bin/bash
# ==============================================================================
# ModelScope DSW 一键全自动离线推理与比赛打包脚本
# ==============================================================================
set -e

# 参数 1: LoRA 权重目录 (默认: best/epoch_03)
LORA_PATH=${1:-"best/epoch_03"}

# 参数 2: 数据集根目录 (默认: data)
DATA_DIR=${2:-"data"}

# 参数 3: 基础模型路径或 HuggingFace/ModelScope ID (默认: Qwen/Qwen3-VL-8B-Instruct)
MODEL_PATH=${3:-"Qwen/Qwen3-VL-8B-Instruct"}

# 输出目录
OUTPUT_DIR="outputs/inference"

echo "=============================================================================="
echo "🚀 [ModelScope DSW] 启动离线推理与比赛提交包生成"
echo "  Base Model: ${MODEL_PATH}"
echo "  LoRA Path:  ${LORA_PATH}"
echo "  Data Dir:   ${DATA_DIR}"
echo "  Output Dir: ${OUTPUT_DIR}"
echo "=============================================================================="

# 检查测试集文件是否存在
TEST_JSON="${DATA_DIR}/Test/queries/queries.json"
if [ ! -f "${TEST_JSON}" ]; then
    echo "⚠️ 未在 ${TEST_JSON} 找到官方测试集文件，尝试查找 ${DATA_DIR}/test.json..."
    if [ -f "${DATA_DIR}/test.json" ]; then
        TEST_JSON="${DATA_DIR}/test.json"
    else
        echo "❌ 错误: 未找到测试集 queries.json，请确认数据已上传至 ${DATA_DIR}/"
        exit 1
    fi
fi

# 执行推理（在仓库根目录运行本脚本）
python offline/infer.py \
    --model-path "${MODEL_PATH}" \
    --test-json "${TEST_JSON}" \
    --data-dir "${DATA_DIR}" \
    --lora-path "${LORA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-save 50

echo "=============================================================================="
echo "🎉 [ModelScope DSW] 推理与打包已全部完成！"
echo "  提交包位置: ${OUTPUT_DIR}/<run_id>/submission.zip"
echo "=============================================================================="
