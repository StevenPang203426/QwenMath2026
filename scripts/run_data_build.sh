#!/bin/bash
# ============================================================
# 方案2 前置: 数据构建
# 调用 DeepSeek V4 API 生成 CoT 训练数据
#
# 用法:
#   bash scripts/run_data_build.sh          # 默认先小批量测试 20 条
#   bash scripts/run_data_build.sh 50       # 测试 50 条
#   bash scripts/run_data_build.sh all      # 全量生成（12000 条）
# ============================================================
set -e

# 需要设置环境变量 DEEPSEEK_API_KEY
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "错误: 请先设置环境变量 DEEPSEEK_API_KEY"
    echo "  export DEEPSEEK_API_KEY=your_api_key"
    exit 1
fi

# 解析参数：数量限制
LIMIT=${1:-20}
if [ "$LIMIT" = "all" ]; then
    LIMIT=0
    LIMIT_FLAG=""
    echo "=============================="
    echo " CoT 数据构建（全量）"
    echo "=============================="
else
    LIMIT_FLAG="--limit $LIMIT"
    echo "=============================="
    echo " CoT 数据构建（测试: ${LIMIT} 条）"
    echo "=============================="
fi

# 1. 生成 CoT 数据（含正确推理 + 错误推理）
echo "[1/3] 调用 DeepSeek API 生成 CoT 数据..."
python -m src.data.data_builder \
    --input data/raw/train.json \
    --output data/processed/train_cot_raw.json \
    --api_key "$DEEPSEEK_API_KEY" \
    --model deepseek-v4-pro \
    --workers 4 \
    --generate_wrong \
    $LIMIT_FLAG

# 2. 转换为 SFT 格式（过滤答案不匹配的样本）
echo "[2/3] 转换为 SFT 训练格式..."
python -m src.data.preprocessor \
    --action sft \
    --input data/processed/train_cot_raw.json \
    --output data/processed/train_cot.json

# 3. 转换为 DPO 格式
echo "[3/3] 转换为 DPO 偏好对格式..."
python -m src.data.preprocessor \
    --action dpo \
    --input data/processed/train_cot_raw.json \
    --output data/processed/train_dpo.json

echo "=============================="
echo " 数据构建完成！"
echo " CoT SFT 数据: data/processed/train_cot.json"
echo " DPO 偏好数据: data/processed/train_dpo.json"
if [ "$LIMIT" != "0" ]; then
    echo ""
    echo " 当前为测试模式（${LIMIT} 条），确认无误后运行:"
    echo "   bash scripts/run_data_build.sh all"
fi
echo "=============================="
