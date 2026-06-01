#!/bin/bash
# ============================================================
# 表达式方案 SFT 训练
# 用法: bash scripts/run_sft_expr.sh
# 前置条件: 先运行 run_expr_data_build.sh convert_sft
# ============================================================
set -e

echo "=============================="
echo " 表达式 SFT 训练"
echo "=============================="

if [ ! -f "data/processed/train_expr.json" ]; then
    echo "错误: data/processed/train_expr.json 不存在"
    echo "请先运行: bash scripts/run_expr_data_build.sh convert_sft"
    exit 1
fi

python -m src.training.sft_trainer --config configs/sft_expr.yaml

echo "=============================="
echo " 表达式 SFT 训练完成！"
echo " Checkpoint: outputs/checkpoints/sft_expr/best"
echo "=============================="
