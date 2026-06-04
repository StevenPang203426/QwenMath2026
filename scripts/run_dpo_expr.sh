#!/bin/bash
# ============================================================
# 表达式方案 DPO 训练
# 用法: bash scripts/run_dpo_expr.sh
# 前置条件: train_expr_dpo.json + sft_expr_safe/best
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

echo "=============================="
echo " 表达式 DPO 训练"
echo "=============================="

if [ ! -f "data/processed/train_expr_dpo.json" ]; then
    echo "错误: data/processed/train_expr_dpo.json 不存在"
    echo "请先运行: bash scripts/run_expr_data_build.sh convert_dpo"
    exit 1
fi

if [ ! -d "outputs/checkpoints/sft_expr_safe/best" ]; then
    echo "错误: 安全表达式 SFT checkpoint 不存在"
    echo "请先运行: bash scripts/run_sft_expr_safe.sh"
    exit 1
fi

$PYTHON -m src.training.dpo_trainer --config configs/dpo_expr.yaml

echo "=============================="
echo " 表达式 DPO 训练完成！"
echo " Checkpoint: outputs/checkpoints/dpo_expr/best"
echo "=============================="
