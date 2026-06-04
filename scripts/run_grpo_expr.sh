#!/bin/bash
# ============================================================
# 表达式方案 GRPO 训练
# 用法: bash scripts/run_grpo_expr.sh
# 前置条件: 先完成 run_sft_expr_safe.sh
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

echo "=============================="
echo " 表达式 GRPO 训练"
echo "=============================="

if [ ! -d "outputs/checkpoints/sft_expr_safe/best" ]; then
    echo "错误: 表达式 SFT checkpoint 不存在"
    echo "请先运行: bash scripts/run_sft_expr_safe.sh"
    exit 1
fi

$PYTHON -m src.training.grpo_expr_trainer --config configs/grpo_expr.yaml

echo "=============================="
echo " 表达式 GRPO 训练完成！"
echo " Checkpoint: outputs/checkpoints/grpo_expr/best"
echo "=============================="
