#!/bin/bash
# ============================================================
# 表达式方案 DPO→GRPO 训练
# 用法: bash scripts/run_grpo_expr_from_dpo.sh
# 前置条件: 先完成 run_dpo_expr.sh
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

echo "=============================="
echo " 表达式 DPO→GRPO 训练"
echo "=============================="

if [ ! -d "outputs/checkpoints/dpo_expr/best" ]; then
    echo "错误: 表达式 DPO checkpoint 不存在"
    echo "请先运行: bash scripts/run_dpo_expr.sh"
    exit 1
fi

$PYTHON -m src.training.grpo_expr_trainer --config configs/grpo_expr_from_dpo.yaml

echo "=============================="
echo " 表达式 DPO→GRPO 训练完成！"
echo " Checkpoint: outputs/checkpoints/grpo_expr_from_dpo/best"
echo "=============================="
