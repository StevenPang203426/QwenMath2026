#!/bin/bash
# ============================================================
# 表达式路线 DPO→GRPO 保守加速
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

if [ ! -d "outputs/checkpoints/dpo_expr_clean/best" ]; then
    echo "错误: outputs/checkpoints/dpo_expr_clean/best 不存在"
    echo "请先运行: bash scripts/run_dpo_expr_clean.sh"
    exit 1
fi

$PYTHON -m src.training.grpo_expr_trainer --config configs/grpo_expr_from_dpo_clean_fast.yaml

echo "Checkpoint: outputs/checkpoints/grpo_expr_from_dpo_clean_fast/best"
