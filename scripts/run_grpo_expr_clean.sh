#!/bin/bash
# ============================================================
# 表达式路线 GRPO 主线
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

if [ ! -d "outputs/checkpoints/sft_expr_clean/best" ]; then
    echo "错误: outputs/checkpoints/sft_expr_clean/best 不存在"
    echo "请先运行: bash scripts/run_sft_expr_clean.sh"
    exit 1
fi

$PYTHON -m src.training.grpo_expr_trainer --config configs/grpo_expr_clean.yaml

echo "Checkpoint: outputs/checkpoints/grpo_expr_clean/best"
