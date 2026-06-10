#!/bin/bash
# ============================================================
# 表达式路线 DPO 消融
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

if [ ! -f "data/processed/train_expr_dpo_clean.json" ]; then
    echo "错误: data/processed/train_expr_dpo_clean.json 不存在"
    echo "请先运行: bash scripts/run_expr_training_data.sh dpo"
    exit 1
fi

if [ ! -d "outputs/checkpoints/sft_expr_clean/best" ]; then
    echo "错误: outputs/checkpoints/sft_expr_clean/best 不存在"
    echo "请先运行: bash scripts/run_sft_expr_clean.sh"
    exit 1
fi

$PYTHON -m src.training.dpo_trainer --config configs/dpo_expr_clean.yaml

echo "Checkpoint: outputs/checkpoints/dpo_expr_clean/best"
