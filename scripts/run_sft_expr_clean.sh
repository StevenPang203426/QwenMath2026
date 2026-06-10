#!/bin/bash
# ============================================================
# 表达式路线 SFT（clean 安全表达式数据）
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

if [ ! -f "data/splits/train_expr_clean_train.json" ]; then
    echo "错误: clean expression split 不存在"
    echo "请先运行: bash scripts/run_expr_training_data.sh clean"
    exit 1
fi

$PYTHON -m src.training.sft_trainer --config configs/sft_expr_clean.yaml

echo "Checkpoint: outputs/checkpoints/sft_expr_clean/best"
