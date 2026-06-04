#!/bin/bash
# ============================================================
# 表达式方案 SFT 训练（安全修复数据）
# 用法: bash scripts/run_sft_expr_safe.sh
# 前置条件: bash scripts/run_expr_repair.sh 生成 train_expr_safe.json
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

echo "=============================="
echo " 安全表达式 SFT 训练"
echo "=============================="

if [ ! -f "data/processed/train_expr_safe.json" ]; then
    echo "错误: data/processed/train_expr_safe.json 不存在"
    echo "请先运行: bash scripts/run_expr_repair.sh"
    exit 1
fi

$PYTHON -m src.training.sft_trainer --config configs/sft_expr_safe.yaml

echo "=============================="
echo " 安全表达式 SFT 训练完成！"
echo " Checkpoint: outputs/checkpoints/sft_expr_safe/best"
echo "=============================="
