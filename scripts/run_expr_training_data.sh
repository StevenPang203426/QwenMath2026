#!/bin/bash
# ============================================================
# 表达式路线 clean SFT/DPO 数据构建
# 用法:
#   bash scripts/run_expr_training_data.sh clean
#   bash scripts/run_expr_training_data.sh dpo
#   bash scripts/run_expr_training_data.sh all
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}
MODE=${1:-all}

echo "=============================="
echo " 表达式路线训练数据构建"
echo " 模式: $MODE"
echo "=============================="

if [ "$MODE" = "clean" ] || [ "$MODE" = "all" ]; then
    $PYTHON -m src.data.expression_training_data clean
    echo "clean SFT 数据: data/processed/train_expr_clean.json"
    echo "train split: data/splits/train_expr_clean_train.json"
    echo "val split: data/splits/train_expr_clean_val.json"
fi

if [ "$MODE" = "dpo" ] || [ "$MODE" = "all" ]; then
    if [ ! -f "data/processed/train_expr_clean.json" ]; then
        echo "错误: data/processed/train_expr_clean.json 不存在，请先运行 clean"
        exit 1
    fi
    $PYTHON -m src.data.expression_training_data dpo
    echo "DPO 数据: data/processed/train_expr_dpo_clean.json"
fi

if [ "$MODE" != "clean" ] && [ "$MODE" != "dpo" ] && [ "$MODE" != "all" ]; then
    echo "未知模式: $MODE"
    echo "可选: clean | dpo | all"
    exit 1
fi

echo "=============================="
echo " 完成！"
echo "=============================="
