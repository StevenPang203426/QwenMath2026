#!/bin/bash
# ============================================================
# 合并原始高可信、自动修复、增强数据
# 用法:
#   bash scripts/run_clean_data_merge.sh
#   bash scripts/run_clean_data_merge.sh --require_expr_for_raw
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}
EXTRA_ARGS="$@"

echo "=============================="
echo " 合并清洗增强数据"
echo "=============================="

mkdir -p data/processed

$PYTHON -m src.data.merge_clean_data $EXTRA_ARGS

echo "合并数据已生成: data/processed/train_clean_augmented.json"
echo "统一修复数据已生成: data/processed/train_repairs_unified.json"
echo "合并统计报告已生成: data/processed/train_clean_augmented_report.json"
echo "=============================="
echo " 完成！"
echo "=============================="
