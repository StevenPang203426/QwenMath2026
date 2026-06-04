#!/bin/bash
# ============================================================
# 规则优先数据增强
# 用法:
#   bash scripts/run_data_augment.sh [limit]
# ============================================================
set -e

LIMIT=${1:-0}

echo "=============================="
echo " 规则数据增强"
echo "=============================="

mkdir -p data/processed data/processed/intermediate/augmentation

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    LIMIT_ARG="--limit $LIMIT"
    echo " 限制: $LIMIT 条"
fi

python -m src.data.rule_augmentor $LIMIT_ARG
python -m src.data.augmentation_auditor

echo "增强数据已生成: data/processed/train_augmented.json"
echo "增强 clean 数据已生成: data/processed/intermediate/augmentation/train_augmented_clean.json"
echo "增强审计报告已生成: data/processed/intermediate/augmentation/augmentation_report.json"
echo "=============================="
echo " 完成！"
echo "=============================="
