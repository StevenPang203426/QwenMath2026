#!/bin/bash
# ============================================================
# Format repair for expression records.
# Usage:
#   bash scripts/run_format_repair.sh [limit]
# ============================================================
set -e

LIMIT=${1:-0}

echo "=============================="
echo " 表达式答案格式修复"
echo "=============================="

mkdir -p data/processed/intermediate/format_repair

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    LIMIT_ARG="--limit $LIMIT"
    echo " 限制: $LIMIT 条"
fi

python -m src.data.format_repair $LIMIT_ARG

echo "格式修复数据已生成: data/processed/intermediate/format_repair/expr_format_repaired.json"
echo "格式修复拒绝数据已生成: data/processed/intermediate/format_repair/expr_format_rejected.json"
echo "格式修复报告已生成: data/processed/intermediate/format_repair/format_repair_report.json"
echo "=============================="
echo " 完成！"
echo "=============================="
