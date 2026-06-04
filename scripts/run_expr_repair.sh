#!/bin/bash
# ============================================================
# 幂等修复不规范表达式数据
# 用法:
#   export DEEPSEEK_API_KEY=your_key
#   bash scripts/run_expr_repair.sh [limit]
#   bash scripts/run_expr_repair.sh no_api [limit]
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-100}

MODE=${1:-api}
LIMIT=0

NO_API_ARG=""
if [ "$MODE" = "no_api" ]; then
    NO_API_ARG="--no_api"
    LIMIT=${2:-0}
elif [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "错误: 请先设置 DEEPSEEK_API_KEY，或运行: bash scripts/run_expr_repair.sh no_api"
    exit 1
else
    LIMIT=${1:-0}
fi

echo "=============================="
echo " 表达式规范修复"
echo "=============================="
echo " 检查点: 每 $CHECKPOINT_EVERY 条写盘"

mkdir -p data/processed/intermediate/expression_repair

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    LIMIT_ARG="--limit $LIMIT"
    echo " 限制: $LIMIT 条"
fi

$PYTHON -m src.data.expression_repair --checkpoint_every "$CHECKPOINT_EVERY" $NO_API_ARG $LIMIT_ARG

echo "安全表达式: data/processed/intermediate/expression_repair/expr_correct_safe.json"
echo "修复记录: data/processed/intermediate/expression_repair/expr_repaired.json"
echo "拒绝记录: data/processed/intermediate/expression_repair/expr_rejected.json"
echo "安全 SFT: data/processed/train_expr_safe.json"
echo "=============================="
echo " 完成！"
echo "=============================="
