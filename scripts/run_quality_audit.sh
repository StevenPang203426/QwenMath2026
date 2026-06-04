#!/bin/bash
# ============================================================
# 数据质量审计与高门槛自动修复
# 用法:
#   bash scripts/run_quality_audit.sh candidates [limit]
#   bash scripts/run_quality_audit.sh audit [limit]
#   bash scripts/run_quality_audit.sh repair
#   bash scripts/run_quality_audit.sh all [limit]
# ============================================================
set -e

MODE=${1:-"candidates"}
LIMIT=${2:-0}

echo "=============================="
echo " 数据质量审计"
echo " 模式: $MODE"
echo "=============================="

mkdir -p data/processed

LIMIT_ARG=""
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
    LIMIT_ARG="--limit $LIMIT"
    echo " 限制: $LIMIT 条"
fi

run_candidates() {
    python -m src.data.quality_auditor candidates $LIMIT_ARG
    echo "候选已生成: data/processed/quality_candidates.json"
}

run_audit() {
    if [ -z "$DEEPSEEK_API_KEY" ]; then
        echo "错误: 请设置 DEEPSEEK_API_KEY 或先只运行 candidates"
        exit 1
    fi
    python -m src.data.quality_auditor audit --api_key "$DEEPSEEK_API_KEY" $LIMIT_ARG
    echo "审计已生成: data/processed/quality_audit.json"
}

run_repair() {
    python -m src.data.quality_auditor repair
    echo "修复数据已生成: data/processed/train_repaired.json"
}

if [ "$MODE" = "candidates" ]; then
    run_candidates
elif [ "$MODE" = "audit" ]; then
    run_audit
elif [ "$MODE" = "repair" ]; then
    run_repair
elif [ "$MODE" = "all" ]; then
    run_candidates
    run_audit
    run_repair
else
    echo "未知模式: $MODE"
    echo "可选: candidates | audit | repair | all"
    exit 1
fi

echo "=============================="
echo " 完成！"
echo "=============================="
