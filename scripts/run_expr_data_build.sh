#!/bin/bash
# ============================================================
# 表达式数据构建脚本
# 用法:
#   全量正确表达式:  bash scripts/run_expr_data_build.sh correct <api_key>
#   全量错误表达式:  bash scripts/run_expr_data_build.sh wrong <api_key>
#   小批量测试:      bash scripts/run_expr_data_build.sh correct <api_key> 50
#   转换 SFT 数据:   bash scripts/run_expr_data_build.sh convert_sft
#   转换 DPO 数据:   bash scripts/run_expr_data_build.sh convert_dpo
#   导出审计候选:    bash scripts/run_expr_data_build.sh export_failed
# ============================================================
set -e

MODE=${1:-"correct"}
API_KEY=${2:-"$DEEPSEEK_API_KEY"}
LIMIT=${3:-0}

echo "=============================="
echo " 表达式数据构建"
echo " 模式: $MODE"
echo "=============================="

mkdir -p data/processed

if [ "$MODE" = "correct" ] || [ "$MODE" = "wrong" ]; then
    if [ -z "$API_KEY" ]; then
        echo "错误: 请提供 API key 或设置 DEEPSEEK_API_KEY 环境变量"
        exit 1
    fi

    LIMIT_ARG=""
    if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
        LIMIT_ARG="--limit $LIMIT"
        echo " 限制: $LIMIT 条"
    fi

    python -m src.data.expr_builder $MODE $API_KEY $LIMIT_ARG

elif [ "$MODE" = "convert_sft" ]; then
    python -m src.data.expr_builder convert_sft
    echo "SFT 数据已生成: data/processed/train_expr.json"

elif [ "$MODE" = "convert_dpo" ]; then
    python -m src.data.expr_builder convert_dpo
    echo "DPO 数据已生成: data/processed/train_expr_dpo.json"

elif [ "$MODE" = "export_failed" ]; then
    python -m src.data.expr_builder export_failed
    echo "审计候选已导出: data/processed/quality_candidates.json"
    echo "下一步: bash scripts/run_quality_audit.sh"

else
    echo "未知模式: $MODE"
    echo "可选: correct | wrong | convert_sft | convert_dpo | export_failed"
    exit 1
fi

echo "=============================="
echo " 完成！"
echo "=============================="
