#!/bin/bash
# ============================================================
# CoT GRPO reward 修复 smoke
# 用法:
#   bash scripts/run_grpo_cot_reward_smoke.sh strict
#   bash scripts/run_grpo_cot_reward_smoke.sh balanced
#   bash scripts/run_grpo_cot_reward_smoke.sh all
# ============================================================
set -euo pipefail

MODE="${1:-all}"
PYTHON_BIN="${PYTHON:-.venv/bin/python}"

if [ ! -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "错误: SFT-CoT checkpoint 不存在，请先运行 scripts/run_sft_cot.sh" >&2
    exit 1
fi

run_one() {
    local variant="$1"
    local config="configs/grpo_cot_reward_${variant}_smoke.yaml"
    echo "=============================="
    echo " CoT GRPO reward smoke: ${variant}"
    echo " Config: ${config}"
    echo "=============================="
    "${PYTHON_BIN}" -m src.training.grpo_trainer --config "${config}"
}

case "${MODE}" in
    strict)
        run_one strict
        ;;
    balanced)
        run_one balanced
        ;;
    all)
        run_one strict
        run_one balanced
        ;;
    *)
        echo "Usage: bash scripts/run_grpo_cot_reward_smoke.sh [strict|balanced|all]" >&2
        exit 2
        ;;
esac
