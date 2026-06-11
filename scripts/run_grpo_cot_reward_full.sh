#!/bin/bash
# ============================================================
# CoT GRPO reward 修复 full training
# 当前 smoke 胜出版本: balanced
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-configs/grpo_cot_reward_balanced_full.yaml}"

if [ ! -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "错误: SFT-CoT checkpoint 不存在，请先运行 scripts/run_sft_cot.sh" >&2
    exit 1
fi

echo "=============================="
echo " CoT GRPO reward full training"
echo " Config: ${CONFIG_PATH}"
echo "=============================="

"${PYTHON_BIN}" -m src.training.grpo_trainer --config "${CONFIG_PATH}"
