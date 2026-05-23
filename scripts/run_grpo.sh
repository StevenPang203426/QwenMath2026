#!/bin/bash
# ============================================================
# 方案4: GRPO 训练
# 组相对策略优化（DeepSeek-R1 风格）
# 前置条件: 先完成 run_sft_cot.sh
# ============================================================
set -e

echo "=============================="
echo " GRPO 训练"
echo "=============================="

if [ ! -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "错误: SFT-CoT checkpoint 不存在，请先运行 scripts/run_sft_cot.sh"
    exit 1
fi

# GRPO 训练
echo "[1/1] 开始 GRPO 训练..."
python -m src.training.grpo_trainer --config configs/grpo.yaml

echo "=============================="
echo " GRPO 训练完成！"
echo " Checkpoint: outputs/checkpoints/grpo/best"
echo "=============================="
