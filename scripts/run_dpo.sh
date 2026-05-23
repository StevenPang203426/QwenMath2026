#!/bin/bash
# ============================================================
# 方案3: DPO 训练
# 在 SFT-CoT checkpoint 基础上进行偏好优化
# 前置条件: 先完成 run_data_build.sh + run_sft_cot.sh
# ============================================================
set -e

echo "=============================="
echo " DPO 训练"
echo "=============================="

# 检查前置条件
if [ ! -f "data/processed/train_dpo.json" ]; then
    echo "错误: DPO 数据不存在，请先运行 scripts/run_data_build.sh"
    exit 1
fi

if [ ! -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "错误: SFT-CoT checkpoint 不存在，请先运行 scripts/run_sft_cot.sh"
    exit 1
fi

# DPO 训练
echo "[1/1] 开始 DPO 训练..."
python -m src.training.dpo_trainer --config configs/dpo.yaml

echo "=============================="
echo " DPO 训练完成！"
echo " Checkpoint: outputs/checkpoints/dpo/best"
echo "=============================="
