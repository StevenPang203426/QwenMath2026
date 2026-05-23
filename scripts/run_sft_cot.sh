#!/bin/bash
# ============================================================
# 方案2: CoT SFT 训练
# 使用带推理步骤的数据微调模型
# 前置条件: 先运行 run_data_build.sh 生成 CoT 数据
# ============================================================
set -e

echo "=============================="
echo " CoT SFT 训练"
echo "=============================="

# 检查数据是否存在
if [ ! -f "data/processed/train_cot.json" ]; then
    echo "错误: CoT 数据不存在，请先运行 scripts/run_data_build.sh"
    exit 1
fi

# SFT 训练
echo "[1/1] 开始 CoT SFT 训练..."
python -m src.training.sft_trainer --config configs/sft_cot.yaml

echo "=============================="
echo " CoT SFT 训练完成！"
echo " Checkpoint: outputs/checkpoints/sft_cot/best"
echo "=============================="
