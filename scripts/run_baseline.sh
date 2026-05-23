#!/bin/bash
# ============================================================
# 方案0: Baseline SFT 训练
# 直接预测数字答案（无推理过程）
# ============================================================
set -e

echo "=============================="
echo " Baseline SFT 训练"
echo "=============================="

# 1. 数据划分（训练集/验证集）
echo "[1/3] 划分训练集和验证集..."
python -m src.data.preprocessor \
    --action split \
    --input data/raw/train.json \
    --output data/splits/train_split.json \
    --val_output data/splits/val_split.json

# 2. SFT 训练
echo "[2/3] 开始 Baseline SFT 训练..."
python -m src.training.sft_trainer --config configs/sft_baseline.yaml

# 3. 推理
echo "[3/3] 推理生成 submit.csv..."
python -m src.inference.batch_infer --config configs/infer.yaml

echo "=============================="
echo " Baseline 完成！"
echo " 提交文件: outputs/submissions/submit_baseline.csv"
echo "=============================="
