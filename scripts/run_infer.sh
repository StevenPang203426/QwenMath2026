#!/bin/bash
# ============================================================
# 推理脚本
# 用法: bash scripts/run_infer.sh [方案名]
# 方案名: baseline | cot_prompt | sft_cot | dpo | grpo
# ============================================================
set -e

METHOD=${1:-"sft_cot"}

echo "=============================="
echo " 推理: $METHOD"
echo "=============================="

# 修改 infer.yaml 中的 active_method（用 sed 替换）
sed -i "s/^active_method:.*/active_method: \"$METHOD\"/" configs/infer.yaml

python -m src.inference.batch_infer --config configs/infer.yaml

echo "=============================="
echo " 推理完成！"
echo " 提交文件: outputs/submissions/submit_${METHOD}.csv"
echo "=============================="
