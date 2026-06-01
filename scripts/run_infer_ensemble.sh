#!/bin/bash
# ============================================================
# 集成推理脚本（多路投票融合）
# 用法: bash scripts/run_infer_ensemble.sh
#
# 前置条件:
#   - 至少完成一种训练（SFT/DPO/GRPO）
#   - 配置好 configs/infer_ensemble.yaml
#
# 功能:
#   - 多温度采样（同模型 5 次，temperature 0.1~0.7）
#   - 多 checkpoint 投票（SFT/DPO/GRPO）
#   - Majority vote 选择最终答案
#   - 上下文感知的答案后处理（百分数/分数/取整/保留位数）
# ============================================================
set -e

echo "=============================="
echo " 集成推理（多路投票融合）"
echo "=============================="

# 清除 pyc 缓存
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# 检查配置文件
if [ ! -f "configs/infer_ensemble.yaml" ]; then
    echo "错误: configs/infer_ensemble.yaml 不存在"
    echo "请参考以下模板创建:"
    echo ""
    echo "  inherit: base"
    echo "  ensemble:"
    echo "    n_samples: 5"
    echo "    temperatures: [0.1, 0.3, 0.5, 0.6, 0.7]"
    echo "    checkpoints:"
    echo "      - name: grpo"
    echo "        adapter_path: outputs/checkpoints/grpo/best"
    echo "        weight: 1.5"
    echo "      - name: dpo"
    echo "        adapter_path: outputs/checkpoints/dpo/best"
    echo "        weight: 1.2"
    echo "      - name: sft"
    echo "        adapter_path: outputs/checkpoints/sft_cot/best"
    echo "        weight: 1.0"
    echo "  output:"
    echo "    dir: outputs/submissions"
    exit 1
fi

# 检查至少一个 checkpoint 存在
FOUND=0
for ckpt in outputs/checkpoints/grpo/best outputs/checkpoints/dpo/best outputs/checkpoints/sft_cot/best; do
    if [ -d "$ckpt" ]; then
        echo "找到 checkpoint: $ckpt"
        FOUND=1
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "警告: 未找到任何 checkpoint，将使用基础模型推理"
fi

python -m src.inference.ensemble_infer --config configs/infer_ensemble.yaml

echo "=============================="
echo " 集成推理完成！"
echo " 提交文件: outputs/submissions/submit_ensemble.csv"
echo "=============================="
