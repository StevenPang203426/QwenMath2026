#!/bin/bash
# ============================================================
# 全流程自动化脚本
# 按依赖顺序执行所有方案
# ============================================================
set -e

echo "╔══════════════════════════════════════╗"
echo "║   小学数学应用题自动解题 - 全流程    ║"
echo "╚══════════════════════════════════════╝"

START_TIME=$(date +%s)

# ---- 阶段 0: 环境检查 ----
echo ""
echo "========== 阶段 0: 环境检查 =========="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"

# ---- 阶段 1: Baseline ----
echo ""
echo "========== 阶段 1: Baseline SFT =========="
bash scripts/run_baseline.sh

# ---- 阶段 2: 方案1 CoT 提示工程 ----
echo ""
echo "========== 阶段 2: CoT 提示工程 =========="
bash scripts/run_infer.sh cot_prompt

# ---- 阶段 3: 数据构建 ----
echo ""
echo "========== 阶段 3: 数据构建 =========="
bash scripts/run_data_build.sh

# ---- 阶段 4: CoT SFT ----
echo ""
echo "========== 阶段 4: CoT SFT =========="
bash scripts/run_sft_cot.sh
bash scripts/run_infer.sh sft_cot

# ---- 阶段 5: DPO ----
echo ""
echo "========== 阶段 5: DPO =========="
bash scripts/run_dpo.sh
bash scripts/run_infer.sh dpo

# ---- 阶段 6: GRPO ----
echo ""
echo "========== 阶段 6: GRPO =========="
bash scripts/run_grpo.sh
bash scripts/run_infer.sh grpo

# ---- 完成 ----
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "╔══════════════════════════════════════╗"
echo "║           全部实验完成！             ║"
echo "╚══════════════════════════════════════╝"
echo "总耗时: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m $((ELAPSED % 60))s"
echo ""
echo "提交文件:"
ls -la outputs/submissions/
