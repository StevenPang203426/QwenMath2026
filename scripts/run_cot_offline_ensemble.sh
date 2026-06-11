#!/bin/bash
# ============================================================
# CoT ablation 详情离线 weighted vote
# 不运行模型，只读取 val_*_details.json。
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"

"${PYTHON_BIN}" -m src.analysis.cot_ensemble_offline   --val "${VAL_PATH:-data/splits/train_expr_clean_val.json}"   --details_dir "${DETAILS_DIR:-outputs/evaluation/cot_prompt_ablation_fixed_base}"   --output_dir "${OUTPUT_DIR:-outputs/evaluation/cot_prompt_ensemble_offline}"   --candidates "${CANDIDATES:-sft_cot:direct=1.0,grpo:zero_shot_cot=0.35,grpo:few_shot_cot=0.30,dpo:few_shot_cot=0.0}"   --baseline_gate "${BASELINE_GATE:-0.7376}"   "$@"
