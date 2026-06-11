#!/bin/bash
# ============================================================
# DPO 数据与训练状态审计
# 用法:
#   bash scripts/run_dpo_audit.sh
#   SKIP_TOKENIZER=1 bash scripts/run_dpo_audit.sh
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
ARGS=(
  --config "${CONFIG_PATH:-configs/dpo.yaml}"
  --data "${DPO_DATA:-data/processed/train_dpo.json}"
  --reference_data "${REFERENCE_DATA:-data/raw/train.json}"
  --checkpoint_root "${CHECKPOINT_ROOT:-outputs/checkpoints/dpo}"
  --output_dir "${OUTPUT_DIR:-outputs/evaluation/dpo_audit}"
)

if [[ "${SKIP_TOKENIZER:-0}" == "1" ]]; then
  ARGS+=(--skip_tokenizer)
fi

"${PYTHON_BIN}" -m src.analysis.dpo_audit "${ARGS[@]}" "$@"
