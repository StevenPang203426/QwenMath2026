#!/bin/bash
# ============================================================
# Unified inference and submission entrypoint
#
# Usage scenarios:
#   - Generate validation/test submission files from trained checkpoints.
#   - Keep final submission commands separate from validation-only analysis.
#
# Commands:
#   bash scripts/submit.sh infer [baseline|cot_prompt|sft_cot|dpo|grpo]
#   bash scripts/submit.sh expr4
#   bash scripts/submit.sh vote
#   bash scripts/submit.sh ensemble
#   bash scripts/submit.sh all-experiments
#
# Key environment variables:
#   PYTHON=.venv/bin/python
#   INFER_CONFIG=configs/inference/infer.yaml
#   OUTPUT_DIR=outputs/submissions
#
# Default outputs:
#   outputs/submissions/submit_*.csv
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
INFER_CONFIG="${INFER_CONFIG:-configs/inference/infer.yaml}"
die() { echo "错误: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; }

infer_method() {
  local method="$1" tmp_config
  tmp_config="$(mktemp)"
  "${PYTHON_BIN}" - "$INFER_CONFIG" "$tmp_config" "$method" <<'PY'
import sys, yaml
src, dst, method = sys.argv[1:4]
with open(src, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
config['active_method'] = method
with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
PY
  "${PYTHON_BIN}" -m src.inference.batch_infer --config "$tmp_config"
  rm -f "$tmp_config"
}

cmd="${1:-help}"
shift || true
case "${cmd}" in
  help|-h|--help) usage ;;
  infer) infer_method "${1:-sft_cot}" ;;
  expr4) "${PYTHON_BIN}" -m src.inference.expr4_eval_submit submit "$@" ;;
  ensemble)
    CONFIG="${ENSEMBLE_CONFIG:-configs/inference/infer_ensemble.yaml}"
    [[ -f "${CONFIG}" ]] || die "${CONFIG} 不存在。请先创建 ensemble 配置。"
    "${PYTHON_BIN}" -m src.inference.ensemble_infer --config "${CONFIG}" ;;
  vote)
    mkdir -p outputs/submissions
    [[ -d outputs/checkpoints/grpo/best ]] && infer_method grpo && cp outputs/submissions/submit_grpo.csv outputs/submissions/submit_cot_grpo.csv
    [[ -d outputs/checkpoints/sft_cot/best ]] && infer_method sft_cot && cp outputs/submissions/submit_sft_cot.csv outputs/submissions/submit_cot_sft.csv
    if [[ -d outputs/checkpoints/grpo_expr/best ]]; then
      "${PYTHON_BIN}" - <<'PY'
import json
from src.inference.expr_predictor import ExpressionPredictor
from src.inference.expression_candidate_infer import run_expression_candidate_inference
base_model_name = 'model_cache/Qwen/Qwen2.5-0.5B-Instruct'
with open('data/raw/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
predictor = ExpressionPredictor(base_model_name=base_model_name, adapter_path='outputs/checkpoints/grpo_expr/best')
run_expression_candidate_inference(
    predictor=predictor,
    test_data=test_data,
    source='expr_grpo',
    instruction='请写出可计算的数学表达式。',
    output_csv='outputs/submissions/submit_expr_grpo.csv',
    details_json='outputs/submissions/expr_grpo_details.json',
)
PY
    fi
    if [[ -d outputs/checkpoints/sft_expr_safe/best ]]; then
      "${PYTHON_BIN}" - <<'PY'
import json
from src.inference.expr_predictor import ExpressionPredictor
from src.inference.expression_candidate_infer import run_expression_candidate_inference
base_model_name = 'model_cache/Qwen/Qwen2.5-0.5B-Instruct'
with open('data/raw/test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)
predictor = ExpressionPredictor(base_model_name=base_model_name, adapter_path='outputs/checkpoints/sft_expr_safe/best')
run_expression_candidate_inference(
    predictor=predictor,
    test_data=test_data,
    source='expr_sft_safe',
    instruction='请写出可计算的数学表达式。',
    output_csv='outputs/submissions/submit_expr_sft.csv',
    details_json='outputs/submissions/expr_sft_safe_details.json',
)
PY
    fi
    "${PYTHON_BIN}" -m src.inference.expression_vote_csv \
      --test data/raw/test.json \
      --output outputs/submissions/submit_voted.csv \
      --report outputs/submissions/submit_voted_report.json \
      --expr outputs/submissions/expr_grpo_details.json:expr_grpo:1.5 \
      --expr outputs/submissions/expr_sft_safe_details.json:expr_sft_safe:1.0 \
      --cot outputs/submissions/submit_cot_grpo.csv:cot_grpo:1.5 \
      --cot outputs/submissions/submit_cot_sft.csv:cot_sft:1.0 ;;
  all-experiments)
    bash scripts/train.sh sft baseline
    bash scripts/submit.sh infer baseline
    bash scripts/data.sh cot-build "${LIMIT:-20}"
    bash scripts/train.sh sft cot
    bash scripts/submit.sh infer sft_cot
    bash scripts/train.sh dpo cot
    bash scripts/submit.sh infer dpo
    bash scripts/train.sh grpo cot legacy
    bash scripts/submit.sh infer grpo ;;
  *) usage; die "未知提交命令: ${cmd}" ;;
esac
