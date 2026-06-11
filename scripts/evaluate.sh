#!/bin/bash
# ============================================================
# Unified evaluation and analysis entrypoint
#
# Usage scenarios:
#   - Run validation ablations, offline ensembles, DPO audits, and utility checks.
#   - Keep experiment parameters visible through env vars instead of many one-off scripts.
#
# Commands:
#   bash scripts/evaluate.sh cot-ablation [smoke|val|all]
#   bash scripts/evaluate.sh cot-ensemble
#   bash scripts/evaluate.sh dpo-audit
#   bash scripts/evaluate.sh expr4-validate
#   bash scripts/evaluate.sh expr-grpo-ablation
#   bash scripts/evaluate.sh classify [data/raw/test.json]
#   bash scripts/evaluate.sh postprocess <input.csv> [output.csv] [data/raw/test.json]
#   bash scripts/evaluate.sh postprocessor-tests
#
# Key environment variables:
#   PYTHON=.venv/bin/python
#   OUTPUT_DIR=outputs/evaluation/<experiment>
#   MODELS=sft_cot,dpo,grpo
#   PROMPTS=direct,zero_shot_cot,few_shot_cot
#   SKIP_TOKENIZER=1 for dpo-audit without tokenizer length checks.
#
# Default outputs:
#   outputs/evaluation/<experiment>/, outputs/submissions/*_postprocessed.csv
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
die() { echo "错误: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; }
cmd="${1:-help}"
shift || true

case "${cmd}" in
  help|-h|--help) usage ;;
  cot-ablation)
    MODE="${1:-all}"; shift || true
    BATCH_SIZE="${BATCH_SIZE:-64}"
    VLLM_SHOW_PROGRESS="${VLLM_SHOW_PROGRESS:-1}"
    COMMON_ARGS=(
      --val "${VAL_PATH:-data/splits/train_expr_clean_val.json}"
      --test "${TEST_PATH:-data/raw/test.json}"
      --output_dir "${OUTPUT_DIR:-outputs/evaluation/cot_prompt_ablation_fixed_base}"
      --models "${MODELS:-sft_cot,dpo,grpo}"
      --prompts "${PROMPTS:-direct,zero_shot_cot,few_shot_cot}"
      --engine "${ENGINE:-vllm}"
      --batch_size "${BATCH_SIZE}"
      --max_new_tokens "${MAX_NEW_TOKENS:-512}"
      --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
      --vllm_max_lora_rank "${VLLM_MAX_LORA_RANK:-16}"
      --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-2048}"
      --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-${BATCH_SIZE}}"
      --sft_merged_dir "${SFT_MERGED_DIR:-outputs/checkpoints/sft_cot_merged}"
    )
    [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]] && COMMON_ARGS+=(--vllm_enforce_eager)
    [[ "${VLLM_SHOW_PROGRESS}" == "1" ]] && COMMON_ARGS+=(--vllm_show_progress)
    case "${MODE}" in
      smoke) "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation "${COMMON_ARGS[@]}" --limit "${LIMIT:-50}" --skip_test "$@" ;;
      val) "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation "${COMMON_ARGS[@]}" --skip_test "$@" ;;
      all) "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation "${COMMON_ARGS[@]}" "$@" ;;
      *) die "cot-ablation 模式只支持 smoke|val|all" ;;
    esac ;;
  cot-ensemble)
    "${PYTHON_BIN}" -m src.analysis.cot_ensemble_offline \
      --val "${VAL_PATH:-data/splits/train_expr_clean_val.json}" \
      --details_dir "${DETAILS_DIR:-outputs/evaluation/cot_prompt_ablation_fixed_base}" \
      --output_dir "${OUTPUT_DIR:-outputs/evaluation/cot_prompt_ensemble_offline}" \
      --candidates "${CANDIDATES:-sft_cot:direct=1.0,grpo:zero_shot_cot=0.35,grpo:few_shot_cot=0.30,dpo:few_shot_cot=0.0}" \
      --baseline_gate "${BASELINE_GATE:-0.7376}" "$@" ;;
  dpo-audit)
    ARGS=(--config "${CONFIG_PATH:-configs/cot/dpo.yaml}" --data "${DPO_DATA:-data/processed/train_dpo.json}" --reference_data "${REFERENCE_DATA:-data/raw/train.json}" --checkpoint_root "${CHECKPOINT_ROOT:-outputs/checkpoints/dpo}" --output_dir "${OUTPUT_DIR:-outputs/evaluation/dpo_audit}")
    [[ "${SKIP_TOKENIZER:-0}" == "1" ]] && ARGS+=(--skip_tokenizer)
    "${PYTHON_BIN}" -m src.analysis.dpo_audit "${ARGS[@]}" "$@" ;;
  expr4-validate) "${PYTHON_BIN}" -m src.inference.expr4_eval_submit validate "$@" ;;
  expr-grpo-ablation) "${PYTHON_BIN}" -m src.inference.expr_grpo_ablation "$@" ;;
  classify)
    DATA_FILE="${1:-data/raw/test.json}"
    "${PYTHON_BIN}" - "$DATA_FILE" <<'PY'
import json, sys
from src.inference.question_classifier import classify_batch, detect_constraints
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    data = json.load(f)
questions = []
for item in data:
    q = item.get('question', '')
    if isinstance(q, list):
        q = q[0].get('content', '') if q else ''
    questions.append(str(q))
stats = classify_batch(questions)
print(f"总题数:       {stats['total']}")
for key, label in [('percentage','百分数题'),('fraction','分数题'),('ceil','向上取整题'),('floor','向下取整题'),('round','保留位数题'),('no_constraint','无特殊约束')]:
    print(f"{label}: {stats[key]:>5d}  ({stats[key]/stats['total']*100:.1f}%)")
print('\n=== 各类型样例 ===')
examples = {'百分': [], '分数': [], '向上': [], '向下': [], '保留': []}
for q in questions:
    text = ' '.join(detect_constraints(q))
    for key in examples:
        if key in text and len(examples[key]) < 2:
            examples[key].append(q[:60] + ('...' if len(q) > 60 else ''))
for key, items in examples.items():
    if items:
        print(f"\n[{key}类]")
        for item in items:
            print(f"  {item}")
PY
    ;;
  postprocess)
    INPUT_CSV="${1:-}"; [[ -n "${INPUT_CSV}" ]] || die "postprocess 需要输入 csv"
    OUTPUT_CSV="${2:-${INPUT_CSV%.csv}_postprocessed.csv}"
    TEST_JSON="${3:-data/raw/test.json}"
    "${PYTHON_BIN}" - "$INPUT_CSV" "$OUTPUT_CSV" "$TEST_JSON" <<'PY'
import csv, json, sys
from src.inference.answer_postprocessor import postprocess_answer
input_csv, output_csv, test_json = sys.argv[1:4]
with open(test_json, 'r', encoding='utf-8') as f:
    questions = {str(item['id']): item.get('question', '') for item in json.load(f)}
with open(input_csv, newline='', encoding='utf-8') as f:
    rows = list(csv.reader(f))
fixed, changed = [], 0
for row in rows:
    if len(row) < 2:
        continue
    new_answer = postprocess_answer(row[1], questions.get(row[0], ''))
    changed += int(new_answer != row[1])
    fixed.append([row[0], new_answer])
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    csv.writer(f).writerows(fixed)
print(f"后处理完成: {output_csv}; changed={changed}/{len(fixed)}")
PY
    ;;
  postprocessor-tests) "${PYTHON_BIN}" tests/unit/test_postprocessor.py ;;
  *) usage; die "未知评测命令: ${cmd}" ;;
esac
