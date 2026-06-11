#!/bin/bash
set -euo pipefail

MODE="${1:-all}"
shift || true

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation/cot_prompt_ablation_fixed_base}"
VAL_PATH="${VAL_PATH:-data/splits/train_expr_clean_val.json}"
TEST_PATH="${TEST_PATH:-data/raw/test.json}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VLLM_SHOW_PROGRESS="${VLLM_SHOW_PROGRESS:-1}"

COMMON_ARGS=(
  --val "${VAL_PATH}"
  --test "${TEST_PATH}"
  --output_dir "${OUTPUT_DIR}"
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

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" ]]; then
  COMMON_ARGS+=(--vllm_enforce_eager)
fi

if [[ "${VLLM_SHOW_PROGRESS}" == "1" ]]; then
  COMMON_ARGS+=(--vllm_show_progress)
fi

case "${MODE}" in
  smoke)
    "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation \
      "${COMMON_ARGS[@]}" \
      --limit "${LIMIT:-50}" \
      --skip_test \
      "$@"
    ;;
  val)
    "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation \
      "${COMMON_ARGS[@]}" \
      --skip_test \
      "$@"
    ;;
  all)
    "${PYTHON_BIN}" -m src.inference.cot_prompt_ablation \
      "${COMMON_ARGS[@]}" \
      "$@"
    ;;
  *)
    echo "Usage: bash scripts/run_cot_prompt_ablation.sh [smoke|val|all] [extra args]" >&2
    exit 2
    ;;
esac
