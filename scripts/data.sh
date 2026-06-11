#!/bin/bash
# ============================================================
# Unified data pipeline entrypoint
#
# Usage scenarios:
#   - Build CoT data, expression data, quality repair data, and clean expression splits.
#   - Keep API-dependent and no-API data steps behind one documented interface.
#
# Commands:
#   bash scripts/data.sh cot-build [20|50|all]
#   bash scripts/data.sh expr-build [correct|wrong|convert_sft|convert_dpo|export_failed] [api_key] [limit]
#   bash scripts/data.sh expr-repair [api|no_api] [limit]
#   bash scripts/data.sh expr-training [clean|dpo|all]
#   bash scripts/data.sh quality-audit [candidates|audit|repair|all] [limit]
#   bash scripts/data.sh format-repair [limit]
#   bash scripts/data.sh augment [limit]
#   bash scripts/data.sh clean-merge [--require_expr_for_raw]
#
# Key environment variables:
#   PYTHON=.venv/bin/python
#   DEEPSEEK_API_KEY=...       Required for API-backed generation/audit.
#   CHECKPOINT_EVERY=100       Expression repair checkpoint interval.
#
# Default outputs:
#   data/processed/, data/splits/, data/processed/intermediate/
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"
die() { echo "错误: $*" >&2; exit 1; }
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; }
cmd="${1:-help}"
shift || true

case "${cmd}" in
  help|-h|--help) usage ;;
  cot-build)
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] || die "请先设置 DEEPSEEK_API_KEY"
    LIMIT="${1:-20}"; LIMIT_FLAG=()
    [[ "${LIMIT}" != "all" ]] && LIMIT_FLAG=(--limit "${LIMIT}")
    "${PYTHON_BIN}" -m src.data.data_builder --input data/raw/train.json --output data/processed/train_cot_raw.json --api_key "${DEEPSEEK_API_KEY}" --model deepseek-v4-flash --workers "${WORKERS:-4}" --generate_wrong "${LIMIT_FLAG[@]}"
    "${PYTHON_BIN}" -m src.data.preprocessor --action sft --input data/processed/train_cot_raw.json --output data/processed/train_cot.json
    "${PYTHON_BIN}" -m src.data.preprocessor --action dpo --input data/processed/train_cot_raw.json --output data/processed/train_dpo.json ;;
  expr-build)
    MODE="${1:-correct}"; API_KEY="${2:-${DEEPSEEK_API_KEY:-}}"; LIMIT="${3:-0}"
    mkdir -p data/processed data/processed/intermediate/quality
    case "${MODE}" in
      correct|wrong)
        [[ -n "${API_KEY}" ]] || die "请提供 API key 或设置 DEEPSEEK_API_KEY"
        LIMIT_ARG=(); [[ "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]] && LIMIT_ARG=(--limit "${LIMIT}")
        "${PYTHON_BIN}" -m src.data.expr_builder "${MODE}" "${API_KEY}" "${LIMIT_ARG[@]}" ;;
      convert_sft|convert_dpo|export_failed) "${PYTHON_BIN}" -m src.data.expr_builder "${MODE}" ;;
      *) die "expr-build 只支持 correct|wrong|convert_sft|convert_dpo|export_failed" ;;
    esac ;;
  expr-repair)
    MODE="${1:-api}"; LIMIT="${2:-0}"; ARGS=(--checkpoint_every "${CHECKPOINT_EVERY:-100}")
    if [[ "${MODE}" == "no_api" ]]; then ARGS+=(--no_api); elif [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then die "请先设置 DEEPSEEK_API_KEY，或运行: bash scripts/data.sh expr-repair no_api"; fi
    [[ "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]] && ARGS+=(--limit "${LIMIT}")
    "${PYTHON_BIN}" -m src.data.expression_repair "${ARGS[@]}" ;;
  expr-training) "${PYTHON_BIN}" -m src.data.expression_training_data "${1:-all}" ;;
  quality-audit)
    MODE="${1:-candidates}"; LIMIT="${2:-0}"; LIMIT_ARG=()
    [[ "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]] && LIMIT_ARG=(--limit "${LIMIT}")
    run_candidates() { "${PYTHON_BIN}" -m src.data.quality_auditor candidates "${LIMIT_ARG[@]}"; }
    run_audit() { [[ -n "${DEEPSEEK_API_KEY:-}" ]] || die "请设置 DEEPSEEK_API_KEY 或先只运行 candidates"; "${PYTHON_BIN}" -m src.data.quality_auditor audit --api_key "${DEEPSEEK_API_KEY}" "${LIMIT_ARG[@]}"; }
    run_repair() { "${PYTHON_BIN}" -m src.data.quality_auditor repair; }
    case "${MODE}" in candidates) run_candidates ;; audit) run_audit ;; repair) run_repair ;; all) run_candidates; run_audit; run_repair ;; *) die "quality-audit 只支持 candidates|audit|repair|all" ;; esac ;;
  format-repair)
    LIMIT="${1:-0}"; ARGS=(); [[ "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]] && ARGS+=(--limit "${LIMIT}")
    "${PYTHON_BIN}" -m src.data.format_repair "${ARGS[@]}" ;;
  augment)
    LIMIT="${1:-0}"; ARGS=(); [[ "${LIMIT}" =~ ^[0-9]+$ && "${LIMIT}" -gt 0 ]] && ARGS+=(--limit "${LIMIT}")
    "${PYTHON_BIN}" -m src.data.rule_augmentor "${ARGS[@]}"
    "${PYTHON_BIN}" -m src.data.augmentation_auditor ;;
  clean-merge) "${PYTHON_BIN}" -m src.data.merge_clean_data "$@" ;;
  *) usage; die "未知数据命令: ${cmd}" ;;
esac
