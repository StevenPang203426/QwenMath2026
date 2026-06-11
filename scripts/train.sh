#!/bin/bash
# ============================================================
# Unified training entrypoint
#
# Usage scenarios:
#   - Train SFT/DPO/GRPO models for CoT, expression, and baseline routes.
#   - Run GRPO reward smoke/full experiments without adding one-off scripts.
#
# Commands:
#   bash scripts/train.sh sft baseline
#   bash scripts/train.sh sft cot
#   bash scripts/train.sh sft expr [clean|safe|legacy]
#   bash scripts/train.sh dpo cot
#   bash scripts/train.sh dpo expr [clean|legacy]
#   bash scripts/train.sh grpo cot [legacy|strict-smoke|balanced-smoke|smoke|full|balanced-full]
#   bash scripts/train.sh grpo expr [clean|clean-fast|clean-vllm|from-dpo-clean|from-dpo-clean-fast|from-dpo-clean-vllm|legacy|from-dpo-legacy]
#
# Key environment variables:
#   PYTHON=.venv/bin/python       Python executable.
#   STRICT_VLLM_VERSION=1         Fail if local vLLM is outside TRL's declared range.
#
# Default outputs:
#   outputs/checkpoints/<experiment>/best
# ============================================================
set -euo pipefail

PYTHON_BIN="${PYTHON:-.venv/bin/python}"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0" >&2; }
die() { echo "错误: $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "$1 不存在。请先运行: $2"; }
need_dir() { [[ -d "$1" ]] || die "$1 不存在。请先运行: $2"; }

run_config() {
  local module="$1"
  local config="$2"
  echo "=============================="
  echo " Training: ${module}"
  echo " Config:   ${config}"
  echo "=============================="
  "${PYTHON_BIN}" -m "${module}" --config "${config}"
}

check_vllm_version() {
  "${PYTHON_BIN}" - <<'PY'
from packaging.version import Version
import os
import trl
import vllm
v = Version(vllm.__version__)
if not (Version("0.12.0") <= v <= Version("0.18.0")):
    msg = (
        f"当前 vLLM={vllm.__version__}, TRL={trl.__version__}。"
        "TRL 1.4.0 仅声明支持 vLLM 0.12.0 到 0.18.0；"
        "本环境已通过 smoke，但如果运行异常请改用 fast 变体。"
    )
    if os.environ.get("STRICT_VLLM_VERSION") == "1":
        raise SystemExit(msg)
    print("警告:", msg)
PY
}

method="${1:-help}"
route="${2:-}"
variant="${3:-default}"
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true
shift $(( $# >= 1 ? 1 : 0 )) || true

case "${method}:${route}:${variant}" in
  help:*:*|-h:*:*|--help:*:*) usage ;;
  sft:baseline:default) run_config src.training.sft_trainer configs/cot/sft_baseline.yaml ;;
  sft:cot:default)
    need_file data/processed/train_cot.json "bash scripts/data.sh cot-build all"
    run_config src.training.sft_trainer configs/cot/sft_cot.yaml ;;
  sft:expr:legacy)
    need_file data/processed/train_expr.json "bash scripts/data.sh expr-build convert_sft"
    run_config src.training.sft_trainer configs/expr/sft_expr.yaml ;;
  sft:expr:clean|sft:expr:default)
    need_file data/splits/train_expr_clean_train.json "bash scripts/data.sh expr-training clean"
    run_config src.training.sft_trainer configs/expr/sft_expr_clean.yaml ;;
  sft:expr:safe)
    need_file data/processed/train_expr_safe.json "bash scripts/data.sh expr-repair"
    run_config src.training.sft_trainer configs/expr/sft_expr_safe.yaml ;;
  dpo:cot:default)
    need_file data/processed/train_dpo.json "bash scripts/data.sh cot-build all"
    need_dir outputs/checkpoints/sft_cot/best "bash scripts/train.sh sft cot"
    run_config src.training.dpo_trainer configs/cot/dpo.yaml ;;
  dpo:expr:legacy)
    need_file data/processed/train_expr_dpo.json "bash scripts/data.sh expr-build convert_dpo"
    need_dir outputs/checkpoints/sft_expr_safe/best "bash scripts/train.sh sft expr safe"
    run_config src.training.dpo_trainer configs/expr/dpo_expr.yaml ;;
  dpo:expr:clean|dpo:expr:default)
    need_file data/processed/train_expr_dpo_clean.json "bash scripts/data.sh expr-training dpo"
    need_dir outputs/checkpoints/sft_expr_clean/best "bash scripts/train.sh sft expr clean"
    run_config src.training.dpo_trainer configs/expr/dpo_expr_clean.yaml ;;
  grpo:cot:legacy|grpo:cot:default)
    need_dir outputs/checkpoints/sft_cot/best "bash scripts/train.sh sft cot"
    run_config src.training.grpo_trainer configs/cot/grpo.yaml ;;
  grpo:cot:strict-smoke)
    need_dir outputs/checkpoints/sft_cot/best "bash scripts/train.sh sft cot"
    run_config src.training.grpo_trainer configs/experiments/grpo_cot_reward_strict_smoke.yaml ;;
  grpo:cot:balanced-smoke)
    need_dir outputs/checkpoints/sft_cot/best "bash scripts/train.sh sft cot"
    run_config src.training.grpo_trainer configs/experiments/grpo_cot_reward_balanced_smoke.yaml ;;
  grpo:cot:smoke)
    case "${1:-all}" in
      strict) exec bash scripts/train.sh grpo cot strict-smoke ;;
      balanced) exec bash scripts/train.sh grpo cot balanced-smoke ;;
      all) bash scripts/train.sh grpo cot strict-smoke; bash scripts/train.sh grpo cot balanced-smoke ;;
      *) die "smoke 只支持 strict|balanced|all" ;;
    esac ;;
  grpo:cot:full|grpo:cot:balanced-full)
    need_dir outputs/checkpoints/sft_cot/best "bash scripts/train.sh sft cot"
    run_config src.training.grpo_trainer configs/experiments/grpo_cot_reward_balanced_full.yaml ;;
  grpo:expr:legacy)
    need_dir outputs/checkpoints/sft_expr_safe/best "bash scripts/train.sh sft expr safe"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr.yaml ;;
  grpo:expr:from-dpo-legacy)
    need_dir outputs/checkpoints/dpo_expr/best "bash scripts/train.sh dpo expr legacy"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_from_dpo.yaml ;;
  grpo:expr:clean|grpo:expr:default)
    need_dir outputs/checkpoints/sft_expr_clean/best "bash scripts/train.sh sft expr clean"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_clean.yaml ;;
  grpo:expr:clean-fast)
    need_dir outputs/checkpoints/sft_expr_clean/best "bash scripts/train.sh sft expr clean"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_clean_fast.yaml ;;
  grpo:expr:clean-vllm)
    need_dir outputs/checkpoints/sft_expr_clean/best "bash scripts/train.sh sft expr clean"
    check_vllm_version
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_clean_vllm.yaml ;;
  grpo:expr:from-dpo-clean)
    need_dir outputs/checkpoints/dpo_expr_clean/best "bash scripts/train.sh dpo expr clean"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_from_dpo_clean.yaml ;;
  grpo:expr:from-dpo-clean-fast)
    need_dir outputs/checkpoints/dpo_expr_clean/best "bash scripts/train.sh dpo expr clean"
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_from_dpo_clean_fast.yaml ;;
  grpo:expr:from-dpo-clean-vllm)
    need_dir outputs/checkpoints/dpo_expr_clean/best "bash scripts/train.sh dpo expr clean"
    check_vllm_version
    run_config src.training.grpo_expr_trainer configs/expr/grpo_expr_from_dpo_clean_vllm.yaml ;;
  *) usage; die "未知训练命令: ${method} ${route} ${variant}" ;;
esac
