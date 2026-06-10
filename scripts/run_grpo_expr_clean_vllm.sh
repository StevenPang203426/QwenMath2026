#!/bin/bash
# ============================================================
# 表达式路线 GRPO vLLM 加速
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

if [ ! -d "outputs/checkpoints/sft_expr_clean/best" ]; then
    echo "错误: outputs/checkpoints/sft_expr_clean/best 不存在"
    echo "请先运行: bash scripts/run_sft_expr_clean.sh"
    exit 1
fi

$PYTHON - <<'PY'
from packaging.version import Version

import trl
import vllm
import os

v = Version(vllm.__version__)
if not (Version("0.12.0") <= v <= Version("0.18.0")):
    msg = (
        f"当前 vLLM={vllm.__version__}, TRL={trl.__version__}。"
        "TRL 1.4.0 仅声明支持 vLLM 0.12.0 到 0.18.0；"
        "本环境已通过 1-step smoke，但如果运行异常请改用 scripts/run_grpo_expr_clean_fast.sh。"
    )
    if os.environ.get("STRICT_VLLM_VERSION") == "1":
        raise SystemExit(msg)
    print("警告:", msg)
PY

$PYTHON -m src.training.grpo_expr_trainer --config configs/grpo_expr_clean_vllm.yaml

echo "Checkpoint: outputs/checkpoints/grpo_expr_clean_vllm/best"
