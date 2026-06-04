#!/bin/bash
# ============================================================
# 4 模型投票推理脚本
# 用法: bash scripts/run_infer_vote.sh
#
# 流程:
#   1. CoT-GRPO 推理 → submit_cot_grpo.csv
#   2. CoT-SFT 推理 → submit_cot_sft.csv
#   3. Expr-GRPO 推理 → submit_expr_grpo.csv
#   4. Expr-SFT 推理 → submit_expr_sft.csv
#   5. 4 路投票融合 → submit_voted.csv
#
# 前置条件:
#   - outputs/checkpoints/grpo/best (CoT-GRPO)
#   - outputs/checkpoints/sft_cot/best (CoT-SFT)
#   - outputs/checkpoints/grpo_expr/best (Expr-GRPO)
#   - outputs/checkpoints/sft_expr_safe/best (Expr-SFT)
# ============================================================
set -e

PYTHON=${PYTHON:-.venv/bin/python}

echo "=============================="
echo " 4 模型投票推理"
echo "=============================="

# 清除缓存
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

mkdir -p outputs/submissions

# 检查可用模型
MODELS_FOUND=0
for ckpt in grpo sft_cot grpo_expr sft_expr_safe; do
    if [ -d "outputs/checkpoints/$ckpt/best" ]; then
        echo "  [OK] $ckpt"
        MODELS_FOUND=$((MODELS_FOUND + 1))
    else
        echo "  [--] $ckpt (未找到，跳过)"
    fi
done

echo ""
echo "可用模型: $MODELS_FOUND / 4"

if [ $MODELS_FOUND -eq 0 ]; then
    echo "错误: 没有可用的 checkpoint"
    exit 1
fi

# 逐模型推理
echo ""
echo ">>> 开始推理..."

# CoT-GRPO
if [ -d "outputs/checkpoints/grpo/best" ]; then
    echo "[1/4] CoT-GRPO 推理..."
    sed -i 's/^active_method:.*/active_method: "grpo"/' configs/infer.yaml
    $PYTHON -m src.inference.batch_infer --config configs/infer.yaml
    cp outputs/submissions/submit_grpo.csv outputs/submissions/submit_cot_grpo.csv
fi

# CoT-SFT
if [ -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "[2/4] CoT-SFT 推理..."
    sed -i 's/^active_method:.*/active_method: "sft_cot"/' configs/infer.yaml
    $PYTHON -m src.inference.batch_infer --config configs/infer.yaml
    cp outputs/submissions/submit_sft_cot.csv outputs/submissions/submit_cot_sft.csv
fi

# Expr-GRPO
if [ -d "outputs/checkpoints/grpo_expr/best" ]; then
    echo "[3/4] Expr-GRPO 推理..."
    $PYTHON -c "
import json, csv, logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from src.models.model_loader import load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.expression_candidate_infer import run_expression_candidate_inference
from pathlib import Path
from tqdm import tqdm

EXPR_INST = '请为以下数学题写出一个Python可直接计算的中缀数学表达式。用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。'

model, tokenizer = load_peft_model(
    base_model_name='$(grep "name:" configs/base.yaml | head -1 | sed "s/.*: *\"//" | sed "s/\".*//")',
    adapter_path='outputs/checkpoints/grpo_expr/best',
    torch_dtype='bfloat16',
)
predictor = MathPredictor(model=model, tokenizer=tokenizer, use_cot=True, max_new_tokens=128, temperature=0.1, do_sample=False)

with open('data/raw/test.json', 'r') as f:
    test_data = json.load(f)

run_expression_candidate_inference(
    predictor=predictor,
    test_data=test_data,
    source='expr_grpo',
    instruction=EXPR_INST,
    output_csv='outputs/submissions/submit_expr_grpo.csv',
    details_json='outputs/submissions/expr_grpo_details.json',
    temperatures=[0.1, 0.3, 0.7],
)
print('Expr-GRPO done')
"
fi

# Expr-SFT
if [ -d "outputs/checkpoints/sft_expr_safe/best" ]; then
    echo "[4/4] Expr-SFT-Safe 推理..."
    $PYTHON -c "
import json, csv, logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from src.models.model_loader import load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.expression_candidate_infer import run_expression_candidate_inference
from pathlib import Path
from tqdm import tqdm

EXPR_INST = '请为以下数学题写出一个Python可直接计算的中缀数学表达式。用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。'

model, tokenizer = load_peft_model(
    base_model_name='$(grep "name:" configs/base.yaml | head -1 | sed "s/.*: *\"//" | sed "s/\".*//")',
    adapter_path='outputs/checkpoints/sft_expr_safe/best',
    torch_dtype='bfloat16',
)
predictor = MathPredictor(model=model, tokenizer=tokenizer, use_cot=True, max_new_tokens=128, temperature=0.1, do_sample=False)

with open('data/raw/test.json', 'r') as f:
    test_data = json.load(f)

run_expression_candidate_inference(
    predictor=predictor,
    test_data=test_data,
    source='expr_sft_safe',
    instruction=EXPR_INST,
    output_csv='outputs/submissions/submit_expr_sft.csv',
    details_json='outputs/submissions/expr_sft_safe_details.json',
    temperatures=[0.1, 0.3, 0.7],
)
print('Expr-SFT done')
"
fi

# 投票融合
# 表达式优先融合
echo ""
echo ">>> 表达式优先融合..."
$PYTHON -m src.inference.expression_vote_csv \
    --test data/raw/test.json \
    --output outputs/submissions/submit_voted.csv \
    --report outputs/submissions/submit_voted_report.json \
    --expr outputs/submissions/expr_grpo_details.json:expr_grpo:1.5 \
    --expr outputs/submissions/expr_sft_safe_details.json:expr_sft_safe:1.0 \
    --cot outputs/submissions/submit_cot_grpo.csv:cot_grpo:1.5 \
    --cot outputs/submissions/submit_cot_sft.csv:cot_sft:1.0

echo ""
echo "=============================="
echo " 4 模型投票推理完成！"
echo " 最终提交: outputs/submissions/submit_voted.csv"
echo "=============================="
