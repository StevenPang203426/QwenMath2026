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
#   - outputs/checkpoints/sft_expr/best (Expr-SFT)
# ============================================================
set -e

echo "=============================="
echo " 4 模型投票推理"
echo "=============================="

# 清除缓存
find src -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

mkdir -p outputs/submissions

# 检查可用模型
MODELS_FOUND=0
for ckpt in grpo sft_cot grpo_expr sft_expr; do
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
    python -m src.inference.batch_infer --config configs/infer.yaml
    cp outputs/submissions/submit_grpo.csv outputs/submissions/submit_cot_grpo.csv
fi

# CoT-SFT
if [ -d "outputs/checkpoints/sft_cot/best" ]; then
    echo "[2/4] CoT-SFT 推理..."
    sed -i 's/^active_method:.*/active_method: "sft_cot"/' configs/infer.yaml
    python -m src.inference.batch_infer --config configs/infer.yaml
    cp outputs/submissions/submit_sft_cot.csv outputs/submissions/submit_cot_sft.csv
fi

# Expr-GRPO
if [ -d "outputs/checkpoints/grpo_expr/best" ]; then
    echo "[3/4] Expr-GRPO 推理..."
    python -c "
import json, csv, logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from src.models.model_loader import load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.expr_predictor import expr_predict_single
from src.inference.question_classifier import build_adaptive_prompt
from src.data.answer_extractor import extract_answer
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

with open('outputs/submissions/submit_expr_grpo.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for item in tqdm(test_data, desc='Expr-GRPO'):
        q = item['question']
        q_text = q[0].get('content','') if isinstance(q, list) else str(q)
        result = predictor.predict_single(question=q_text, instruction=EXPR_INST)
        pred = expr_predict_single(result.get('raw_output',''), q_text)
        writer.writerow([item['id'], pred['answer']])
print('Expr-GRPO done')
"
fi

# Expr-SFT
if [ -d "outputs/checkpoints/sft_expr/best" ]; then
    echo "[4/4] Expr-SFT 推理..."
    python -c "
import json, csv, logging, sys
sys.path.insert(0, '.')
logging.basicConfig(level=logging.INFO)
from src.models.model_loader import load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.expr_predictor import expr_predict_single
from pathlib import Path
from tqdm import tqdm

EXPR_INST = '请为以下数学题写出一个Python可直接计算的中缀数学表达式。用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。'

model, tokenizer = load_peft_model(
    base_model_name='$(grep "name:" configs/base.yaml | head -1 | sed "s/.*: *\"//" | sed "s/\".*//")',
    adapter_path='outputs/checkpoints/sft_expr/best',
    torch_dtype='bfloat16',
)
predictor = MathPredictor(model=model, tokenizer=tokenizer, use_cot=True, max_new_tokens=128, temperature=0.1, do_sample=False)

with open('data/raw/test.json', 'r') as f:
    test_data = json.load(f)

with open('outputs/submissions/submit_expr_sft.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for item in tqdm(test_data, desc='Expr-SFT'):
        q = item['question']
        q_text = q[0].get('content','') if isinstance(q, list) else str(q)
        result = predictor.predict_single(question=q_text, instruction=EXPR_INST)
        pred = expr_predict_single(result.get('raw_output',''), q_text)
        writer.writerow([item['id'], pred['answer']])
print('Expr-SFT done')
"
fi

# 投票融合
echo ""
echo ">>> 投票融合..."
python -c "
import csv, sys, os
sys.path.insert(0, '.')
from collections import Counter
from src.utils.metrics import normalize_number

files = {
    'cot_grpo':  ('outputs/submissions/submit_cot_grpo.csv',  1.5),
    'cot_sft':   ('outputs/submissions/submit_cot_sft.csv',   1.0),
    'expr_grpo': ('outputs/submissions/submit_expr_grpo.csv',  1.5),
    'expr_sft':  ('outputs/submissions/submit_expr_sft.csv',   1.0),
}

# 加载所有结果
all_results = {}  # id -> [(source, answer, weight)]
for name, (path, weight) in files.items():
    if not os.path.exists(path):
        print(f'  跳过 {name}: 文件不存在')
        continue
    print(f'  加载 {name} (权重 {weight})')
    with open(path, 'r') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                item_id = row[0]
                if item_id not in all_results:
                    all_results[item_id] = []
                all_results[item_id].append((name, row[1], weight))

# 投票
with open('outputs/submissions/submit_voted.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    cross_agree = 0
    total = len(all_results)
    for item_id in sorted(all_results.keys(), key=lambda x: int(x)):
        candidates = all_results[item_id]
        # 按标准化答案分组
        groups = {}
        for src, ans, w in candidates:
            key = normalize_number(ans) or ans.strip()
            if key not in groups:
                groups[key] = {'votes': 0, 'weight': 0, 'sources': set(), 'raw': ans}
            groups[key]['votes'] += 1
            groups[key]['weight'] += w
            groups[key]['sources'].add(src.split('_')[0])  # 'cot' or 'expr'

        # 跨路线一致性优先
        best = None
        for key, g in groups.items():
            if len(g['sources']) > 1:  # cot + expr 都投了
                if best is None or g['weight'] > best[1]['weight']:
                    best = (key, g)
                cross_agree += 1
                break

        if best is None:
            # 没有跨路线一致，选权重最高的
            sorted_g = sorted(groups.items(), key=lambda x: (x[1]['votes'], x[1]['weight']), reverse=True)
            best = sorted_g[0]

        writer.writerow([item_id, best[1]['raw']])

    print(f'')
    print(f'总题数: {total}')
    print(f'跨路线一致: {cross_agree} ({cross_agree/total*100:.1f}%)')
    print(f'输出: outputs/submissions/submit_voted.csv')
"

echo ""
echo "=============================="
echo " 4 模型投票推理完成！"
echo " 最终提交: outputs/submissions/submit_voted.csv"
echo "=============================="
