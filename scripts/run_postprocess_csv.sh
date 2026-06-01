#!/bin/bash
# ============================================================
# 对已有的 submit.csv 进行后处理
# 用法: bash scripts/run_postprocess_csv.sh <输入csv> [输出csv]
#
# 功能:
#   - 读取已有的推理结果 CSV（id, answer）
#   - 结合原始题目文本，对每个答案进行上下文感知后处理
#   - 输出新的 CSV 文件
#
# 适用场景:
#   - 已经跑完推理但未做后处理的旧结果
#   - 想对比后处理前后的差异
# ============================================================
set -e

INPUT_CSV=${1:-""}
OUTPUT_CSV=${2:-""}
TEST_JSON=${3:-"data/raw/test.json"}

if [ -z "$INPUT_CSV" ]; then
    echo "用法: bash scripts/run_postprocess_csv.sh <输入csv> [输出csv] [测试集json]"
    echo ""
    echo "示例:"
    echo "  bash scripts/run_postprocess_csv.sh outputs/submissions/submit_grpo.csv"
    echo "  bash scripts/run_postprocess_csv.sh submit.csv submit_fixed.csv"
    exit 1
fi

if [ -z "$OUTPUT_CSV" ]; then
    # 自动生成输出文件名
    BASENAME=$(basename "$INPUT_CSV" .csv)
    OUTPUT_CSV="$(dirname "$INPUT_CSV")/${BASENAME}_postprocessed.csv"
fi

echo "=============================="
echo " 答案后处理"
echo " 输入: $INPUT_CSV"
echo " 输出: $OUTPUT_CSV"
echo " 测试集: $TEST_JSON"
echo "=============================="

# 清除 pyc 缓存
find src/inference -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

python -c "
import csv
import json
import sys
sys.path.insert(0, '.')

from src.inference.answer_postprocessor import postprocess_answer

# 加载测试集（获取 question 文本）
with open('$TEST_JSON', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 构建 id → question 映射
id_to_question = {}
for item in test_data:
    q = item.get('question', '')
    if isinstance(q, list):
        q = q[0].get('content', '') if q else ''
    id_to_question[str(item['id'])] = str(q)

# 读取输入 CSV
rows = []
with open('$INPUT_CSV', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            rows.append(row)

# 后处理并写入输出
changed = 0
with open('$OUTPUT_CSV', 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for row in rows:
        item_id = row[0]
        old_answer = row[1]
        question = id_to_question.get(item_id, '')
        new_answer = postprocess_answer(old_answer, question)
        if new_answer != old_answer:
            changed += 1
        writer.writerow([item_id, new_answer])

print(f'')
print(f'总条数:   {len(rows)}')
print(f'修改数:   {changed}  ({changed/len(rows)*100:.1f}%)')
print(f'未修改:   {len(rows)-changed}')
"

echo ""
echo "=============================="
echo " 后处理完成！"
echo " 输出文件: $OUTPUT_CSV"
echo "=============================="
