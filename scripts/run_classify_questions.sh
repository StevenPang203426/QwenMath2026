#!/bin/bash
# ============================================================
# 题目类型分布统计脚本
# 用法: bash scripts/run_classify_questions.sh [数据文件]
#
# 功能:
#   - 分析训练集/测试集中各类题目的分布
#   - 统计百分数/分数/取整/保留位数等类型占比
#   - 帮助了解后处理规则的覆盖率
# ============================================================
set -e

DATA_FILE=${1:-"data/raw/test.json"}

echo "=============================="
echo " 题目类型分布统计"
echo " 数据文件: $DATA_FILE"
echo "=============================="

# 清除 pyc 缓存
find src/inference -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

python -c "
import json
import sys
sys.path.insert(0, '.')

from src.inference.question_classifier import classify_batch, detect_constraints

with open('$DATA_FILE', 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = []
for item in data:
    q = item.get('question', '')
    if isinstance(q, list):
        q = q[0].get('content', '') if q else ''
    questions.append(str(q))

stats = classify_batch(questions)

print()
print(f'总题数:       {stats[\"total\"]}')
print(f'百分数题:     {stats[\"percentage\"]:>5d}  ({stats[\"percentage\"]/stats[\"total\"]*100:.1f}%)')
print(f'分数题:       {stats[\"fraction\"]:>5d}  ({stats[\"fraction\"]/stats[\"total\"]*100:.1f}%)')
print(f'向上取整题:   {stats[\"ceil\"]:>5d}  ({stats[\"ceil\"]/stats[\"total\"]*100:.1f}%)')
print(f'向下取整题:   {stats[\"floor\"]:>5d}  ({stats[\"floor\"]/stats[\"total\"]*100:.1f}%)')
print(f'保留位数题:   {stats[\"round\"]:>5d}  ({stats[\"round\"]/stats[\"total\"]*100:.1f}%)')
print(f'无特殊约束:   {stats[\"no_constraint\"]:>5d}  ({stats[\"no_constraint\"]/stats[\"total\"]*100:.1f}%)')
print()

# 展示部分样例
print('=== 各类型样例 ===')
examples = {
    '百分': [], '分数': [], '向上': [], '向下': [], '保留': []
}
for q in questions:
    constraints = detect_constraints(q)
    if not constraints:
        continue
    text = ' '.join(constraints)
    for key in examples:
        if key in text and len(examples[key]) < 2:
            examples[key].append(q[:60] + ('...' if len(q)>60 else ''))

for key, items in examples.items():
    if items:
        print(f'\n[{key}类]')
        for item in items:
            print(f'  {item}')
"

echo ""
echo "=============================="
echo " 统计完成！"
echo "=============================="
