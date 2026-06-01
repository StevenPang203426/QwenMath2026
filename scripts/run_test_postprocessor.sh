#!/bin/bash
# ============================================================
# 答案后处理规则引擎 单元测试
# 用法: bash scripts/run_test_postprocessor.sh
# ============================================================
set -e

echo "=============================="
echo " 答案后处理规则引擎 单元测试"
echo "=============================="

# 清除 pyc 缓存，确保使用最新源码
find src/inference -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

python tests/test_postprocessor.py

echo "=============================="
echo " 测试完成！"
echo "=============================="
