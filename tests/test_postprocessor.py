"""
答案后处理规则引擎单元测试
"""
import sys
sys.path.insert(0, ".")

from src.inference.answer_postprocessor import (
    postprocess_answer,
    detect_answer_type,
    parse_numeric,
    format_percentage,
    format_fraction,
)
from src.inference.question_classifier import (
    detect_constraints,
    build_adaptive_prompt,
)


def test_detect_answer_type():
    """测试题目类型检测"""
    # 百分数
    assert detect_answer_type("这批产品的合格率是百分之几？")[0] == "percentage"
    assert detect_answer_type("求发芽率。")[0] == "percentage"
    assert detect_answer_type("含盐率是多少？")[0] == "percentage"

    # 分数
    assert detect_answer_type("甲是乙的几分之几？")[0] == "fraction"
    assert detect_answer_type("占总数的比例是多少？")[0] == "fraction"

    # ceil
    assert detect_answer_type("至少需要几辆车？")[0] == "ceil_integer"
    assert detect_answer_type("最少要租几条船？")[0] == "ceil_integer"

    # floor
    assert detect_answer_type("最多能做几件？")[0] == "floor_integer"
    assert detect_answer_type("最多可以买几个？")[0] == "floor_integer"

    # 保留位数
    assert detect_answer_type("保留两位小数。")[0] == "round_n"
    assert detect_answer_type("精确到0.1")[0] == "round_n"

    # 整数（量词）
    assert detect_answer_type("一共有多少人？")[0] == "integer"
    assert detect_answer_type("还剩多少个苹果？")[0] == "integer"

    # 自动
    assert detect_answer_type("结果是多少？")[0] == "auto"

    print("  detect_answer_type: ALL PASSED")


def test_postprocess_percentage():
    """测试百分数后处理"""
    # 模型输出 25，题目问百分率 → 25%
    assert postprocess_answer("25", "合格率是百分之几？") == "25%"
    # 模型输出 0.25，题目问百分率 → 25%
    assert postprocess_answer("0.25", "合格率是百分之几？") == "25%"
    # 模型已输出 25%，直接通过
    assert postprocess_answer("25%", "合格率是百分之几？") == "25%"
    # 模型输出 12.5，题目问百分率 → 12.5%
    assert postprocess_answer("12.5", "发芽率是百分之几？") == "12.5%"
    # 百分数语义 + 百分号前保留一位小数，仍应保留百分号
    assert postprocess_answer("23.4375", "甲数是乙数的百分之几?(百分号前保留一位小数)") == "23.4%"

    print("  postprocess_percentage: ALL PASSED")


def test_postprocess_fraction():
    """测试分数后处理"""
    # 模型输出 0.6 → 3/5
    assert postprocess_answer("0.6", "甲是乙的几分之几？") == "3/5"
    # 模型输出 0.5 → 1/2
    assert postprocess_answer("0.5", "占总数的几分之几？") == "1/2"
    # 模型已输出 3/5，直接通过
    assert postprocess_answer("3/5", "甲是乙的几分之几？") == "3/5"
    # 模型输出 0.333... → 1/3
    assert postprocess_answer("0.333", "占几分之几？") == "1/3"

    print("  postprocess_fraction: ALL PASSED")


def test_postprocess_ceil_floor():
    """测试取整后处理"""
    # 小数 + 至少 → ceil
    assert postprocess_answer("3.2", "至少需要几辆车？") == "4"
    # 已是整数 + 至少 → 直接通过
    assert postprocess_answer("4", "至少需要几辆车？") == "4"
    # 小数 + 最多 → floor
    assert postprocess_answer("3.8", "最多能做几件？") == "3"
    # 已是整数 + 最多 → 直接通过
    assert postprocess_answer("3", "最多能做几件？") == "3"

    print("  postprocess_ceil_floor: ALL PASSED")


def test_postprocess_round():
    """测试保留位数"""
    assert postprocess_answer("3.14159", "保留两位小数") == "3.14"
    assert postprocess_answer("3.1", "保留两位小数") == "3.10"
    assert postprocess_answer("7.85", "精确到0.1") == "7.9"

    print("  postprocess_round: ALL PASSED")


def test_postprocess_integer():
    """测试整数量词推断"""
    assert postprocess_answer("12.0", "一共有多少人？") == "12"
    assert postprocess_answer("12", "一共有多少人？") == "12"

    print("  postprocess_integer: ALL PASSED")


def test_postprocess_auto():
    """测试自动格式化"""
    assert postprocess_answer("24.0", "结果是多少？") == "24"
    assert postprocess_answer("3.5", "结果是多少？") == "3.5"
    assert postprocess_answer("", "题目") == "0"

    print("  postprocess_auto: ALL PASSED")


def test_basic_clean():
    """测试基础清理"""
    # 单位去除
    assert postprocess_answer("25千米", "结果是多少？") == "25"
    # 日期格式
    assert postprocess_answer("4月5日", "结果是多少？") == "4/5"

    print("  basic_clean: ALL PASSED")


def test_question_classifier():
    """测试自适应 prompt"""
    base = "请解答。"

    # 百分数约束
    result = build_adaptive_prompt("合格率是百分之几？", base)
    assert "百分数" in result

    # 分数约束
    result = build_adaptive_prompt("占几分之几？", base)
    assert "分数" in result

    # 无约束
    result = build_adaptive_prompt("结果是多少？", base)
    assert result == base

    print("  question_classifier: ALL PASSED")


if __name__ == "__main__":
    print("Running postprocessor tests...")
    tests = [
        test_detect_answer_type,
        test_postprocess_percentage,
        test_postprocess_fraction,
        test_postprocess_ceil_floor,
        test_postprocess_round,
        test_postprocess_integer,
        test_postprocess_auto,
        test_basic_clean,
        test_question_classifier,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {t.__name__} -> {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED!")
