"""
题目类型分类器 + 自适应 Prompt 构建

在推理前分析题目类型，将格式约束动态注入 system prompt，
与 answer_postprocessor.py 形成双保险：
  - Prompt 引导模型输��正确格式（提高模型输出质量）
  - 后处理规则兜底修正（确保最终格式正确）
"""
import re
from typing import List


# ============================================================
# 格式约束检测规则
# ============================================================

_CONSTRAINTS_RULES = [
    # (正则模式, 约束描述)
    # 百分数
    (re.compile(r'百分之几|百��率|百分比|百分数'), "答案写成百分数形式，如25%"),
    (re.compile(
        r'(?:合格|成活|发芽|出勤|命中|正确|错误|出错|达标|及格|'
        r'利润|增长|下降|节约|浪费|损失|淘汰|录取|中奖|'
        r'浓度|纯度|盐度|含盐|含糖|酒精)率(?!的|高|低)'
    ), "答案写成百分数形式，如12.5%"),
    # 分数
    (re.compile(r'几分之几|分率'), "答案写成分数形式，如3/5，不要化成小数"),
    (re.compile(r'比例|比率|比重|占比'), "答案写成分数形式，如2/3"),
    # 取整
    (re.compile(r'(?:至少|最少|起码).*?[辆人个只条棵���张本台次箱船车桶圈]'),
     "如果结果不是整数，向上取整"),
    (re.compile(r'(?:至多|最多|顶多).*?(?:能|可以|可|装|分|做|买)'),
     "如果结果不是整数，向下取整"),
    # 保留位数
    (re.compile(r'保留[到]?[一1][位]小数|精确到0\.1|精确到十分位'),
     "答案保留一位小数"),
    (re.compile(r'保留[到]?[两二2][位]小数|精确到0\.01|精确到百分位'),
     "答案保留两位小数"),
    (re.compile(r'保留[到]?整数|精确到[个]?位|精确到整'),
     "答案取整数"),
]


def detect_constraints(question: str) -> List[str]:
    """
    检测题目中隐含的格式约束

    Args:
        question: 原始题目文本

    Returns:
        约束描述列表
    """
    if not question:
        return []

    constraints = []
    for pattern, description in _CONSTRAINTS_RULES:
        if pattern.search(question):
            constraints.append(description)

    # 去重（百分数类可能命中多条）
    seen = set()
    unique = []
    for c in constraints:
        key = c[:4]  # 取前4字符作为去重key
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def build_adaptive_prompt(question: str, base_instruction: str) -> str:
    """
    根据题目类型构建自适应 prompt

    如果检测到格式约束，追加到 base_instruction 末尾。
    否则原样返回。

    Args:
        question: 原始题目文本
        base_instruction: 基础 system instruction

    Returns:
        增强后的 instruction
    """
    constraints = detect_constraints(question)

    if not constraints:
        return base_instruction

    constraint_text = "；".join(constraints) + "。"
    return f"{base_instruction}\n【格式约束】{constraint_text}"


# ============================================================
# 批量分类统计（用于调试/分析）
# ============================================================

def classify_batch(questions: List[str]) -> dict:
    """
    批量分类题目并统计各类型分布

    Args:
        questions: 题目列表

    Returns:
        统计字典
    """
    stats = {
        "total": len(questions),
        "percentage": 0,
        "fraction": 0,
        "ceil": 0,
        "floor": 0,
        "round": 0,
        "no_constraint": 0,
    }

    for q in questions:
        constraints = detect_constraints(q)
        if not constraints:
            stats["no_constraint"] += 1
        else:
            text = " ".join(constraints)
            if "百分" in text:
                stats["percentage"] += 1
            elif "分数" in text:
                stats["fraction"] += 1
            elif "向上" in text:
                stats["ceil"] += 1
            elif "向下" in text:
                stats["floor"] += 1
            elif "小数" in text or "取整" in text:
                stats["round"] += 1

    return stats
