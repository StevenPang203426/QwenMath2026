"""
Question constraint classifier for adaptive inference prompts.
"""
from __future__ import annotations

from typing import Any

from src.utils.answer_normalizer import detect_answer_format, normalize_question_text


_CONSTRAINT_TEXT = {
    "percentage": "答案写成百分数形式，如25%",
    "fraction": "答案写成分数形式，如3/5，不要化成小数",
    "ceil_integer": "如果结果不是整数，向上取整",
    "floor_integer": "如果结果不是整数，向下取整",
    "integer": "答案写成整数形式，不要带单位",
}


def detect_constraints(question: Any) -> list[str]:
    decision = detect_answer_format(question)
    if decision.answer_type == "round_n":
        if decision.param == 0:
            return ["答案取整数"]
        return [f"答案保留{decision.param}位小数"]
    constraint = _CONSTRAINT_TEXT.get(decision.answer_type)
    return [constraint] if constraint else []


def build_adaptive_prompt(question: Any, base_instruction: str) -> str:
    constraints = detect_constraints(question)
    if not constraints:
        return base_instruction
    return f"{base_instruction}\n【格式约束】{'；'.join(constraints)}。"


def classify_batch(questions: list[Any]) -> dict:
    stats = {
        "total": len(questions),
        "percentage": 0,
        "fraction": 0,
        "ceil": 0,
        "floor": 0,
        "round": 0,
        "no_constraint": 0,
    }
    for question in questions:
        decision = detect_answer_format(normalize_question_text(question))
        if decision.answer_type == "percentage":
            stats["percentage"] += 1
        elif decision.answer_type == "fraction":
            stats["fraction"] += 1
        elif decision.answer_type == "ceil_integer":
            stats["ceil"] += 1
        elif decision.answer_type == "floor_integer":
            stats["floor"] += 1
        elif decision.answer_type in {"round_n", "integer"}:
            stats["round"] += 1
        else:
            stats["no_constraint"] += 1
    return stats
