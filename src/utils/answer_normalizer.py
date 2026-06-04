"""
Shared answer normalization for expression data and inference output.

This module is intentionally lightweight: it must not import training
dependencies. The rules are inspired by the Math2 competition post-processing
logic and are centralized here so data repair, augmentation audit, and
inference use the same contract.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from typing import Any


ANSWER_TYPES = [
    "percentage",
    "fraction",
    "round_n",
    "ceil_integer",
    "floor_integer",
    "integer",
    "auto",
]


@dataclass(frozen=True)
class AnswerFormatDecision:
    answer_type: str
    param: int | None = None
    reason: str = ""


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_PCT_KEYWORDS = re.compile(r"百分之几|百分率|百分比|百分数|百分号")
_RATE_KEYWORDS = re.compile(
    r"(?:合格|成活|发芽|出勤|命中|正确|错误|出错|达标|及格|"
    r"利润|增长|下降|节约|浪费|损失|淘汰|录取|中奖|"
    r"浓度|纯度|盐度|含盐|含糖|酒精)率"
)
_FRAC_KEYWORDS = re.compile(
    r"几分之几|几分几之|分率|占[了]?[总全]|是[总全].*的几分之|"
    r"比例|比率|比重|占比"
)
_CEIL_KEYWORDS = re.compile(r"至少|最少|起码")
_CEIL_CONTEXT = re.compile(r"租|船|车|拉完|运完|需要|需|要|限乘|乘坐|准乘|限载|限坐")
_FLOOR_KEYWORDS = re.compile(r"至多|最多|顶多")
_FLOOR_CONTEXT = re.compile(r"能|可以|可|装|分|做|买|坐|乘")
_INTEGER_UNIT = re.compile(
    r"多少[辆人个只条棵支张把本台次箱盒包瓶页桶圈束件副头幅排枝根袋套双对名位艘]|"
    r"几[辆人个只条棵支张把本台次箱盒包瓶页桶圈束件副头幅排枝根袋套双对名位艘]"
)
_ROUND_PATTERNS = [
    (re.compile(r"保留[到]?整数|精确到[个]?位|精确到整"), 0),
    (re.compile(r"保留[到]?[一1][位]小数|精确到0\.1|精确到十分位|保留[到]?小数点后[一1]位"), 1),
    (re.compile(r"保留[到]?[两二2][位]小数|精确到0\.01|精确到百分位|保留[到]?小数点后[两二2]位"), 2),
    (re.compile(r"保留[到]?[三3][位]小数|精确到0\.001|精确到千分位|保留[到]?小数点后[三3]位"), 3),
]

_UNITS = [
    "平方厘米",
    "平方米",
    "立方米",
    "千米",
    "公里",
    "厘米",
    "毫米",
    "千克",
    "公斤",
    "小时",
    "分钟",
    "km",
    "cm",
    "mm",
    "kg",
    "米",
    "克",
    "吨",
    "元",
    "角",
    "分",
    "块",
    "个",
    "只",
    "条",
    "本",
    "台",
    "辆",
    "棵",
    "支",
    "张",
    "把",
    "秒",
    "天",
    "年",
    "月",
    "周",
    "人",
    "名",
    "位",
    "双",
    "对",
    "箱",
    "盒",
    "包",
    "瓶",
    "页",
    "m",
    "g",
]
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")
_LATEX_BRACES = re.compile(r"[{}$]")


def normalize_question_text(question: Any) -> str:
    """Normalize raw question values into plain text."""
    if question is None:
        return ""
    if isinstance(question, list):
        parts = []
        for part in question:
            if isinstance(part, dict):
                parts.append(str(part.get("content", "")))
            else:
                parts.append(str(part))
        return "".join(parts).replace('\\"', '"').strip().strip('"')
    text = str(question).replace('\\"', '"').replace("\\n", "\n").replace("\\t", " ")
    parsed = _try_parse_list_string(text)
    if parsed is not None:
        return normalize_question_text(parsed)
    return text.strip().strip('"')


def _try_parse_list_string(text: str) -> list[Any] | None:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def get_last_clause(text: str) -> str:
    parts = re.split(r"[，。？！；,;?!]", normalize_question_text(text))
    for part in reversed(parts):
        if part.strip():
            return part.strip()
    return normalize_question_text(text)


def detect_answer_type(question: Any) -> tuple[str, int | None]:
    decision = detect_answer_format(question)
    return decision.answer_type, decision.param


def detect_answer_format(question: Any) -> AnswerFormatDecision:
    text = normalize_question_text(question)
    if not text:
        return AnswerFormatDecision("auto", reason="empty_question")
    last_clause = get_last_clause(text)

    if _PCT_KEYWORDS.search(text) or _RATE_KEYWORDS.search(last_clause):
        if not re.search(r"率的|率高|率低", last_clause):
            return AnswerFormatDecision("percentage", reason="percentage_keyword")

    for pattern, digits in _ROUND_PATTERNS:
        if pattern.search(text):
            return AnswerFormatDecision("round_n", digits, "explicit_precision")

    if _FRAC_KEYWORDS.search(last_clause):
        return AnswerFormatDecision("fraction", reason="fraction_keyword")

    if _CEIL_KEYWORDS.search(last_clause):
        if _CEIL_CONTEXT.search(last_clause) or _INTEGER_UNIT.search(last_clause):
            return AnswerFormatDecision("ceil_integer", reason="ceil_discrete")
    if _FLOOR_KEYWORDS.search(last_clause):
        if _FLOOR_CONTEXT.search(last_clause) or _INTEGER_UNIT.search(last_clause):
            return AnswerFormatDecision("floor_integer", reason="floor_discrete")

    if _INTEGER_UNIT.search(last_clause):
        return AnswerFormatDecision("integer", reason="integer_unit")

    return AnswerFormatDecision("auto", reason="default")


def safe_eval_expression(expr: str) -> float:
    if not expr or not str(expr).strip():
        raise ValueError("Empty expression")
    expr_clean = normalize_expression(expr)
    tree = ast.parse(expr_clean, mode="eval")
    return _eval_node(tree.body)


def normalize_expression(expr: str) -> str:
    return str(expr).strip().replace("^", "**").replace("×", "*").replace("÷", "/")


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported binary op: {op_type}")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero")
        return _SAFE_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary op: {op_type}")
        return _SAFE_OPS[op_type](_eval_node(node.operand))
    raise ValueError(f"Unsupported node: {type(node).__name__}")


def parse_numeric(answer: str) -> float | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    pct_match = re.match(r"^(-?\d+\.?\d*)%$", text)
    if pct_match:
        return float(pct_match.group(1))
    frac_match = re.match(r"^(-?\d+)/(\d+)$", text)
    if frac_match:
        numerator = int(frac_match.group(1))
        denominator = int(frac_match.group(2))
        return numerator / denominator if denominator != 0 else None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def normalize_answer_for_question(raw_answer: str, question: Any) -> str:
    if raw_answer is None or not str(raw_answer).strip():
        return "0"
    raw_text = str(raw_answer).strip()
    answer = basic_clean_answer(raw_text)
    if re.search(r"\d+\s*月\s*\d+", raw_text):
        return final_clean_answer(answer)

    decision = detect_answer_format(question)
    if _is_already_formatted(answer, decision.answer_type):
        return final_clean_answer(answer)

    value = parse_numeric(answer)
    if value is None:
        return final_clean_answer(answer)

    if decision.answer_type == "percentage":
        return format_percentage(value, normalize_question_text(question))
    if decision.answer_type == "fraction":
        return format_fraction(value)
    if decision.answer_type == "round_n":
        return format_rounded(value, decision.param or 0)
    if decision.answer_type == "ceil_integer":
        return str(math.ceil(value)) if value != int(value) else str(int(value))
    if decision.answer_type == "floor_integer":
        return str(math.floor(value)) if value != int(value) else str(int(value))
    if decision.answer_type == "integer":
        return str(int(value)) if value == int(value) else str(round(value))
    return format_auto(value)


def _is_already_formatted(answer: str, answer_type: str) -> bool:
    if answer_type == "percentage" and answer.endswith("%"):
        return True
    if answer_type == "fraction" and "/" in answer and "%" not in answer:
        return True
    return False


def format_percentage(value: float, question: Any) -> str:
    if value > 1:
        pct_value = value
    elif 0 < value <= 1:
        pct_value = value * 100
    elif value == 0:
        pct_value = 0
    else:
        pct_value = value

    digits = detect_percentage_precision(question)
    if digits is not None:
        formatted = f"{pct_value:.{digits}f}"
    elif pct_value == int(pct_value):
        formatted = str(int(pct_value))
    else:
        formatted = f"{pct_value:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def detect_percentage_precision(question: Any) -> int | None:
    text = normalize_question_text(question)
    if re.search(r"保留整数|保留到1%|精确到1%|保留到百分之一|精确到百分之一", text):
        return 0
    if re.search(r"保留[一1]位|精确到0\.1%", text):
        return 1
    if re.search(r"保留[两二2]位|精确到0\.01%", text):
        return 2
    return None


def format_fraction(value: float) -> str:
    if value == int(value):
        return str(int(value))
    frac = Fraction(value).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def format_rounded(value: float, digits: int) -> str:
    if digits == 0:
        rounded = Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return str(int(rounded))
    quantize_str = "0." + "0" * digits
    rounded = Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    return str(rounded)


def format_auto(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def basic_clean_answer(answer: str) -> str:
    text = str(answer).replace("\n", " ").strip()
    text = re.sub(r"(\d+)\s*月\s*(\d+)\s*日?", r"\1/\2", text)
    for unit in sorted(_UNITS, key=len, reverse=True):
        text = text.replace(unit, "")
    text = _LATEX_CMD.sub("", text)
    text = _LATEX_BRACES.sub("", text)
    return text.strip()


def final_clean_answer(answer: str) -> str:
    text = str(answer).strip()
    if re.match(r"^-?\d+\.?\d*%?$", text):
        return text
    if re.match(r"^-?\d+/\d+$", text):
        return text
    nums = re.findall(r"-?\d+\.?\d*(?:/\d+)?%?", text)
    if nums:
        return nums[-1]
    return text


def numeric_answer_value(answer: str) -> float | None:
    return parse_numeric(final_clean_answer(basic_clean_answer(str(answer))))


def answers_match(a: Any, b: Any, tol: float = 1e-4) -> bool:
    if str(a).strip() == str(b).strip():
        return True
    va = numeric_answer_value(str(a))
    vb = numeric_answer_value(str(b))
    return va is not None and vb is not None and abs(va - vb) < tol


def answers_match_by_question(raw_answer: Any, gold_answer: Any, question: Any, tol: float = 1e-4) -> bool:
    normalized = normalize_answer_for_question(str(raw_answer), question)
    if str(normalized).strip() == str(gold_answer).strip():
        return True
    return answers_match(normalized, gold_answer, tol=tol)
