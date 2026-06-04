"""
Expression safety contract shared by data repair, rewards, and inference.

The public interface is deliberately small: callers ask whether an expression
is safe to trust as a pure arithmetic expression, and get stable reasons when
it is not.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpressionPolicyResult:
    ok: bool
    normalized: str
    reasons: tuple[str, ...]


_LATEX_PATTERN = re.compile(r"\\[a-zA-Z]+|[{}$]")
_CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_SINGLE_EQ_PATTERN = re.compile(r"(?<![<>=!])=(?!=)")
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def validate_expression(expr: object) -> ExpressionPolicyResult:
    """Validate that expr is pure arithmetic: numbers plus + - * / ** ()."""
    normalized = _normalize_text(expr)
    reasons: list[str] = []

    if not normalized:
        return ExpressionPolicyResult(False, normalized, ("empty",))
    if _LATEX_PATTERN.search(normalized):
        reasons.append("latex")
    if _SINGLE_EQ_PATTERN.search(normalized):
        reasons.append("equation")

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        reasons.append("syntax")
        return _result(normalized, reasons)

    _visit(tree.body, reasons)
    return _result(normalized, reasons)


def is_safe_expression(expr: object) -> bool:
    return validate_expression(expr).ok


def _normalize_text(expr: object) -> str:
    return str(expr or "").strip()


def _result(normalized: str, reasons: list[str]) -> ExpressionPolicyResult:
    deduped = tuple(dict.fromkeys(reasons))
    return ExpressionPolicyResult(not deduped, normalized, deduped)


def _visit(node: ast.AST, reasons: list[str]) -> None:
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            reasons.append("constant")
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            reasons.append("disallowed_operator")
        _visit(node.left, reasons)
        _visit(node.right, reasons)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            reasons.append("disallowed_operator")
        _visit(node.operand, reasons)
        return

    if isinstance(node, ast.IfExp):
        reasons.append("conditional")
        _visit(node.body, reasons)
        _visit(node.test, reasons)
        _visit(node.orelse, reasons)
        return

    if isinstance(node, ast.Compare):
        reasons.append("comparison")
        _visit(node.left, reasons)
        for comparator in node.comparators:
            _visit(comparator, reasons)
        return

    if isinstance(node, ast.Call):
        reasons.append("call")
        for arg in node.args:
            _visit(arg, reasons)
        return

    if isinstance(node, ast.Name):
        if _CHINESE_PATTERN.search(node.id):
            reasons.append("syntax")
        else:
            reasons.append("name")
        return

    if isinstance(node, ast.BoolOp):
        reasons.append("boolean")
        for value in node.values:
            _visit(value, reasons)
        return

    reasons.append(type(node).__name__.lower())
