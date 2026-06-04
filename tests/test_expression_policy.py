"""
Behavior tests for the expression safety contract.
"""
import sys

sys.path.insert(0, ".")

from src.data.expr_builder import verify_expression
from src.utils.expression_policy import validate_expression


def test_expression_policy_accepts_only_pure_arithmetic():
    valid = validate_expression("3*(4+5)/2")
    assert valid.ok
    assert valid.normalized == "3*(4+5)/2"

    rejected = {
        "x+3=7": "equation",
        "3 if 2 > 1 else 4": "conditional",
        "3>2": "comparison",
        "abs(-3)+2": "call",
        "round(3.2)": "call",
        "5%2": "disallowed_operator",
        "7//2": "disallowed_operator",
        "a+2": "name",
        "3米+2": "syntax",
        r"\frac{1}{2}": "latex",
    }
    for expr, reason in rejected.items():
        result = validate_expression(expr)
        assert not result.ok, expr
        assert reason in result.reasons, (expr, result.reasons)


def test_expr_builder_rejects_policy_violations_even_when_eval_matches():
    result = verify_expression("5%2", "1", "结果是多少？")
    assert not result["valid"]
    assert result["error"] == "policy_violation: disallowed_operator"


if __name__ == "__main__":
    test_expression_policy_accepts_only_pure_arithmetic()
    test_expr_builder_rejects_policy_violations_even_when_eval_matches()
    print("expression policy tests: ALL PASSED")
