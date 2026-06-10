"""
Behavior tests for expression prediction safety.
"""
import sys

sys.path.insert(0, ".")

from src.inference.expr_predictor import expr_predict_single


def test_expr_predict_single_rejects_complex_eval_result_without_crashing():
    result = expr_predict_single(
        "<expr>(-1)**0.5</expr><answer>0</answer>",
        "结果是多少？",
    )

    assert result == {
        "answer": "0",
        "expression": "(-1)**0.5",
        "eval_success": False,
        "source": "answer_tag_fallback",
    }


if __name__ == "__main__":
    test_expr_predict_single_rejects_complex_eval_result_without_crashing()
    print("expr predictor tests: ALL PASSED")
