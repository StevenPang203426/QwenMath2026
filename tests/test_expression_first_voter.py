"""
Behavior tests for expression-first voting.
"""
import sys

sys.path.insert(0, ".")

from src.inference.expression_first_voter import vote_expression_first


def test_expression_first_vote_prefers_safe_expression_majority_over_cot():
    result = vote_expression_first(
        question="至少需要几辆车？",
        expr_candidates=[
            {"source": "expr_sft_t0.1", "expression": "16/5", "weight": 1.0},
            {"source": "expr_grpo_t0.3", "expression": "3.2", "weight": 1.5},
            {"source": "expr_bad", "expression": "ceil(3.2)", "weight": 2.0},
        ],
        cot_candidates=[{"source": "cot_grpo", "answer": "3", "weight": 2.0}],
    )

    assert result["answer"] == "4"
    assert result["source"] == "expression_majority"
    assert result["safe_expression_candidates"] == 2
    assert result["rejected_expression_candidates"] == 1


def test_expression_first_vote_uses_cot_to_break_expression_ties_and_fallback():
    tied = vote_expression_first(
        question="至少需要几辆车？",
        expr_candidates=[
            {"source": "expr_a", "expression": "16/5"},
            {"source": "expr_b", "expression": "5"},
        ],
        cot_candidates=[{"source": "cot", "answer": "4"}],
    )
    assert tied["answer"] == "4"
    assert tied["source"] == "cot_verified_expression"

    fallback = vote_expression_first(
        question="结果是多少？",
        expr_candidates=[{"source": "expr_bad", "expression": "x+1=3"}],
        cot_candidates=[{"source": "cot", "answer": "12"}],
    )
    assert fallback["answer"] == "12"
    assert fallback["source"] == "cot_fallback"


if __name__ == "__main__":
    test_expression_first_vote_prefers_safe_expression_majority_over_cot()
    test_expression_first_vote_uses_cot_to_break_expression_ties_and_fallback()
    print("expression-first voter tests: ALL PASSED")
