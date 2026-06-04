"""
Behavior tests for expression GRPO rewards and expression policy.
"""
import sys

sys.path.insert(0, ".")

from src.models.reward_expr import expr_clean_fn, expr_correctness_fn


def test_expr_rewards_penalize_policy_violations_even_when_answer_matches():
    completions = ["<expr>5%2</expr><answer>1</answer>"]
    assert expr_correctness_fn(completions, answer=["1"]) == [-0.5]
    assert expr_clean_fn(completions) == [-0.3]


def test_expr_correctness_uses_question_aware_normalization():
    completions = ["<expr>16/5</expr><answer>3.2</answer>"]
    assert expr_correctness_fn(completions, answer=["4"], prompt=["至少需要几辆车？"]) == [1.0]


if __name__ == "__main__":
    test_expr_rewards_penalize_policy_violations_even_when_answer_matches()
    test_expr_correctness_uses_question_aware_normalization()
    print("expr reward policy tests: ALL PASSED")
