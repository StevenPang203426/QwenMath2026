"""
Behavior tests for expression GRPO rewards and expression policy.
"""
import sys

sys.path.insert(0, ".")

from src.models.reward_expr import (
    expr_answer_consistency_fn,
    expr_clean_fn,
    expr_correctness_fn,
    expr_output_cleanliness_fn,
)


def test_expr_rewards_penalize_policy_violations_even_when_answer_matches():
    completions = ["<expr>5%2</expr><answer>1</answer>"]
    assert expr_correctness_fn(completions, answer=["1"]) == [-0.6]
    assert expr_clean_fn(completions) == [-0.3]


def test_expr_correctness_uses_question_aware_normalization():
    completions = ["<expr>16/5</expr><answer>3.2</answer>"]
    assert expr_correctness_fn(completions, answer=["4"], prompt=["至少需要几辆车？"]) == [1.0]


def test_expr_answer_consistency_does_not_replace_eval_correctness():
    completions = ["<expr>16/5</expr><answer>3</answer>"]
    assert expr_correctness_fn(completions, answer=["4"], prompt=["至少需要几辆车？"]) == [1.0]
    assert expr_answer_consistency_fn(completions, prompt=["至少需要几辆车？"]) == [-0.2]


def test_expr_output_cleanliness_penalizes_text_outside_tags():
    clean = ["<expr>2+3</expr><answer>5</answer>"]
    noisy = ["先计算一下。<expr>2+3</expr><answer>5</answer>所以答案是5。"]
    assert expr_output_cleanliness_fn(clean) == [0.0]
    assert expr_output_cleanliness_fn(noisy) == [-0.2]


if __name__ == "__main__":
    test_expr_rewards_penalize_policy_violations_even_when_answer_matches()
    test_expr_correctness_uses_question_aware_normalization()
    test_expr_answer_consistency_does_not_replace_eval_correctness()
    test_expr_output_cleanliness_penalizes_text_outside_tags()
    print("expr reward policy tests: ALL PASSED")
