"""Behavior tests for repaired CoT GRPO reward policies."""
import sys

sys.path.insert(0, ".")

from src.models.reward import build_reward_funcs, repaired_logic_word_reward_fn


def _total_reward(funcs, completion: str, answer: str = "2", prompt: str = "结果是多少？") -> float:
    return sum(fn([completion], answer=[answer], prompt=[prompt])[0] for fn in funcs)


def test_strict_reward_requires_answer_tag_for_correctness():
    funcs = build_reward_funcs(variant="strict")
    correctness = funcs[0]
    tagged = "<think>首先计算1加1，因此得到2，所以最终答案如下。</think><answer>2</answer>"
    missing_tag = "首先计算1加1，因此得到2，所以答案是2。"

    assert correctness([tagged], answer=["2"], prompt=["结果是多少？"]) == [1.0]
    assert correctness([missing_tag], answer=["2"], prompt=["结果是多少？"]) == [-0.6]
    assert _total_reward(funcs, tagged) > _total_reward(funcs, missing_tag)


def test_balanced_reward_does_not_give_full_credit_without_answer_tag():
    funcs = build_reward_funcs(variant="balanced")
    correctness = funcs[0]
    format_reward = funcs[1]
    tagged = "<think>首先计算1加1，因此得到2，所以最终答案如下。</think><answer>2</answer>"
    missing_tag = "首先计算1加1，因此得到2，所以答案是2。"

    assert correctness([tagged], answer=["2"], prompt=["结果是多少？"]) == [1.0]
    assert correctness([missing_tag], answer=["2"], prompt=["结果是多少？"]) == [0.35]
    assert format_reward([missing_tag]) == [-0.3]
    assert _total_reward(funcs, tagged) > _total_reward(funcs, missing_tag)


def test_logic_words_cannot_beat_wrong_answer():
    funcs = build_reward_funcs(variant="strict")
    correct = "<think>首先计算1加1，因此得到2，所以最终答案如下。</think><answer>2</answer>"
    wrong_with_many_logic_words = (
        "<think>首先已知题意，因为根据条件，设中间量，因此得到一个结果，"
        "由于继续推导，所以那么最终写出答案。</think><answer>3</answer>"
    )

    assert repaired_logic_word_reward_fn([wrong_with_many_logic_words])[0] <= 0.15
    assert _total_reward(funcs, correct) > _total_reward(funcs, wrong_with_many_logic_words)


def test_legacy_reward_remains_default():
    funcs = build_reward_funcs()
    assert [fn.__name__ for fn in funcs][:3] == [
        "correctness_reward_fn",
        "format_reward_fn",
        "logic_word_reward_fn",
    ]


if __name__ == "__main__":
    test_strict_reward_requires_answer_tag_for_correctness()
    test_balanced_reward_does_not_give_full_credit_without_answer_tag()
    test_logic_words_cannot_beat_wrong_answer()
    test_legacy_reward_remains_default()
    print("cot reward policy tests: ALL PASSED")
