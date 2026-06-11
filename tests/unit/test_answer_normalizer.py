"""
Behavior tests for shared answer normalization edge cases.
"""
import sys

sys.path.insert(0, ".")

from src.utils.answer_normalizer import format_auto, normalize_answer_for_question


def test_nonfinite_answers_normalize_to_zero_without_crashing():
    for raw_answer in ["inf", "-inf", "infinity", "nan", "∞"]:
        assert normalize_answer_for_question(raw_answer, "结果是多少？") == "0"

    assert format_auto(float("inf")) == "0"
    assert format_auto(float("nan")) == "0"
    assert format_auto((-1) ** 0.5) == "0"


if __name__ == "__main__":
    test_nonfinite_answers_normalize_to_zero_without_crashing()
    print("answer normalizer tests: ALL PASSED")
