"""
Behavior tests for expression GRPO DPO ablation.
"""
import sys

sys.path.insert(0, ".")

from src.inference.expr_grpo_ablation import (
    build_disagreements,
    discordance_summary,
    exact_sign_test_p_value,
    filter_complete_details_for_records,
    record_id_sort_key,
    score_model_report,
    vote_temperature_candidates,
)


def _detail(item_id: str, expression: str, temperature: float) -> dict:
    return {
        "id": item_id,
        "expression": expression,
        "temperature": temperature,
        "source": f"m_t{temperature:g}",
        "model": "m",
    }


def test_temperature_vote_uses_safe_expression_majority():
    result = vote_temperature_candidates(
        "结果是多少？",
        [
            _detail("1", "1+1", 0.1),
            _detail("1", "4/2", 0.3),
            _detail("1", "3", 0.7),
        ],
    )

    assert result["answer"] == "2"
    assert result["safe_expression_candidates"] == 3
    assert result["details"]["winner"]["count"] == 2


def test_temperature_vote_tie_prefers_lowest_temperature_candidate():
    result = vote_temperature_candidates(
        "结果是多少？",
        [
            _detail("1", "1+1", 0.1),
            _detail("1", "1+2", 0.3),
            _detail("1", "1+3", 0.7),
        ],
    )

    assert result["answer"] == "2"
    assert result["details"]["winner"]["temperatures"] == [0.1]


def test_temperature_vote_rejects_unsafe_or_non_real_expressions():
    result = vote_temperature_candidates(
        "结果是多少？",
        [
            _detail("1", "x+1=3", 0.1),
            _detail("1", "(-1)**0.5", 0.3),
            _detail("1", "1+1", 0.7),
        ],
    )

    assert result["answer"] == "2"
    assert result["safe_expression_candidates"] == 1
    assert result["rejected_expression_candidates"] == 2


def test_temperature_vote_falls_back_when_no_safe_expression_exists():
    result = vote_temperature_candidates(
        "结果是多少？",
        [
            _detail("1", "x+1=3", 0.1),
            _detail("1", "not an expression", 0.3),
        ],
    )

    assert result["answer"] == "0"
    assert result["source"] == "no_safe_expression"
    assert result["fallback"] is True


def test_cached_full_details_can_cover_limited_record_subset():
    records = [{"id": "1"}, {"id": "2"}]
    details = [
        _detail("1", "1+1", 0.1),
        _detail("1", "1+1", 0.3),
        _detail("1", "1+1", 0.7),
        _detail("2", "1+2", 0.1),
        _detail("2", "1+2", 0.3),
        _detail("2", "1+2", 0.7),
        _detail("3", "1+3", 0.1),
        _detail("3", "1+3", 0.3),
        _detail("3", "1+3", 0.7),
    ]

    matching = filter_complete_details_for_records(details, records)

    assert matching is not None
    assert [item["id"] for item in matching] == ["1", "1", "1", "2", "2", "2"]


def test_record_id_sort_key_handles_numeric_and_string_ids():
    assert sorted(["10", "2", "repair_1", "aug_1"], key=record_id_sort_key) == [
        "2",
        "10",
        "aug_1",
        "repair_1",
    ]


def test_validation_metrics_and_dpo_discordance_summary():
    records = [
        {"id": "1", "question": "结果是多少？", "answer": "2"},
        {"id": "2", "question": "结果是多少？", "answer": "5"},
        {"id": "3", "question": "结果是多少？", "answer": "9"},
    ]
    single_report = [
        {"id": "1", "answer": "2", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 3, "rejected_expression_candidates": 0},
        {"id": "2", "answer": "4", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 3, "rejected_expression_candidates": 0},
        {"id": "3", "answer": "9", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 2, "rejected_expression_candidates": 1},
    ]
    dpo_report = [
        {"id": "1", "answer": "2", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 3, "rejected_expression_candidates": 0},
        {"id": "2", "answer": "5", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 3, "rejected_expression_candidates": 0},
        {"id": "3", "answer": "8", "source": "expression_temperature_vote", "fallback": False, "safe_expression_candidates": 3, "rejected_expression_candidates": 0},
    ]

    single_score = score_model_report(records, single_report)
    dpo_score = score_model_report(records, dpo_report)
    disagreements = build_disagreements(records, single_report, dpo_report, has_gold=True)
    summary = discordance_summary(disagreements)

    assert single_score["correct"] == 2
    assert dpo_score["correct"] == 2
    assert [item["outcome"] for item in disagreements] == ["dpo_win", "single_grpo_win"]
    assert summary["dpo_wins"] == 1
    assert summary["single_grpo_wins"] == 1
    assert summary["net_dpo_wins"] == 0
    assert exact_sign_test_p_value(1, 1) == 1.0


if __name__ == "__main__":
    test_temperature_vote_uses_safe_expression_majority()
    test_temperature_vote_tie_prefers_lowest_temperature_candidate()
    test_temperature_vote_rejects_unsafe_or_non_real_expressions()
    test_temperature_vote_falls_back_when_no_safe_expression_exists()
    test_cached_full_details_can_cover_limited_record_subset()
    test_record_id_sort_key_handles_numeric_and_string_ids()
    test_validation_metrics_and_dpo_discordance_summary()
    print("expr grpo ablation tests: ALL PASSED")
