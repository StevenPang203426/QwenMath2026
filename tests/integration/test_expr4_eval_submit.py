"""
Behavior tests for 4-expression voting with CoT soft calibration.
"""
import sys

sys.path.insert(0, ".")

from src.inference.expr4_eval_submit import (
    COT_WEIGHTS,
    EXPR_WEIGHT_PRESETS,
    VotingStrategy,
    answer_close_by_question,
    answer_from_expression,
    details_complete_for_records,
    evaluate_strategies,
    vote_expr4_soft_calibrated,
)


PERCENT_QUESTION = "某班合格率是多少？结果写成百分数。"


def _strategy(
    expr_weights=None,
    cot_bonus_scale=0.3,
):
    return VotingStrategy(
        preset="test",
        expr_weights=expr_weights or dict(EXPR_WEIGHT_PRESETS["equal"]),
        cot_weights=dict(COT_WEIGHTS),
        cot_bonus_scale=cot_bonus_scale,
    )


def test_cot_percent_forms_match_expression_answer_by_numeric_tolerance():
    assert answer_close_by_question("25%", "0.25", PERCENT_QUESTION)
    assert answer_close_by_question("25%", "25", PERCENT_QUESTION)
    assert answer_close_by_question("25%", "25%", PERCENT_QUESTION)

    result = vote_expr4_soft_calibrated(
        question=PERCENT_QUESTION,
        expr_candidates=[
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "1/4"},
        ],
        cot_candidates=[
            {"model": "cot_sft", "source": "cot_sft", "answer": "0.25"},
            {"model": "cot_dpo", "source": "cot_dpo", "answer": "25"},
            {"model": "cot_grpo", "source": "cot_grpo", "answer": "25%"},
        ],
        strategy=_strategy(),
    )

    assert result["answer"] == "25%"
    assert result["details"]["cot_support_count"] == 3
    assert result["details"]["cot_support_models"] == ["cot_sft", "cot_dpo", "cot_grpo"]


def test_cot_support_cannot_override_expression_majority():
    result = vote_expr4_soft_calibrated(
        question="结果是多少？",
        expr_candidates=[
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "1+1"},
            {"model": "dpo_expr_clean", "source": "dpo_expr_clean_t0.1", "expression": "4/2"},
            {"model": "grpo_expr_clean_vllm", "source": "grpo_expr_clean_vllm_t0.1", "expression": "3"},
        ],
        cot_candidates=[
            {"model": "cot_sft", "source": "cot_sft", "answer": "3"},
            {"model": "cot_dpo", "source": "cot_dpo", "answer": "3"},
            {"model": "cot_grpo", "source": "cot_grpo", "answer": "3"},
        ],
        strategy=_strategy(cot_bonus_scale=0.5),
    )

    assert result["answer"] == "2"
    assert result["details"]["expr_count"] == 2
    assert result["fallback"] is False


def test_cot_dpo_participates_as_soft_bonus_when_expression_votes_tie():
    result = vote_expr4_soft_calibrated(
        question="结果是多少？",
        expr_candidates=[
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "1+1"},
            {"model": "dpo_expr_clean", "source": "dpo_expr_clean_t0.1", "expression": "1+2"},
        ],
        cot_candidates=[
            {"model": "cot_dpo", "source": "cot_dpo", "answer": "3"},
        ],
        strategy=_strategy(cot_bonus_scale=0.5),
    )

    assert result["answer"] == "3"
    assert result["details"]["cot_support_models"] == ["cot_dpo"]
    assert result["source"] == "expression_cot_calibrated"


def test_cot_fallback_only_when_no_safe_expression_candidate():
    result = vote_expr4_soft_calibrated(
        question="结果是多少？",
        expr_candidates=[
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "x+1=3"},
            {"model": "dpo_expr_clean", "source": "dpo_expr_clean_t0.1", "expression": "not an expr"},
        ],
        cot_candidates=[
            {"model": "cot_sft", "source": "cot_sft", "answer": "7"},
            {"model": "cot_dpo", "source": "cot_dpo", "answer": "7"},
        ],
        strategy=_strategy(),
    )

    assert result["answer"] == "7"
    assert result["source"] == "cot_fallback"
    assert result["fallback"] is True


def test_complex_expression_is_rejected_without_crashing_vote():
    assert answer_from_expression("(-1)**0.5", "结果是多少？") is None

    result = vote_expr4_soft_calibrated(
        question="结果是多少？",
        expr_candidates=[
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "(-1)**0.5"},
            {"model": "dpo_expr_clean", "source": "dpo_expr_clean_t0.1", "expression": "1+1"},
        ],
        cot_candidates=[],
        strategy=_strategy(),
    )

    assert result["answer"] == "2"
    assert result["safe_expression_candidates"] == 1
    assert result["rejected_expression_candidates"] == 1


def test_strategy_grid_selection_tie_break_prefers_smaller_cot_bonus():
    records = [{"id": "1", "question": "结果是多少？", "answer": "2"}]
    expr_by_id = {
        "1": [
            {"model": "sft_expr_clean", "source": "sft_expr_clean_t0.1", "expression": "1+1"},
        ]
    }
    cot_by_id = {"1": [{"model": "cot_dpo", "source": "cot_dpo", "answer": "2"}]}

    selected, metrics = evaluate_strategies(records, expr_by_id, cot_by_id)

    assert len(metrics) == len(EXPR_WEIGHT_PRESETS) * 4
    assert selected["accuracy"] == 1.0
    assert selected["fallback"] == 0
    assert selected["strategy"]["cot_bonus_scale"] == 0.0


def test_details_complete_rejects_stale_limit_files():
    records = [{"id": str(i), "question": "结果是多少？"} for i in range(10)]
    stale_cot_details = [{"id": str(i), "answer": "1"} for i in range(8)]
    full_cot_details = [{"id": str(i), "answer": "1"} for i in range(10)]
    full_expr_details = [
        {"id": str(i), "answer": "1", "temperature": temperature}
        for i in range(10)
        for temperature in [0.1, 0.3, 0.7]
    ]

    assert details_complete_for_records(stale_cot_details, records, 1) is False
    assert details_complete_for_records(full_cot_details, records, 1) is True
    assert details_complete_for_records(full_expr_details, records, 3) is True


if __name__ == "__main__":
    test_cot_percent_forms_match_expression_answer_by_numeric_tolerance()
    test_cot_support_cannot_override_expression_majority()
    test_cot_dpo_participates_as_soft_bonus_when_expression_votes_tie()
    test_cot_fallback_only_when_no_safe_expression_candidate()
    test_complex_expression_is_rejected_without_crashing_vote()
    test_strategy_grid_selection_tie_break_prefers_smaller_cot_bonus()
    test_details_complete_rejects_stale_limit_files()
    print("expr4 eval submit tests: ALL PASSED")
