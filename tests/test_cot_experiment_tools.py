"""Tests for CoT repair experiment helper tools."""
import sys

sys.path.insert(0, ".")

from src.analysis.cot_ensemble_offline import (
    parse_candidate_specs,
    evaluate_weighted_vote,
    weighted_vote_for_record,
)
from src.analysis.dpo_audit import audit_dpo_pairs
from src.training.rl_utils import drop_conflicting_grpo_generation_args
from src.utils.config import load_config


def test_offline_weighted_vote_is_stable_and_prompt_specific():
    records = [
        {"id": "1", "question": "结果是多少？", "answer": "2"},
        {"id": "2", "question": "结果是多少？", "answer": "5"},
    ]
    specs = parse_candidate_specs(
        "sft_cot:direct=1.0,grpo:zero_shot_cot=0.35,grpo:few_shot_cot=0.30,dpo:few_shot_cot=0.0"
    )
    details = {
        "sft_cot:direct": {
            "1": {"id": "1", "answer": "2"},
            "2": {"id": "2", "answer": "4"},
        },
        "grpo:zero_shot_cot": {
            "1": {"id": "1", "answer": "3"},
            "2": {"id": "2", "answer": "5"},
        },
        "grpo:few_shot_cot": {
            "1": {"id": "1", "answer": "2"},
            "2": {"id": "2", "answer": "5"},
        },
        "dpo:few_shot_cot": {
            "1": {"id": "1", "answer": "9"},
            "2": {"id": "2", "answer": "9"},
        },
    }

    first_vote = weighted_vote_for_record(records[0], details, specs)
    second_vote = weighted_vote_for_record(records[1], details, specs)
    report = evaluate_weighted_vote(records, details, specs)

    assert first_vote["answer"] == "2"
    assert second_vote["answer"] == "4"
    assert "dpo:few_shot_cot" not in first_vote["winner_sources"]
    assert report["accuracy"] == 0.5
    assert report["oracle_accuracy"] == 1.0
    assert report["source_usage"] == {"sft_cot:direct": 2}


def test_dpo_audit_counts_pair_quality_and_token_truncation():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return " ".join(item["content"] for item in messages) + " assistant"

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": str(text).split()}

    pairs = [
        {
            "id": "1",
            "question": "结果是多少？",
            "instruction": "请一步一步思考。",
            "chosen": "<think>计算</think><answer>2</answer>",
            "rejected": "<think>计算</think><answer>3</answer>",
        },
        {
            "id": "2",
            "question": "结果是多少？",
            "instruction": "请一步一步思考。",
            "chosen": "答案是4",
            "rejected": "<think>计算</think><answer>5</answer>",
        },
    ]
    refs = {
        "1": {"id": "1", "question": "结果是多少？", "answer": "2"},
        "2": {"id": "2", "question": "结果是多少？", "answer": "5"},
    }

    report = audit_dpo_pairs(pairs, reference_by_id=refs, tokenizer=FakeTokenizer(), max_prompt_length=2, max_length=4)

    assert report["counts"]["missing_chosen_answer_tag"] == 1
    assert report["counts"]["chosen_matches_reference"] == 1
    assert report["counts"]["rejected_matches_reference"] == 1
    assert report["counts"]["rejected_correct_chosen_wrong"] == 1
    assert report["counts"]["prompt_over_max_prompt_length"] == 2
    assert report["token_stats"]["enabled"] is True


def test_grpo_generation_schedule_args_are_mutually_exclusive():
    cleaned = drop_conflicting_grpo_generation_args({
        "generation_batch_size": 4,
        "steps_per_generation": 1,
        "temperature": 0.8,
    })

    assert cleaned["generation_batch_size"] == 4
    assert "steps_per_generation" not in cleaned
    assert cleaned["temperature"] == 0.8


def test_smoke_configs_do_not_overwrite_formal_grpo_checkpoint():
    strict = load_config("configs/grpo_cot_reward_strict_smoke.yaml")
    balanced = load_config("configs/grpo_cot_reward_balanced_smoke.yaml")
    full = load_config("configs/grpo_cot_reward_balanced_full.yaml")

    assert strict.reward.variant == "strict"
    assert balanced.reward.variant == "balanced"
    assert full.reward.variant == "balanced"
    assert strict.training.max_steps == 50
    assert balanced.training.max_steps == 50
    assert strict.training.output_dir != "outputs/checkpoints/grpo"
    assert balanced.training.output_dir != "outputs/checkpoints/grpo"
    assert full.training.output_dir == "outputs/checkpoints/grpo_cot_reward_balanced_full"
    assert getattr(strict.grpo, "generation_batch_size", None) == 4
    assert getattr(balanced.grpo, "generation_batch_size", None) == 4
    assert getattr(strict.grpo, "steps_per_generation", None) is None
    assert getattr(balanced.grpo, "steps_per_generation", None) is None


if __name__ == "__main__":
    test_offline_weighted_vote_is_stable_and_prompt_specific()
    test_dpo_audit_counts_pair_quality_and_token_truncation()
    test_grpo_generation_schedule_args_are_mutually_exclusive()
    test_smoke_configs_do_not_overwrite_formal_grpo_checkpoint()
    print("cot experiment tool tests: ALL PASSED")
