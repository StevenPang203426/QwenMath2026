"""
Behavior tests for inference-stage few-shot CoT prompt ablation.
"""
import sys

sys.path.insert(0, ".")

from src.inference.cot_prompt_ablation import (
    build_parser,
    build_prompt_disagreements,
    build_prompt_messages,
    exact_sign_test_p_value,
    filter_complete_details_for_records,
    generate_raw_outputs_vllm_batch,
    merge_detail_records,
    paired_summary,
    reusable_details_for_records,
    score_details,
)


def _detail(item_id: str, answer: str, prompt: str = "direct", model: str = "dpo") -> dict:
    return {
        "id": item_id,
        "question": "结果是多少？",
        "model": model,
        "prompt": prompt,
        "raw_output": f"<answer>{answer}</answer>",
        "raw_answer": answer,
        "answer": answer,
        "extraction_level": "标签匹配",
        "has_answer_tag": True,
        "empty_extraction": False,
        "prompt_tokens": 10,
        "raw_length_chars": 20,
    }


def test_prompt_builders_change_only_prompt_shape():
    question = "小明有2个苹果，又买了3个，一共有几个？"

    direct = build_prompt_messages("direct", question)
    zero = build_prompt_messages("zero_shot_cot", question)
    few = build_prompt_messages("few_shot_cot", question)

    assert len(direct) == 2
    assert [item["role"] for item in direct] == ["system", "user"]
    assert len(zero) == 2
    assert [item["role"] for item in zero] == ["system", "user"]
    assert len([item for item in few if item["role"] == "assistant"]) == 3
    assert few[-1] == {"role": "user", "content": question}


def test_score_details_uses_question_aware_answer_matching():
    records = [
        {"id": "1", "question": "某班合格率是多少？结果写成百分数。", "answer": "25%"},
        {"id": "2", "question": "结果是多少？", "answer": "3"},
    ]
    details = [
        _detail("1", "0.25"),
        _detail("2", "3"),
    ]

    score = score_details(records, details)

    assert score["accuracy"] == 1.0
    assert score["correct"] == 2
    assert score["malformed_output_count"] == 0
    assert score["empty_extraction_count"] == 0
    assert score["avg_prompt_tokens"] == 10


def test_paired_summary_counts_prompt_wins_and_sign_test():
    records = [
        {"id": "1", "question": "结果是多少？", "answer": "2"},
        {"id": "2", "question": "结果是多少？", "answer": "5"},
        {"id": "3", "question": "结果是多少？", "answer": "9"},
        {"id": "4", "question": "结果是多少？", "answer": "10"},
    ]
    direct = [_detail("1", "2"), _detail("2", "4"), _detail("3", "9"), _detail("4", "1")]
    few = [_detail("1", "2"), _detail("2", "5"), _detail("3", "8"), _detail("4", "1")]

    summary = paired_summary(records, direct, few, "dpo:direct", "dpo:few_shot_cot")

    assert summary["both_correct"] == 1
    assert summary["both_wrong"] == 1
    assert summary["a_only_correct"] == 1
    assert summary["b_only_correct"] == 1
    assert summary["net_b_wins"] == 0
    assert exact_sign_test_p_value(1, 1) == 1.0


def test_disagreements_include_answer_or_correctness_changes():
    records = [
        {"id": "1", "question": "结果是多少？", "answer": "2"},
        {"id": "2", "question": "结果是多少？", "answer": "5"},
    ]
    details_by_prompt = {
        "direct": [_detail("1", "2", "direct"), _detail("2", "4", "direct")],
        "zero_shot_cot": [_detail("1", "2", "zero_shot_cot"), _detail("2", "4", "zero_shot_cot")],
        "few_shot_cot": [_detail("1", "2", "few_shot_cot"), _detail("2", "5", "few_shot_cot")],
    }

    rows = build_prompt_disagreements(records, details_by_prompt, "dpo", has_gold=True)

    assert len(rows) == 1
    assert rows[0]["id"] == "2"
    assert rows[0]["direct_correct"] is False
    assert rows[0]["few_shot_cot_correct"] is True



def test_resume_helpers_reuse_existing_rows_and_keep_missing_records():
    records = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    existing = [
        _detail("1", "1", "direct", "grpo"),
        _detail("2", "2", "few_shot_cot", "grpo"),
        _detail("4", "4", "direct", "grpo"),
        _detail("3", "3", "direct", "grpo"),
    ]

    reusable, missing = reusable_details_for_records(existing, records, "grpo", "direct")

    assert [item["id"] for item in reusable] == ["1", "3"]
    assert [item["id"] for item in missing] == ["2"]

    merged = merge_detail_records(existing, [_detail("2", "22", "direct", "grpo")], "grpo", "direct")
    assert {item["id"] for item in merged} == {"1", "2", "3", "4"}
    assert [item for item in merged if item["id"] == "2" and item["prompt"] == "direct"][0]["answer"] == "22"


def test_vllm_generation_uses_lora_request_and_counts_prompt_tokens():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "|".join(f"{item['role']}:{item['content']}" for item in messages)

        def __call__(self, prompts, add_special_tokens=False):
            return {"input_ids": [prompt.split("|") for prompt in prompts]}

    class FakeChoice:
        def __init__(self, text):
            self.text = text

    class FakeOutput:
        def __init__(self, text):
            self.outputs = [FakeChoice(text)]

    class FakeLLM:
        def __init__(self):
            self.calls = []

        def generate(self, prompts, sampling_params, use_tqdm, lora_request):
            self.calls.append({
                "prompts": prompts,
                "sampling_params": sampling_params,
                "use_tqdm": use_tqdm,
                "lora_request": lora_request,
            })
            return [FakeOutput(f"<answer>{idx}</answer>") for idx, _ in enumerate(prompts)]

    lora_request = object()
    llm = FakeLLM()
    raw_outputs, prompt_tokens = generate_raw_outputs_vllm_batch(
        llm=llm,
        tokenizer=FakeTokenizer(),
        questions=["题目1", "题目2"],
        prompt_name="direct",
        max_new_tokens=32,
        temperature=0.1,
        do_sample=False,
        lora_request=lora_request,
        sampling_params_factory=lambda **kwargs: kwargs,
    )

    assert raw_outputs == ["<answer>0</answer>", "<answer>1</answer>"]
    assert prompt_tokens == [2, 2]
    assert llm.calls[0]["lora_request"] is lora_request
    assert llm.calls[0]["sampling_params"] == {
        "max_new_tokens": 32,
        "temperature": 0.1,
        "do_sample": False,
    }


def test_parser_defaults_to_vllm_with_large_batch_for_ablation():
    args = build_parser().parse_args([])
    assert args.engine == "vllm"
    assert args.batch_size == 64


def test_cached_full_details_can_cover_limited_subset():
    records = [{"id": "1"}, {"id": "2"}]
    details = [
        _detail("1", "1", "direct", "grpo"),
        _detail("2", "2", "direct", "grpo"),
        _detail("3", "3", "direct", "grpo"),
    ]

    matching = filter_complete_details_for_records(details, records, "grpo", "direct")

    assert matching is not None
    assert [item["id"] for item in matching] == ["1", "2"]


if __name__ == "__main__":
    test_prompt_builders_change_only_prompt_shape()
    test_score_details_uses_question_aware_answer_matching()
    test_paired_summary_counts_prompt_wins_and_sign_test()
    test_disagreements_include_answer_or_correctness_changes()
    test_resume_helpers_reuse_existing_rows_and_keep_missing_records()
    test_vllm_generation_uses_lora_request_and_counts_prompt_tokens()
    test_parser_defaults_to_vllm_with_large_batch_for_ablation()
    test_cached_full_details_can_cover_limited_subset()
    print("cot prompt ablation tests: ALL PASSED")
