"""
Inference-stage prompt ablation for CoT-trained SFT/DPO/GRPO adapters.

The intended controlled variable is the prompt shape only:
- direct: ask for final answer only
- zero_shot_cot: ask for step-by-step reasoning
- few_shot_cot: prepend the existing few-shot CoT demonstrations
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Callable

from tqdm import tqdm

from src.data.answer_extractor import extract_answer
from src.inference.cot_prompting import build_few_shot_prompt, build_zero_shot_prompt
from src.utils.answer_normalizer import (
    answers_match_by_question,
    detect_answer_format,
    normalize_answer_for_question,
    normalize_question_text,
)

logger = logging.getLogger("math_solver.cot_prompt_ablation")

MODEL_SPECS: dict[str, dict[str, str]] = {
    "sft_cot": {"name": "sft_cot", "adapter_path": "outputs/checkpoints/sft_cot/best"},
    "dpo": {"name": "dpo", "adapter_path": "outputs/checkpoints/dpo/best"},
    "grpo": {"name": "grpo", "adapter_path": "outputs/checkpoints/grpo/best"},
}

PROMPT_NAMES = ["direct", "zero_shot_cot", "few_shot_cot"]
ENGINE_NAMES = ["vllm", "hf"]

DIRECT_INSTRUCTION = (
    "请直接给出最终答案，不要展开推理过程。"
    "只使用<answer></answer>标签包裹最终答案。"
)

ZERO_SHOT_COT_ALIAS = "zero_shot_cot"
FEW_SHOT_COT_ALIAS = "few_shot_cot"


def load_records(path: str, limit: int = 0) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit and limit > 0:
        return records[:limit]
    return records


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: str | Path, rows: list[list[Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_dict_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_base_model_name() -> str:
    from src.utils.config import load_config

    return load_config("configs/base.yaml").model.name


def parse_names(raw: str, available: Iterable[str], label: str) -> list[str]:
    available_list = list(available)
    if not raw or raw.lower() == "all":
        return available_list
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in names if item not in available_list]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; available={available_list}")
    return names


def build_prompt_messages(prompt_name: str, question: Any) -> list[dict[str, str]]:
    question_text = normalize_question_text(question)
    if prompt_name == "direct":
        return [
            {"role": "system", "content": DIRECT_INSTRUCTION},
            {"role": "user", "content": question_text},
        ]
    if prompt_name == ZERO_SHOT_COT_ALIAS:
        return build_zero_shot_prompt(question_text)
    if prompt_name == FEW_SHOT_COT_ALIAS:
        return build_few_shot_prompt(question_text)
    raise ValueError(f"Unknown prompt_name: {prompt_name}")


def resolve_checkpoint_path(adapter_path: str) -> str | None:
    path = Path(adapter_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        return str(path)

    parent = path.parent if path.name == "best" else path
    if parent.is_dir():
        checkpoints = sorted(
            [item for item in parent.iterdir() if item.is_dir() and item.name.startswith("checkpoint-")],
            key=lambda item: int(item.name.split("-")[-1]) if item.name.split("-")[-1].isdigit() else -1,
        )
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resolved %s to latest checkpoint %s", adapter_path, latest)
            return str(latest)
    return None


def ensure_checkpoint(spec: dict[str, str]) -> str:
    adapter_path = resolve_checkpoint_path(spec["adapter_path"])
    if not adapter_path:
        raise FileNotFoundError(f"Missing checkpoint for {spec['name']}: {spec['adapter_path']}")
    return adapter_path


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_chat_prompts(tokenizer: Any, questions: list[str], prompt_name: str) -> list[str]:
    prompts = []
    for question in questions:
        messages = build_prompt_messages(prompt_name, question)
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def count_prompt_tokens(tokenizer: Any, prompts: list[str]) -> list[int]:
    if not prompts:
        return []
    encoded = tokenizer(prompts, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return [len(ids) for ids in input_ids]


def build_vllm_sampling_params(max_new_tokens: int, temperature: float, do_sample: bool):
    from vllm import SamplingParams

    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if do_sample else 0.0,
    )


def generate_raw_outputs_vllm_batch(
    llm: Any,
    tokenizer: Any,
    questions: list[str],
    prompt_name: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    lora_request: Any,
    show_progress: bool,
    sampling_params_factory: Callable[..., Any] = build_vllm_sampling_params,
) -> tuple[list[str], list[int]]:
    if not questions:
        return [], []
    prompts = build_chat_prompts(tokenizer, questions, prompt_name)
    prompt_tokens = count_prompt_tokens(tokenizer, prompts)
    sampling_params = sampling_params_factory(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )
    outputs = llm.generate(
        prompts,
        sampling_params=sampling_params,
        use_tqdm=show_progress,
        lora_request=lora_request,
    )
    raw_outputs = [item.outputs[0].text if item.outputs else "" for item in outputs]
    return raw_outputs, prompt_tokens


def load_vllm_model_and_tokenizer(
    base_model_name: str,
    gpu_memory_utilization: float,
    max_lora_rank: int,
    max_model_len: int,
    max_num_seqs: int,
    enforce_eager: bool,
) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=base_model_name,
        tokenizer=base_model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=max_lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        disable_log_stats=True,
    )
    return llm, tokenizer


def build_lora_request(model_name: str, adapter_path: str, lora_int_id: int) -> Any:
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        lora_name=model_name,
        lora_int_id=lora_int_id,
        lora_path=adapter_path,
    )


def generate_raw_outputs_batch(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    prompt_name: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> tuple[list[str], list[int]]:
    if not questions:
        return [], []

    import torch

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = build_chat_prompts(tokenizer, questions, prompt_name)
    device = next(model.parameters()).device
    try:
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_tokens = [int(count) for count in model_inputs.attention_mask.sum(dim=1).tolist()]
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_length = model_inputs.input_ids.shape[1]
        raw_outputs = [
            tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True)
            for output_ids in generated_ids
        ]
        return raw_outputs, prompt_tokens
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(questions) == 1:
            raise
        logger.warning("Batch generation OOM; splitting batch size=%s", len(questions))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(questions) // 2
        left_outputs, left_tokens = generate_raw_outputs_batch(
            model, tokenizer, questions[:mid], prompt_name, max_new_tokens, temperature, do_sample
        )
        right_outputs, right_tokens = generate_raw_outputs_batch(
            model, tokenizer, questions[mid:], prompt_name, max_new_tokens, temperature, do_sample
        )
        return left_outputs + right_outputs, left_tokens + right_tokens


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def normalize_answer_safe(raw_answer: Any, question: Any) -> str:
    try:
        return normalize_answer_for_question(str(raw_answer), question)
    except Exception as exc:
        logger.warning("Answer normalization failed; fallback to 0: %s", exc)
        return "0"


def detail_file(output_dir: str | Path, prefix: str, model_name: str, prompt_name: str) -> Path:
    return Path(output_dir) / f"{prefix}_{model_name}_{prompt_name}_details.json"


def csv_file(output_dir: str | Path, prefix: str, model_name: str, prompt_name: str) -> Path:
    return Path(output_dir) / f"{prefix}_{model_name}_{prompt_name}.csv"


def detail_is_complete_for_records(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> bool:
    expected_ids = [str(item.get("id", "")) for item in records]
    if len(details) != len(expected_ids):
        return False
    counts = Counter(str(item.get("id", "")) for item in details)
    if not all(counts[item_id] == 1 for item_id in expected_ids):
        return False
    return all(
        str(item.get("model", "")) == model_name and str(item.get("prompt", "")) == prompt_name
        for item in details
    )


def filter_complete_details_for_records(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> list[dict[str, Any]] | None:
    expected_ids = {str(item.get("id", "")) for item in records}
    matching = [item for item in details if str(item.get("id", "")) in expected_ids]
    if detail_is_complete_for_records(matching, records, model_name, prompt_name):
        return matching
    return None


def load_complete_details(
    output_dir: str | Path,
    prefix: str,
    records: list[dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> list[dict[str, Any]] | None:
    path = detail_file(output_dir, prefix, model_name, prompt_name)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        details = json.load(f)
    matching = filter_complete_details_for_records(details, records, model_name, prompt_name)
    if matching is None:
        logger.warning("Cached details are incomplete or stale: %s", path)
    return matching


def _detail_matches_combo(item: dict[str, Any], model_name: str, prompt_name: str) -> bool:
    return str(item.get("model", "")) == model_name and str(item.get("prompt", "")) == prompt_name


def reusable_details_for_records(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = [str(item.get("id", "")) for item in records]
    expected_id_set = set(expected_ids)
    reusable_by_id: dict[str, dict[str, Any]] = {}
    for item in details:
        item_id = str(item.get("id", ""))
        if item_id not in expected_id_set:
            continue
        if not _detail_matches_combo(item, model_name, prompt_name):
            continue
        reusable_by_id.setdefault(item_id, item)

    reusable = [reusable_by_id[item_id] for item_id in expected_ids if item_id in reusable_by_id]
    missing_records = [item for item in records if str(item.get("id", "")) not in reusable_by_id]
    return reusable, missing_records


def merge_detail_records(
    existing: list[dict[str, Any]],
    generated: list[dict[str, Any]],
    model_name: str,
    prompt_name: str,
) -> list[dict[str, Any]]:
    other_records = []
    ordered_ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for item in existing:
        item_id = str(item.get("id", ""))
        if not item_id or not _detail_matches_combo(item, model_name, prompt_name):
            other_records.append(item)
            continue
        if item_id not in by_id:
            ordered_ids.append(item_id)
        by_id[item_id] = item

    for item in generated:
        item_id = str(item.get("id", ""))
        if not item_id:
            continue
        if item_id not in by_id:
            ordered_ids.append(item_id)
        by_id[item_id] = item

    return other_records + [by_id[item_id] for item_id in ordered_ids]


def order_details_for_records(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(item.get("id", "")): item for item in details}
    return [by_id[str(item.get("id", ""))] for item in records if str(item.get("id", "")) in by_id]


def details_to_rows(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> list[list[str]]:
    by_id = {str(item["id"]): item for item in details}
    return [[str(item.get("id", "")), str(by_id.get(str(item.get("id", "")), {}).get("answer", "0"))] for item in records]


def generate_prompt_details(
    records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    model_name: str,
    prompt_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    engine: str = "hf",
    lora_request: Any | None = None,
    checkpoint_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    prepared = [
        {
            "id": str(item.get("id", "")),
            "question": normalize_question_text(item.get("question", "")),
        }
        for item in records
    ]
    details: list[dict[str, Any]] = []
    batches = list(iter_batches(prepared, batch_size))
    for batch in tqdm(batches, desc=f"{model_name}:{prompt_name}"):
        questions = [item["question"] for item in batch]
        if engine == "vllm":
            raw_outputs, prompt_tokens = generate_raw_outputs_vllm_batch(
                llm=model,
                tokenizer=tokenizer,
                questions=questions,
                prompt_name=prompt_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                lora_request=lora_request,
                show_progress=show_progress,
            )
        else:
            raw_outputs, prompt_tokens = generate_raw_outputs_batch(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                prompt_name=prompt_name,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
        batch_details = []
        for item, raw_output, token_count in zip(batch, raw_outputs, prompt_tokens):
            raw_answer, extraction_level = extract_answer(raw_output, return_level=True)
            answer = normalize_answer_safe(raw_answer, item["question"])
            detail = {
                "id": item["id"],
                "question": item["question"],
                "model": model_name,
                "prompt": prompt_name,
                "raw_output": raw_output,
                "raw_answer": raw_answer,
                "answer": answer,
                "extraction_level": extraction_level,
                "has_answer_tag": "<answer>" in raw_output.lower() and "</answer>" in raw_output.lower(),
                "empty_extraction": not bool(str(raw_answer).strip()),
                "prompt_tokens": token_count,
                "raw_length_chars": len(raw_output),
            }
            details.append(detail)
            batch_details.append(detail)
        if checkpoint_callback is not None and batch_details:
            checkpoint_callback(details)
    return details


def prepare_dataset_details(
    records: list[dict[str, Any]],
    output_dir: str | Path,
    prefix: str,
    model_names: list[str],
    prompt_names: list[str],
    base_model_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    reuse_existing: bool,
    generate_missing: bool,
    engine: str,
    vllm_gpu_memory_utilization: float,
    vllm_max_lora_rank: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    details_by_combo: dict[tuple[str, str], list[dict[str, Any]]] = {}
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    missing_by_model: dict[str, list[str]] = {}

    for model_name in model_names:
        for prompt_name in prompt_names:
            path = detail_file(output_dir, prefix, model_name, prompt_name)
            existing_details: list[dict[str, Any]] = []
            reusable: list[dict[str, Any]] = []
            missing_records = list(records)
            if reuse_existing and path.exists():
                with path.open("r", encoding="utf-8") as f:
                    existing_details = json.load(f)
                reusable, missing_records = reusable_details_for_records(
                    existing_details,
                    records,
                    model_name,
                    prompt_name,
                )
                if reusable:
                    logger.info(
                        "Reusing %s/%s/%s cached rows: %s/%s",
                        prefix,
                        model_name,
                        prompt_name,
                        len(reusable),
                        len(records),
                    )

            if not missing_records:
                details = order_details_for_records(records, reusable)
                details_by_combo[(model_name, prompt_name)] = details
                write_csv(csv_file(output_dir, prefix, model_name, prompt_name), details_to_rows(records, details))
                continue

            pending[(model_name, prompt_name)] = {
                "existing_details": existing_details,
                "reusable": reusable,
                "missing_records": missing_records,
            }
            missing_by_model.setdefault(model_name, []).append(prompt_name)

    missing = [(model, prompt) for model, prompts in missing_by_model.items() for prompt in prompts]
    if missing and not generate_missing:
        raise FileNotFoundError(f"Missing complete cached details: {missing}")

    if not missing:
        return details_by_combo

    if engine == "vllm":
        llm, tokenizer = load_vllm_model_and_tokenizer(
            base_model_name=base_model_name,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            max_lora_rank=vllm_max_lora_rank,
        )
        try:
            for lora_int_id, model_name in enumerate(model_names, start=1):
                prompt_names_to_run = missing_by_model.get(model_name, [])
                if not prompt_names_to_run:
                    continue
                spec = MODEL_SPECS[model_name]
                adapter_path = ensure_checkpoint(spec)
                lora_request = build_lora_request(model_name, adapter_path, lora_int_id)
                for prompt_name in prompt_names_to_run:
                    plan = pending[(model_name, prompt_name)]
                    missing_records = plan["missing_records"]
                    logger.info(
                        "Generating %s/%s/%s with vLLM for missing rows: %s/%s",
                        prefix,
                        model_name,
                        prompt_name,
                        len(missing_records),
                        len(records),
                    )
                    checkpoint_path = detail_file(output_dir, prefix, model_name, prompt_name)

                    def checkpoint(
                        generated_so_far: list[dict[str, Any]],
                        plan=plan,
                        model_name=model_name,
                        prompt_name=prompt_name,
                    ) -> None:
                        write_json(
                            checkpoint_path,
                            merge_detail_records(plan["existing_details"], generated_so_far, model_name, prompt_name),
                        )

                    generated = generate_prompt_details(
                        records=missing_records,
                        model=llm,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        prompt_name=prompt_name,
                        batch_size=batch_size,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        engine="vllm",
                        lora_request=lora_request,
                        checkpoint_callback=checkpoint,
                    )
                    merged_file_details = merge_detail_records(
                        plan["existing_details"],
                        generated,
                        model_name,
                        prompt_name,
                    )
                    details = order_details_for_records(records, merged_file_details)
                    if len(details) != len(records):
                        raise RuntimeError(f"Resume merge failed for {(model_name, prompt_name)}")
                    details_by_combo[(model_name, prompt_name)] = details
                    write_json(detail_file(output_dir, prefix, model_name, prompt_name), merged_file_details)
                    write_csv(csv_file(output_dir, prefix, model_name, prompt_name), details_to_rows(records, details))
        finally:
            cleanup_model(llm)
        return details_by_combo

    from src.models.model_loader import load_peft_model

    for model_name, prompt_names_to_run in missing_by_model.items():
        spec = MODEL_SPECS[model_name]
        adapter_path = ensure_checkpoint(spec)
        logger.info("Loading %s from %s", model_name, adapter_path)
        model, tokenizer = load_peft_model(
            base_model_name=base_model_name,
            adapter_path=adapter_path,
            torch_dtype="bfloat16",
        )
        model.eval()
        try:
            for prompt_name in prompt_names_to_run:
                plan = pending[(model_name, prompt_name)]
                missing_records = plan["missing_records"]
                logger.info(
                    "Generating %s/%s/%s with HF for missing rows: %s/%s",
                    prefix,
                    model_name,
                    prompt_name,
                    len(missing_records),
                    len(records),
                )
                checkpoint_path = detail_file(output_dir, prefix, model_name, prompt_name)

                def checkpoint(
                    generated_so_far: list[dict[str, Any]],
                    plan=plan,
                    model_name=model_name,
                    prompt_name=prompt_name,
                ) -> None:
                    write_json(
                        checkpoint_path,
                        merge_detail_records(plan["existing_details"], generated_so_far, model_name, prompt_name),
                    )

                generated = generate_prompt_details(
                    records=missing_records,
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    prompt_name=prompt_name,
                    batch_size=batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    engine="hf",
                    checkpoint_callback=checkpoint,
                )
                merged_file_details = merge_detail_records(
                    plan["existing_details"],
                    generated,
                    model_name,
                    prompt_name,
                )
                details = order_details_for_records(records, merged_file_details)
                if len(details) != len(records):
                    raise RuntimeError(f"Resume merge failed for {(model_name, prompt_name)}")
                details_by_combo[(model_name, prompt_name)] = details
                write_json(detail_file(output_dir, prefix, model_name, prompt_name), merged_file_details)
                write_csv(csv_file(output_dir, prefix, model_name, prompt_name), details_to_rows(records, details))
        finally:
            cleanup_model(model)

    return details_by_combo


def answer_matches_prediction(prediction: Any, gold: Any, question: Any) -> bool:
    try:
        return answers_match_by_question(prediction, gold, question)
    except Exception as exc:
        logger.warning("Answer comparison failed; mark incorrect: %s", exc)
        return False


def score_details(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, Any]:
    records_by_id = {str(item["id"]): item for item in records}
    correct = 0
    extraction_counts: Counter[str] = Counter()
    answer_format_counts: Counter[str] = Counter()
    malformed = 0
    empty = 0
    prompt_tokens = 0
    raw_chars = 0
    scored_details = []

    for item in details:
        item_id = str(item["id"])
        record = records_by_id[item_id]
        question = record.get("question", "")
        gold = record.get("answer", "")
        is_correct = answer_matches_prediction(item.get("answer", ""), gold, question)
        correct += 1 if is_correct else 0
        extraction_counts[str(item.get("extraction_level", "unknown"))] += 1
        answer_format_counts[detect_answer_format(question).answer_type] += 1
        malformed += 0 if item.get("has_answer_tag") else 1
        empty += 1 if item.get("empty_extraction") else 0
        prompt_tokens += int(item.get("prompt_tokens", 0))
        raw_chars += int(item.get("raw_length_chars", 0))
        scored_details.append({
            "id": item_id,
            "prediction": str(item.get("answer", "")),
            "reference": str(gold),
            "correct": is_correct,
        })

    total = len(details)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "malformed_output_count": malformed,
        "empty_extraction_count": empty,
        "avg_prompt_tokens": prompt_tokens / total if total else 0.0,
        "avg_raw_length_chars": raw_chars / total if total else 0.0,
        "extraction_level_counts": dict(extraction_counts),
        "answer_format_counts": dict(answer_format_counts),
        "details": scored_details,
    }


def prediction_correct_map(records: list[dict[str, Any]], details: list[dict[str, Any]]) -> dict[str, bool]:
    records_by_id = {str(item["id"]): item for item in records}
    result: dict[str, bool] = {}
    for item in details:
        item_id = str(item["id"])
        record = records_by_id[item_id]
        result[item_id] = answer_matches_prediction(
            item.get("answer", ""),
            record.get("answer", ""),
            record.get("question", ""),
        )
    return result


def answer_map(details: list[dict[str, Any]]) -> dict[str, str]:
    return {str(item["id"]): str(item.get("answer", "")) for item in details}


def exact_sign_test_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2 ** trials)
    return min(1.0, 2 * probability)


def paired_summary(
    records: list[dict[str, Any]],
    a_details: list[dict[str, Any]],
    b_details: list[dict[str, Any]],
    a_label: str,
    b_label: str,
) -> dict[str, Any]:
    a_correct = prediction_correct_map(records, a_details)
    b_correct = prediction_correct_map(records, b_details)
    both_correct = 0
    both_wrong = 0
    a_only_correct = 0
    b_only_correct = 0
    for item in records:
        item_id = str(item["id"])
        a_ok = a_correct[item_id]
        b_ok = b_correct[item_id]
        if a_ok and b_ok:
            both_correct += 1
        elif not a_ok and not b_ok:
            both_wrong += 1
        elif a_ok:
            a_only_correct += 1
        else:
            b_only_correct += 1
    total = len(records)
    return {
        "a": a_label,
        "b": b_label,
        "total": total,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "a_only_correct": a_only_correct,
        "b_only_correct": b_only_correct,
        "net_b_wins": b_only_correct - a_only_correct,
        "a_accuracy": (both_correct + a_only_correct) / total if total else 0.0,
        "b_accuracy": (both_correct + b_only_correct) / total if total else 0.0,
        "accuracy_delta_b_minus_a": (
            (both_correct + b_only_correct) / total - (both_correct + a_only_correct) / total
            if total
            else 0.0
        ),
        "exact_sign_test_p_value": exact_sign_test_p_value(b_only_correct, a_only_correct),
    }


def record_id_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def build_prompt_disagreements(
    records: list[dict[str, Any]],
    details_by_prompt: dict[str, list[dict[str, Any]]],
    model_name: str,
    has_gold: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records_by_id = {str(item["id"]): item for item in records}
    answers_by_prompt = {prompt: answer_map(details) for prompt, details in details_by_prompt.items()}
    correct_by_prompt = (
        {prompt: prediction_correct_map(records, details) for prompt, details in details_by_prompt.items()}
        if has_gold
        else {}
    )

    for item_id in sorted(records_by_id, key=record_id_sort_key):
        record = records_by_id[item_id]
        answers = {prompt: answers_by_prompt[prompt][item_id] for prompt in details_by_prompt}
        answer_values = set(answers.values())
        correctness_values = (
            {correct_by_prompt[prompt][item_id] for prompt in details_by_prompt}
            if has_gold
            else set()
        )
        if len(answer_values) == 1 and (not has_gold or len(correctness_values) == 1):
            continue
        row: dict[str, Any] = {
            "id": item_id,
            "model": model_name,
            "question": normalize_question_text(record.get("question", "")),
        }
        if has_gold:
            row["gold_answer"] = str(record.get("answer", ""))
        for prompt in PROMPT_NAMES:
            if prompt in answers:
                row[f"{prompt}_answer"] = answers[prompt]
                if has_gold:
                    row[f"{prompt}_correct"] = correct_by_prompt[prompt][item_id]
        rows.append(row)
    return rows


def summarize_test_disagreements(
    records: list[dict[str, Any]],
    details_by_prompt: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    answers_by_prompt = {prompt: answer_map(details) for prompt, details in details_by_prompt.items()}
    all_same = 0
    any_diff = 0
    for item in records:
        item_id = str(item["id"])
        values = {answers_by_prompt[prompt][item_id] for prompt in details_by_prompt}
        if len(values) == 1:
            all_same += 1
        else:
            any_diff += 1
    return {"all_prompts_same": all_same, "any_prompt_diff": any_diff, "total": len(records)}


def build_report(
    val_records: list[dict[str, Any]],
    val_details: dict[tuple[str, str], list[dict[str, Any]]],
    test_records: list[dict[str, Any]],
    test_details: dict[tuple[str, str], list[dict[str, Any]]] | None,
    model_names: list[str],
    prompt_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    val_metrics: dict[str, dict[str, Any]] = {}
    paired: list[dict[str, Any]] = []

    for model_name in model_names:
        val_metrics[model_name] = {}
        for prompt_name in prompt_names:
            metric = score_details(val_records, val_details[(model_name, prompt_name)])
            val_metrics[model_name][prompt_name] = {key: value for key, value in metric.items() if key != "details"}

        if "few_shot_cot" in prompt_names:
            for baseline in ["direct", "zero_shot_cot"]:
                if baseline in prompt_names:
                    paired.append(paired_summary(
                        records=val_records,
                        a_details=val_details[(model_name, baseline)],
                        b_details=val_details[(model_name, "few_shot_cot")],
                        a_label=f"{model_name}:{baseline}",
                        b_label=f"{model_name}:few_shot_cot",
                    ))
        if "direct" in prompt_names and "zero_shot_cot" in prompt_names:
            paired.append(paired_summary(
                records=val_records,
                a_details=val_details[(model_name, "direct")],
                b_details=val_details[(model_name, "zero_shot_cot")],
                a_label=f"{model_name}:direct",
                b_label=f"{model_name}:zero_shot_cot",
            ))

    cross_model: list[dict[str, Any]] = []
    for prompt_name in prompt_names:
        for a_model, b_model in [("sft_cot", "dpo"), ("sft_cot", "grpo"), ("dpo", "grpo")]:
            if a_model in model_names and b_model in model_names:
                cross_model.append(paired_summary(
                    records=val_records,
                    a_details=val_details[(a_model, prompt_name)],
                    b_details=val_details[(b_model, prompt_name)],
                    a_label=f"{a_model}:{prompt_name}",
                    b_label=f"{b_model}:{prompt_name}",
                ))

    test_summary: dict[str, Any] = {}
    if test_details is not None:
        for model_name in model_names:
            per_prompt = {
                prompt_name: test_details[(model_name, prompt_name)]
                for prompt_name in prompt_names
            }
            test_summary[model_name] = summarize_test_disagreements(test_records, per_prompt)

    return {
        "scope": "inference_stage_prompt_ablation",
        "controlled_variable": "prompt_shape",
        "models": {model_name: MODEL_SPECS[model_name] for model_name in model_names},
        "prompts": prompt_names,
        "inputs": {
            "val": args.val,
            "test": args.test,
            "limit": args.limit,
            "skip_test": args.skip_test,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "do_sample": args.do_sample,
            "engine": args.engine,
            "reuse_existing": not args.no_reuse_existing,
            "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            "vllm_max_lora_rank": args.vllm_max_lora_rank,
        },
        "validation": {
            "rows": len(val_records),
            "metrics": val_metrics,
            "within_model_paired_comparisons": paired,
            "cross_model_paired_comparisons": cross_model,
        },
        "test": {
            "rows": len(test_records),
            "has_gold": False,
            "prompt_disagreement_summary": test_summary,
        },
    }


def write_summary_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Few-shot CoT Prompt Ablation",
        "",
        "This experiment keeps the adapter checkpoint fixed and changes only the inference prompt.",
        "Validation accuracy is the primary evidence. Test output has no gold labels, so test artifacts are for submission and disagreement inspection only.",
        "",
        "## Validation Accuracy",
        "",
        "| Model | Prompt | Accuracy | Correct / Total | Missing answer tag | Empty extraction | Avg prompt tokens |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    metrics = report["validation"]["metrics"]
    for model_name, prompt_metrics in metrics.items():
        for prompt_name, metric in prompt_metrics.items():
            lines.append(
                f"| {model_name} | {prompt_name} | {metric['accuracy']:.4f} | "
                f"{metric['correct']} / {metric['total']} | "
                f"{metric['malformed_output_count']} | {metric['empty_extraction_count']} | "
                f"{metric['avg_prompt_tokens']:.1f} |"
            )

    lines.extend([
        "",
        "## Few-shot Effect",
        "",
        "| Comparison | Accuracy delta | Net wins | Sign-test p |",
        "|---|---:|---:|---:|",
    ])
    for item in report["validation"]["within_model_paired_comparisons"]:
        if not item["b"].endswith(":few_shot_cot"):
            continue
        lines.append(
            f"| {item['b']} vs {item['a']} | {item['accuracy_delta_b_minus_a']:.4f} | "
            f"{item['net_b_wins']} | {item['exact_sign_test_p_value']:.6g} |"
        )

    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `val_{model}_{prompt}.csv` and `val_{model}_{prompt}_details.json`",
        "- `test_{model}_{prompt}.csv` and `test_{model}_{prompt}_details.json`",
        "- `val_disagreements.csv` and `test_disagreements.csv`",
        "- `ablation_report.json`",
    ])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    if args.no_reuse_existing and args.only_reuse_existing:
        raise ValueError("--no_reuse_existing and --only_reuse_existing cannot be used together")
    from src.utils.seed import set_seed

    set_seed(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = parse_names(args.models, MODEL_SPECS.keys(), "models")
    prompt_names = parse_names(args.prompts, PROMPT_NAMES, "prompts")
    if args.engine not in ENGINE_NAMES:
        raise ValueError(f"Unknown engine: {args.engine}; available={ENGINE_NAMES}")
    val_records = load_records(args.val, args.limit)
    test_records = [] if args.skip_test else load_records(args.test, args.limit)

    base_model_name = args.base_model or load_base_model_name()
    reuse_existing = not args.no_reuse_existing
    generate_missing = not args.only_reuse_existing

    val_details = prepare_dataset_details(
        records=val_records,
        output_dir=output_dir,
        prefix="val",
        model_names=model_names,
        prompt_names=prompt_names,
        base_model_name=base_model_name,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        reuse_existing=reuse_existing,
        generate_missing=generate_missing,
        engine=args.engine,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_lora_rank=args.vllm_max_lora_rank,
    )

    test_details = None
    if not args.skip_test:
        test_details = prepare_dataset_details(
            records=test_records,
            output_dir=output_dir,
            prefix="test",
            model_names=model_names,
            prompt_names=prompt_names,
            base_model_name=base_model_name,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            reuse_existing=reuse_existing,
            generate_missing=generate_missing,
            engine=args.engine,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_lora_rank=args.vllm_max_lora_rank,
        )

    val_disagreements: list[dict[str, Any]] = []
    test_disagreements: list[dict[str, Any]] = []
    for model_name in model_names:
        val_disagreements.extend(build_prompt_disagreements(
            records=val_records,
            details_by_prompt={prompt: val_details[(model_name, prompt)] for prompt in prompt_names},
            model_name=model_name,
            has_gold=True,
        ))
        if test_details is not None:
            test_disagreements.extend(build_prompt_disagreements(
                records=test_records,
                details_by_prompt={prompt: test_details[(model_name, prompt)] for prompt in prompt_names},
                model_name=model_name,
                has_gold=False,
            ))

    disagreement_fields = (
        ["id", "model", "question", "gold_answer"]
        + [f"{prompt}_answer" for prompt in PROMPT_NAMES]
        + [f"{prompt}_correct" for prompt in PROMPT_NAMES]
    )
    test_disagreement_fields = ["id", "model", "question"] + [f"{prompt}_answer" for prompt in PROMPT_NAMES]
    write_dict_csv(output_dir / "val_disagreements.csv", val_disagreements, disagreement_fields)
    write_dict_csv(output_dir / "test_disagreements.csv", test_disagreements, test_disagreement_fields)

    report = build_report(
        val_records=val_records,
        val_details=val_details,
        test_records=test_records,
        test_details=test_details,
        model_names=model_names,
        prompt_names=prompt_names,
        args=args,
    )
    report["validation"]["disagreement_count"] = len(val_disagreements)
    report["test"]["disagreement_count"] = len(test_disagreements)
    write_json(output_dir / "ablation_report.json", report)
    write_summary_markdown(output_dir / "ablation_summary.md", report)
    logger.info("CoT prompt ablation complete: %s", output_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Few-shot CoT prompt ablation for SFT/DPO/GRPO adapters")
    parser.add_argument("--val", default="data/splits/train_expr_clean_val.json")
    parser.add_argument("--test", default="data/raw/test.json")
    parser.add_argument("--output_dir", default="outputs/evaluation/cot_prompt_ablation")
    parser.add_argument("--models", default="sft_cot,dpo,grpo")
    parser.add_argument("--prompts", default="direct,zero_shot_cot,few_shot_cot")
    parser.add_argument("--base_model", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--engine", choices=ENGINE_NAMES, default="vllm")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_max_lora_rank", type=int, default=16)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--no_reuse_existing", action="store_true")
    parser.add_argument("--only_reuse_existing", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
