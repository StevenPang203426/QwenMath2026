"""
Validate and submit with four expression models plus CoT soft calibration.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from tqdm import tqdm

from src.data.answer_extractor import extract_answer
from src.inference.answer_postprocessor import postprocess_answer
from src.inference.expr_predictor import expr_predict_single
from src.models.model_loader import load_peft_model
from src.utils.answer_normalizer import (
    answers_match,
    format_auto,
    is_finite_real_number,
    normalize_answer_for_question,
    normalize_question_text,
    safe_eval_expression,
)
from src.utils.config import load_config
from src.utils.expression_policy import validate_expression
from src.utils.metrics import normalize_number
from src.utils.seed import set_seed

logger = logging.getLogger("math_solver.expr4_eval_submit")

EXPR_INSTRUCTION = (
    "请为以下数学题写出一个Python可直接计算的中缀数学表达式。"
    "用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。"
)

COT_INSTRUCTION = (
    "请一步一步思考，然后给出数字答案。"
    "用<think></think>标签包裹推理过程，用<answer></answer>标签包裹最终数字答案。"
)

TEMPERATURES = [0.1, 0.3, 0.7]
EXPR_MAX_NEW_TOKENS = 96
COT_MAX_NEW_TOKENS = 512

EXPR_MODELS = [
    {"name": "sft_expr_clean", "adapter_path": "outputs/checkpoints/sft_expr_clean/best"},
    {"name": "dpo_expr_clean", "adapter_path": "outputs/checkpoints/dpo_expr_clean/best"},
    {"name": "grpo_expr_clean_vllm", "adapter_path": "outputs/checkpoints/grpo_expr_clean_vllm/best"},
    {
        "name": "grpo_expr_from_dpo_clean_vllm",
        "adapter_path": "outputs/checkpoints/grpo_expr_from_dpo_clean_vllm/best",
    },
]

COT_MODELS = [
    {"name": "cot_sft", "adapter_path": "outputs/checkpoints/sft_cot/best"},
    {"name": "cot_dpo", "adapter_path": "outputs/checkpoints/dpo/best"},
    {"name": "cot_grpo", "adapter_path": "outputs/checkpoints/grpo/best"},
]

EXPR_WEIGHT_PRESETS = {
    "equal": {
        "sft_expr_clean": 1.0,
        "dpo_expr_clean": 1.0,
        "grpo_expr_clean_vllm": 1.0,
        "grpo_expr_from_dpo_clean_vllm": 1.0,
    },
    "rl_boost": {
        "sft_expr_clean": 1.0,
        "dpo_expr_clean": 1.0,
        "grpo_expr_clean_vllm": 1.5,
        "grpo_expr_from_dpo_clean_vllm": 1.5,
    },
    "dpo_aware": {
        "sft_expr_clean": 1.0,
        "dpo_expr_clean": 1.2,
        "grpo_expr_clean_vllm": 1.4,
        "grpo_expr_from_dpo_clean_vllm": 1.6,
    },
}
COT_WEIGHTS = {"cot_sft": 1.0, "cot_dpo": 1.2, "cot_grpo": 1.2}
COT_BONUS_SCALES = [0.0, 0.15, 0.3, 0.5]


@dataclass(frozen=True)
class VotingStrategy:
    preset: str
    expr_weights: dict[str, float]
    cot_weights: dict[str, float]
    cot_bonus_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "expr_weights": self.expr_weights,
            "cot_weights": self.cot_weights,
            "cot_bonus_scale": self.cot_bonus_scale,
        }


def load_records(path: str, limit: int = 0) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if limit and limit > 0:
        return records[:limit]
    return records


def answer_close_by_question(a: Any, b: Any, question: Any, tol: float = 1e-4) -> bool:
    norm_a = normalize_answer_for_question_safe(a, question)
    norm_b = normalize_answer_for_question_safe(b, question)
    if str(norm_a).strip() == str(norm_b).strip():
        return True
    return answers_match(norm_a, norm_b, tol=tol)


def answer_vote_key(answer: Any, question: Any) -> str:
    normalized = normalize_answer_for_question_safe(answer, question)
    numeric = normalize_number(normalized)
    return numeric if numeric is not None else normalized.strip()


def answer_from_expression(expression: Any, question: str) -> str | None:
    policy = validate_expression(expression)
    if not policy.ok:
        return None
    try:
        value = safe_eval_expression(policy.normalized)
    except Exception:
        return None
    if not is_finite_real_number(value):
        return None
    try:
        return postprocess_answer(format_auto(value), question)
    except Exception as exc:
        logger.warning("表达式答案格式化失败，已拒绝候选: %s", exc)
        return None


def normalize_answer_for_question_safe(raw_answer: Any, question: Any) -> str:
    try:
        return normalize_answer_for_question(str(raw_answer), question)
    except Exception as exc:
        logger.warning("答案归一化失败，已降级为 0: %s", exc)
        return "0"


def expr_predict_single_safe(raw_output: str, question: str) -> dict[str, Any]:
    try:
        return expr_predict_single(raw_output, question)
    except Exception as exc:
        logger.warning("表达式候选解析失败，已降级为失败候选: %s", exc)
        return {
            "answer": "0",
            "expression": "",
            "eval_success": False,
            "source": "expr_predict_error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _model_from_candidate(candidate: dict[str, Any]) -> str:
    if candidate.get("model"):
        return str(candidate["model"])
    source = str(candidate.get("source", ""))
    for model in [item["name"] for item in EXPR_MODELS + COT_MODELS]:
        if source == model or source.startswith(f"{model}_"):
            return model
    return source


def _prepare_expr_candidates(
    question: str,
    expr_candidates: list[dict[str, Any]],
    expr_weights: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    safe_expr = []
    rejected_expr = []
    for candidate in expr_candidates:
        answer = answer_from_expression(candidate.get("expression", ""), question)
        model = _model_from_candidate(candidate)
        if answer is None:
            rejected_expr.append({**candidate, "model": model})
            continue
        safe_expr.append({
            **candidate,
            "model": model,
            "answer": answer,
            "vote_key": answer_vote_key(answer, question),
            "weight": float(expr_weights.get(model, candidate.get("weight", 1.0))),
        })
    return safe_expr, rejected_expr


def _prepare_cot_candidates(
    question: str,
    cot_candidates: list[dict[str, Any]],
    cot_weights: dict[str, float],
) -> list[dict[str, Any]]:
    prepared = []
    for candidate in cot_candidates:
        model = _model_from_candidate(candidate)
        answer = normalize_answer_for_question_safe(candidate.get("answer", ""), question)
        prepared.append({
            **candidate,
            "model": model,
            "answer": answer,
            "vote_key": answer_vote_key(answer, question),
            "weight": float(cot_weights.get(model, candidate.get("weight", 1.0))),
        })
    return prepared


def _group_expr_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate["vote_key"])
        if key not in groups:
            groups[key] = {
                "key": key,
                "answer": candidate["answer"],
                "expr_count": 0,
                "expr_weight": 0.0,
                "expr_sources": [],
                "expr_models": [],
                "cot_support_count": 0,
                "cot_support_weight": 0.0,
                "cot_support_sources": [],
                "cot_support_models": [],
            }
        groups[key]["expr_count"] += 1
        groups[key]["expr_weight"] += float(candidate.get("weight", 1.0))
        groups[key]["expr_sources"].append(candidate.get("source", candidate.get("model", "unknown")))
        groups[key]["expr_models"].append(candidate.get("model", "unknown"))
    return groups


def _add_cot_support(
    question: str,
    groups: dict[str, dict[str, Any]],
    cot_candidates: list[dict[str, Any]],
) -> None:
    for group in groups.values():
        for candidate in cot_candidates:
            if answer_close_by_question(group["answer"], candidate["answer"], question):
                group["cot_support_count"] += 1
                group["cot_support_weight"] += float(candidate.get("weight", 1.0))
                group["cot_support_sources"].append(candidate.get("source", candidate.get("model", "unknown")))
                group["cot_support_models"].append(candidate.get("model", "unknown"))


def _group_cot_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate["vote_key"])
        if key not in groups:
            groups[key] = {
                "key": key,
                "answer": candidate["answer"],
                "count": 0,
                "weight": 0.0,
                "sources": [],
            }
        groups[key]["count"] += 1
        groups[key]["weight"] += float(candidate.get("weight", 1.0))
        groups[key]["sources"].append(candidate.get("source", candidate.get("model", "unknown")))
    return groups


def vote_expr4_soft_calibrated(
    question: str,
    expr_candidates: list[dict[str, Any]],
    cot_candidates: list[dict[str, Any]] | None,
    strategy: VotingStrategy,
) -> dict[str, Any]:
    cot_candidates = cot_candidates or []
    safe_expr, rejected_expr = _prepare_expr_candidates(question, expr_candidates, strategy.expr_weights)
    prepared_cot = _prepare_cot_candidates(question, cot_candidates, strategy.cot_weights)

    if safe_expr:
        groups = _group_expr_candidates(safe_expr)
        _add_cot_support(question, groups, prepared_cot)
        ordered = sorted(
            groups.values(),
            key=lambda group: (
                group["expr_count"],
                group["expr_weight"] + strategy.cot_bonus_scale * group["cot_support_weight"],
                -len(str(group["answer"])),
            ),
            reverse=True,
        )
        winner = ordered[0]
        score = winner["expr_weight"] + strategy.cot_bonus_scale * winner["cot_support_weight"]
        source = (
            "expression_cot_calibrated"
            if winner["cot_support_count"] and strategy.cot_bonus_scale > 0
            else "expression_majority"
        )
        return {
            "answer": winner["answer"],
            "source": source,
            "details": {**winner, "score": score},
            "safe_expression_candidates": len(safe_expr),
            "rejected_expression_candidates": len(rejected_expr),
            "cot_candidates": len(prepared_cot),
            "fallback": False,
        }

    if prepared_cot:
        groups = _group_cot_candidates(prepared_cot)
        winner = sorted(groups.values(), key=lambda group: (group["count"], group["weight"]), reverse=True)[0]
        return {
            "answer": winner["answer"],
            "source": "cot_fallback",
            "details": winner,
            "safe_expression_candidates": 0,
            "rejected_expression_candidates": len(rejected_expr),
            "cot_candidates": len(prepared_cot),
            "fallback": True,
        }

    return {
        "answer": "0",
        "source": "no_candidate",
        "details": {},
        "safe_expression_candidates": 0,
        "rejected_expression_candidates": len(rejected_expr),
        "cot_candidates": 0,
        "fallback": True,
    }


def strategy_grid() -> list[VotingStrategy]:
    return [
        VotingStrategy(
            preset=preset,
            expr_weights=dict(expr_weights),
            cot_weights=dict(COT_WEIGHTS),
            cot_bonus_scale=bonus,
        )
        for preset, expr_weights in EXPR_WEIGHT_PRESETS.items()
        for bonus in COT_BONUS_SCALES
    ]


def _records_by_id(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in records}


def _sort_ids(ids: list[str]) -> list[str]:
    return sorted(ids, key=lambda item: int(item) if str(item).isdigit() else str(item))


def vote_dataset(
    records: list[dict[str, Any]],
    expr_by_id: dict[str, list[dict[str, Any]]],
    cot_by_id: dict[str, list[dict[str, Any]]],
    strategy: VotingStrategy,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    rows: list[list[str]] = []
    report: list[dict[str, Any]] = []
    for item in records:
        item_id = str(item["id"])
        question = normalize_question_text(item.get("question", ""))
        result = vote_expr4_soft_calibrated(
            question=question,
            expr_candidates=expr_by_id.get(item_id, []),
            cot_candidates=cot_by_id.get(item_id, []),
            strategy=strategy,
        )
        rows.append([item_id, result["answer"]])
        report.append({
            "id": item_id,
            "question": question,
            "answer": result["answer"],
            "source": result["source"],
            "safe_expression_candidates": result["safe_expression_candidates"],
            "rejected_expression_candidates": result["rejected_expression_candidates"],
            "cot_candidates": result["cot_candidates"],
            "fallback": result["fallback"],
            "details": result["details"],
        })
    return rows, report


def score_report(
    records: list[dict[str, Any]],
    per_item_report: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_by_id = _records_by_id(records)
    correct = 0
    fallback = 0
    source_counts: dict[str, int] = {}
    details = []
    for item in per_item_report:
        item_id = str(item["id"])
        gold_item = gold_by_id[item_id]
        is_correct = answer_close_by_question(
            item["answer"],
            gold_item.get("answer", ""),
            gold_item.get("question", ""),
        )
        correct += 1 if is_correct else 0
        fallback += 1 if item.get("fallback") else 0
        source = str(item.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        details.append({
            "id": item_id,
            "prediction": item["answer"],
            "reference": str(gold_item.get("answer", "")),
            "correct": is_correct,
            "source": source,
        })
    total = len(per_item_report)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "fallback": fallback,
        "source_counts": source_counts,
        "details": details,
    }


def evaluate_strategies(
    records: list[dict[str, Any]],
    expr_by_id: dict[str, list[dict[str, Any]]],
    cot_by_id: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics = []
    for strategy in strategy_grid():
        _, report = vote_dataset(records, expr_by_id, cot_by_id, strategy)
        score = score_report(records, report)
        metrics.append({
            "strategy": strategy.to_dict(),
            "accuracy": score["accuracy"],
            "correct": score["correct"],
            "total": score["total"],
            "fallback": score["fallback"],
            "source_counts": score["source_counts"],
        })
    selected = sorted(
        metrics,
        key=lambda item: (
            item["accuracy"],
            -item["fallback"],
            -float(item["strategy"]["cot_bonus_scale"]),
        ),
        reverse=True,
    )[0]
    return selected, metrics


def single_model_metrics(
    records: list[dict[str, Any]],
    expr_details_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    gold_by_id = _records_by_id(records)
    result = {}
    for model, details in expr_details_by_model.items():
        first_by_id: dict[str, dict[str, Any]] = {}
        eval_success = 0
        for item in details:
            if item.get("eval_success"):
                eval_success += 1
            item_id = str(item.get("id", ""))
            if item_id and item_id not in first_by_id:
                first_by_id[item_id] = item
        correct = 0
        for item_id, gold in gold_by_id.items():
            pred = first_by_id.get(item_id, {}).get("answer", "0")
            if answer_close_by_question(pred, gold.get("answer", ""), gold.get("question", "")):
                correct += 1
        total = len(gold_by_id)
        result[model] = {
            "accuracy": correct / total if total else 0.0,
            "correct": correct,
            "total": total,
            "eval_success_rate": eval_success / len(details) if details else 0.0,
            "eval_success": eval_success,
            "candidates": len(details),
        }
    return result


def expr_details_to_by_id(
    details_by_model: dict[str, list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for model, details in details_by_model.items():
        for item in details:
            item_id = str(item.get("id", ""))
            result.setdefault(item_id, []).append({
                **item,
                "model": model,
                "weight": float((weights or {}).get(model, 1.0)),
            })
    return result


def cot_details_to_by_id(
    details_by_model: dict[str, list[dict[str, Any]]],
    weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for model, details in details_by_model.items():
        for item in details:
            item_id = str(item.get("id", ""))
            result.setdefault(item_id, []).append({
                **item,
                "model": model,
                "weight": float((weights or {}).get(model, 1.0)),
            })
    return result


def write_csv(path: str, rows: list[list[str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def write_json(path: str, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_checkpoints(models: list[dict[str, str]]) -> None:
    missing = [item["adapter_path"] for item in models if not Path(item["adapter_path"]).is_dir()]
    if missing:
        raise FileNotFoundError("缺少 checkpoint: " + ", ".join(missing))


def cleanup_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def iter_batches(items: list[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch_size = max(1, batch_size)
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_chat_prompts(tokenizer: Any, questions: list[str], instruction: str) -> list[str]:
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return prompts


def generate_raw_outputs_batch(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    instruction: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> list[str]:
    if not questions:
        return []
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts = build_chat_prompts(tokenizer, questions, instruction)
    device = next(model.parameters()).device
    try:
        model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
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
        return [
            tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True)
            for output_ids in generated_ids
        ]
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(questions) == 1:
            raise
        logger.warning("批量生成 OOM，自动拆分 batch: size=%s", len(questions))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mid = len(questions) // 2
        return (
            generate_raw_outputs_batch(
                model, tokenizer, questions[:mid], instruction, max_new_tokens, temperature, do_sample
            )
            + generate_raw_outputs_batch(
                model, tokenizer, questions[mid:], instruction, max_new_tokens, temperature, do_sample
            )
        )


def load_base_model_name() -> str:
    return load_config("configs/base.yaml").model.name


def generate_expression_details(
    records: list[dict[str, Any]],
    output_dir: str,
    prefix: str,
    batch_size: int = 16,
    models: list[dict[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    models = models or EXPR_MODELS
    ensure_checkpoints(models)
    base_model_name = load_base_model_name()
    prepared_records = [
        {
            "id": str(item.get("id", "")),
            "question": normalize_question_text(item.get("question", "")),
        }
        for item in records
    ]
    details_by_model = {}
    for spec in models:
        model_name = spec["name"]
        logger.info("表达式候选生成: %s, batch_size=%s", model_name, batch_size)
        model, tokenizer = load_peft_model(
            base_model_name=base_model_name,
            adapter_path=spec["adapter_path"],
            torch_dtype="bfloat16",
        )
        model.eval()
        details = []
        first_answers: dict[str, str] = {}
        try:
            with tqdm(total=len(prepared_records) * len(TEMPERATURES), desc=model_name) as progress:
                for temperature in TEMPERATURES:
                    do_sample = temperature > 0.0 and temperature != min(TEMPERATURES)
                    for batch in iter_batches(prepared_records, batch_size):
                        raw_outputs = generate_raw_outputs_batch(
                            model=model,
                            tokenizer=tokenizer,
                            questions=[item["question"] for item in batch],
                            instruction=EXPR_INSTRUCTION,
                            max_new_tokens=EXPR_MAX_NEW_TOKENS,
                            temperature=temperature,
                            do_sample=do_sample,
                        )
                        for item, raw_output in zip(batch, raw_outputs):
                            pred = expr_predict_single_safe(raw_output, item["question"])
                            answer = str(pred.get("answer", "0"))
                            details.append({
                                "id": item["id"],
                                "question": item["question"],
                                "model": model_name,
                                "source": f"{model_name}_t{temperature:g}",
                                "temperature": temperature,
                                "raw_output": raw_output,
                                "expression": pred.get("expression", ""),
                                "answer": answer,
                                "eval_success": pred.get("eval_success", False),
                            })
                            if temperature == min(TEMPERATURES):
                                first_answers[item["id"]] = answer
                        progress.update(len(batch))
        finally:
            cleanup_model(model)
        rows = [[item["id"], first_answers.get(item["id"], "0")] for item in prepared_records]
        details_by_model[model_name] = details
        write_json(f"{output_dir}/{prefix}_{model_name}_details.json", details)
        write_csv(f"{output_dir}/{prefix}_{model_name}.csv", rows)
    return details_by_model


def generate_cot_details(
    records: list[dict[str, Any]],
    output_dir: str,
    prefix: str,
    batch_size: int = 8,
    models: list[dict[str, str]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    models = models or COT_MODELS
    ensure_checkpoints(models)
    base_model_name = load_base_model_name()
    prepared_records = [
        {
            "id": str(item.get("id", "")),
            "question": normalize_question_text(item.get("question", "")),
        }
        for item in records
    ]
    details_by_model = {}
    for spec in models:
        model_name = spec["name"]
        logger.info("CoT 辅助生成: %s, batch_size=%s", model_name, batch_size)
        model, tokenizer = load_peft_model(
            base_model_name=base_model_name,
            adapter_path=spec["adapter_path"],
            torch_dtype="bfloat16",
        )
        model.eval()
        details = []
        rows = []
        try:
            for batch in tqdm(list(iter_batches(prepared_records, batch_size)), desc=model_name):
                raw_outputs = generate_raw_outputs_batch(
                    model=model,
                    tokenizer=tokenizer,
                    questions=[item["question"] for item in batch],
                    instruction=COT_INSTRUCTION,
                    max_new_tokens=COT_MAX_NEW_TOKENS,
                    temperature=0.1,
                    do_sample=False,
                )
                for item, raw_output in zip(batch, raw_outputs):
                    raw_answer = extract_answer(raw_output)
                    answer = normalize_answer_for_question_safe(raw_answer, item["question"])
                    details.append({
                        "id": item["id"],
                        "question": item["question"],
                        "model": model_name,
                        "source": model_name,
                        "raw_output": raw_output,
                        "raw_answer": raw_answer,
                        "answer": answer,
                    })
                    rows.append([item["id"], answer])
        finally:
            cleanup_model(model)
        details_by_model[model_name] = details
        write_json(f"{output_dir}/{prefix}_{model_name}_details.json", details)
        write_csv(f"{output_dir}/{prefix}_{model_name}.csv", rows)
    return details_by_model


def load_generated_details(output_dir: str, prefix: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    expr = {}
    cot = {}
    for spec in EXPR_MODELS:
        path = Path(output_dir) / f"{prefix}_{spec['name']}_details.json"
        if path.exists():
            expr[spec["name"]] = load_json(str(path))
    for spec in COT_MODELS:
        path = Path(output_dir) / f"{prefix}_{spec['name']}_details.json"
        if path.exists():
            cot[spec["name"]] = load_json(str(path))
    return expr, cot


def details_complete_for_records(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    expected_per_record: int,
) -> bool:
    expected_ids = [str(item.get("id", "")) for item in records]
    if len(details) != len(expected_ids) * expected_per_record:
        return False
    counts = Counter(str(item.get("id", "")) for item in details)
    return all(counts[item_id] == expected_per_record for item_id in expected_ids)


def load_complete_generated_details(
    output_dir: str,
    prefix: str,
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    expr, cot = load_generated_details(output_dir, prefix)
    complete_expr = {}
    complete_cot = {}

    for spec in EXPR_MODELS:
        model_name = spec["name"]
        details = expr.get(model_name)
        if details and details_complete_for_records(details, records, len(TEMPERATURES)):
            complete_expr[model_name] = details
        elif details:
            logger.warning("已有表达式候选不完整或不匹配，将重跑: %s", model_name)

    for spec in COT_MODELS:
        model_name = spec["name"]
        details = cot.get(model_name)
        if details and details_complete_for_records(details, records, 1):
            complete_cot[model_name] = details
        elif details:
            logger.warning("已有 CoT 候选不完整或不匹配，将重跑: %s", model_name)

    return complete_expr, complete_cot


def prepare_generated_details(
    records: list[dict[str, Any]],
    output_dir: str,
    prefix: str,
    expr_batch_size: int,
    cot_batch_size: int,
    reuse_existing: bool = False,
    resume_existing: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if reuse_existing or resume_existing:
        expr_details_by_model, cot_details_by_model = load_complete_generated_details(output_dir, prefix, records)
    else:
        expr_details_by_model, cot_details_by_model = {}, {}

    if reuse_existing:
        return expr_details_by_model, cot_details_by_model

    missing_expr_specs = [spec for spec in EXPR_MODELS if spec["name"] not in expr_details_by_model]
    missing_cot_specs = [spec for spec in COT_MODELS if spec["name"] not in cot_details_by_model]

    if resume_existing:
        if expr_details_by_model:
            logger.info("断点恢复复用表达式候选: %s", ", ".join(sorted(expr_details_by_model)))
        if cot_details_by_model:
            logger.info("断点恢复复用 CoT 候选: %s", ", ".join(sorted(cot_details_by_model)))

    if missing_expr_specs:
        expr_details_by_model.update(generate_expression_details(
            records,
            output_dir,
            prefix,
            batch_size=expr_batch_size,
            models=missing_expr_specs,
        ))
    if missing_cot_specs:
        cot_details_by_model.update(generate_cot_details(
            records,
            output_dir,
            prefix,
            batch_size=cot_batch_size,
            models=missing_cot_specs,
        ))

    return expr_details_by_model, cot_details_by_model


def validate(args: argparse.Namespace) -> None:
    records = load_records(args.val, args.limit)
    set_seed(42)
    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    expr_details_by_model, cot_details_by_model = prepare_generated_details(
        records=records,
        output_dir=output_dir,
        prefix="val",
        expr_batch_size=args.expr_batch_size,
        cot_batch_size=args.cot_batch_size,
        reuse_existing=args.reuse_existing,
        resume_existing=args.resume_existing,
    )

    missing_expr = {item["name"] for item in EXPR_MODELS} - set(expr_details_by_model)
    missing_cot = {item["name"] for item in COT_MODELS} - set(cot_details_by_model)
    if missing_expr or missing_cot:
        raise FileNotFoundError(f"缺少候选明细: expr={sorted(missing_expr)}, cot={sorted(missing_cot)}")

    selected, metrics = evaluate_strategies(
        records=records,
        expr_by_id=expr_details_to_by_id(expr_details_by_model),
        cot_by_id=cot_details_to_by_id(cot_details_by_model),
    )
    strategy = VotingStrategy(
        preset=selected["strategy"]["preset"],
        expr_weights=selected["strategy"]["expr_weights"],
        cot_weights=selected["strategy"]["cot_weights"],
        cot_bonus_scale=float(selected["strategy"]["cot_bonus_scale"]),
    )
    rows, voted_report = vote_dataset(
        records=records,
        expr_by_id=expr_details_to_by_id(expr_details_by_model, strategy.expr_weights),
        cot_by_id=cot_details_to_by_id(cot_details_by_model, strategy.cot_weights),
        strategy=strategy,
    )
    voted_score = score_report(records, voted_report)
    expression_only_strategy = VotingStrategy(
        preset="expression_equal",
        expr_weights=dict(EXPR_WEIGHT_PRESETS["equal"]),
        cot_weights={},
        cot_bonus_scale=0.0,
    )
    _, expression_only_report = vote_dataset(
        records=records,
        expr_by_id=expr_details_to_by_id(expr_details_by_model, expression_only_strategy.expr_weights),
        cot_by_id={},
        strategy=expression_only_strategy,
    )
    expression_only_score = score_report(records, expression_only_report)

    selection = {
        **selected,
        "strategy": strategy.to_dict(),
        "val_path": args.val,
        "limit": args.limit,
    }
    report = {
        "selection": selection,
        "strategy_metrics": metrics,
        "single_model_metrics": single_model_metrics(records, expr_details_by_model),
        "expression_vote_metrics": {
            key: value for key, value in expression_only_score.items() if key != "details"
        },
        "expression_vote_details": expression_only_score["details"],
        "selected_vote_metrics": {
            key: value for key, value in voted_score.items() if key != "details"
        },
        "soft_calibrated_vote_metrics": {
            key: value for key, value in voted_score.items() if key != "details"
        },
        "selected_vote_details": voted_score["details"],
    }
    write_csv(f"{output_dir}/val_submit_selected.csv", rows)
    write_json(f"{output_dir}/val_report.json", report)
    write_json(f"{output_dir}/selection.json", selection)
    write_json(f"{output_dir}/val_voted_details.json", voted_report)
    logger.info("验证完成: %s", f"{output_dir}/val_report.json")


def submit(args: argparse.Namespace) -> None:
    selection_path = Path(args.selection)
    if not selection_path.exists():
        raise FileNotFoundError(f"selection 不存在，请先运行 validate: {selection_path}")

    records = load_records(args.test, args.limit)
    set_seed(42)
    submission_dir = args.output_dir
    Path(submission_dir).mkdir(parents=True, exist_ok=True)
    details_dir = args.details_dir or submission_dir

    selection = load_json(str(selection_path))
    strategy_data = selection["strategy"]
    strategy = VotingStrategy(
        preset=strategy_data["preset"],
        expr_weights={key: float(value) for key, value in strategy_data["expr_weights"].items()},
        cot_weights={key: float(value) for key, value in strategy_data["cot_weights"].items()},
        cot_bonus_scale=float(strategy_data["cot_bonus_scale"]),
    )

    expr_details_by_model, cot_details_by_model = prepare_generated_details(
        records=records,
        output_dir=details_dir,
        prefix="test",
        expr_batch_size=args.expr_batch_size,
        cot_batch_size=args.cot_batch_size,
        reuse_existing=args.reuse_existing,
        resume_existing=args.resume_existing,
    )

    missing_expr = {item["name"] for item in EXPR_MODELS} - set(expr_details_by_model)
    missing_cot = {item["name"] for item in COT_MODELS} - set(cot_details_by_model)
    if missing_expr or missing_cot:
        raise FileNotFoundError(f"缺少候选明细: expr={sorted(missing_expr)}, cot={sorted(missing_cot)}")

    rows, voted_report = vote_dataset(
        records=records,
        expr_by_id=expr_details_to_by_id(expr_details_by_model, strategy.expr_weights),
        cot_by_id=cot_details_to_by_id(cot_details_by_model, strategy.cot_weights),
        strategy=strategy,
    )
    source_counts: dict[str, int] = {}
    fallback = 0
    for item in voted_report:
        source = str(item.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        fallback += 1 if item.get("fallback") else 0

    report = {
        "selection": selection,
        "test_path": args.test,
        "limit": args.limit,
        "rows": len(rows),
        "source_counts": source_counts,
        "fallback": fallback,
        "details": voted_report,
    }
    write_csv(f"{submission_dir}/submit_expr4_voted.csv", rows)
    write_csv(f"{submission_dir}/submit_final.csv", rows)
    write_json(f"{submission_dir}/expr4_submit_report.json", report)
    logger.info("提交完成: %s", f"{submission_dir}/submit_final.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="4 表达式模型 + CoT 软校对验证与提交")
    sub = parser.add_subparsers(dest="command", required=True)

    val = sub.add_parser("validate")
    val.add_argument("--val", default="data/splits/train_expr_clean_val.json")
    val.add_argument("--output_dir", default="outputs/evaluation/expr4")
    val.add_argument("--limit", type=int, default=0)
    val.add_argument("--expr_batch_size", type=int, default=16)
    val.add_argument("--cot_batch_size", type=int, default=8)
    val.add_argument("--reuse_existing", action="store_true")
    val.add_argument("--resume_existing", action="store_true")

    submit_parser = sub.add_parser("submit")
    submit_parser.add_argument("--test", default="data/raw/test.json")
    submit_parser.add_argument("--selection", default="outputs/evaluation/expr4/selection.json")
    submit_parser.add_argument("--output_dir", default="outputs/submissions")
    submit_parser.add_argument("--details_dir", default="")
    submit_parser.add_argument("--limit", type=int, default=0)
    submit_parser.add_argument("--expr_batch_size", type=int, default=16)
    submit_parser.add_argument("--cot_batch_size", type=int, default=8)
    submit_parser.add_argument("--reuse_existing", action="store_true")
    submit_parser.add_argument("--resume_existing", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "validate":
        validate(args)
    elif args.command == "submit":
        submit(args)
    else:
        raise SystemExit(f"未知命令: {args.command}")


if __name__ == "__main__":
    main()
