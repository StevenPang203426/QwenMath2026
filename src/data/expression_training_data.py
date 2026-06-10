"""
Build clean expression-route training data.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from typing import Any

from src.data.expr_builder import _load_json_records, _save_json_records
from src.utils.answer_normalizer import (
    answers_match_by_question,
    format_auto,
    normalize_question_text,
    safe_eval_expression,
)
from src.utils.expression_policy import validate_expression

logger = logging.getLogger("math_solver.expression_training_data")

EXPR_INSTRUCTION = (
    "请为以下数学题写出一个Python可直接计算的中缀数学表达式。"
    "用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。"
)


def _safe_expression_check(item: dict[str, Any]) -> tuple[bool, str, str]:
    expr = str(item.get("expression", "")).strip()
    if not expr:
        return False, "missing_expression", ""

    policy = validate_expression(expr)
    if not policy.ok:
        return False, "policy_violation", ""

    try:
        raw_answer = format_auto(safe_eval_expression(policy.normalized))
    except Exception:
        return False, "eval_failed", ""

    question = normalize_question_text(item.get("question", ""))
    if not answers_match_by_question(raw_answer, str(item.get("answer", "")), question):
        return False, "answer_mismatch", raw_answer
    return True, "ok", raw_answer


def _expr_record(item: dict[str, Any], raw_eval_result: str) -> dict[str, Any]:
    item_id = str(item.get("id", ""))
    source_id = str(item.get("source_id", item_id))
    return {
        "id": item_id,
        "source_id": source_id,
        "source": item.get("source", ""),
        "question": normalize_question_text(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "expression": str(item.get("expression", "")).strip(),
        "raw_eval_result": raw_eval_result,
        "instruction": EXPR_INSTRUCTION,
    }


def _split_by_source(records: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        groups[str(item.get("source", ""))].append(item)

    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    for source in sorted(groups):
        group = list(groups[source])
        rng.shuffle(group)
        val_count = int(round(len(group) * val_ratio))
        if len(group) > 1 and val_ratio > 0:
            val_count = max(1, min(len(group) - 1, val_count))
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    return _sort_records(train), _sort_records(val)


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[int, object]:
        item_id = str(item.get("id", ""))
        return (0, int(item_id)) if item_id.isdigit() else (1, item_id)

    return sorted(records, key=key)


def build_expression_clean_splits(
    input_path: str,
    output_path: str,
    train_output: str,
    val_output: str,
    report_output: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    records = []
    reject_reasons: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for item in _load_json_records(input_path):
        ok, reason, raw_eval_result = _safe_expression_check(item)
        if not ok:
            reject_reasons[reason] += 1
            continue
        record = _expr_record(item, raw_eval_result)
        records.append(record)
        source_counts[str(record.get("source", ""))] += 1

    records = _sort_records(records)
    train, val = _split_by_source(records, val_ratio=val_ratio, seed=seed)
    report = [{
        "input": input_path,
        "output": output_path,
        "total_records": len(_load_json_records(input_path)),
        "kept": len(records),
        "rejected": sum(reject_reasons.values()),
        "reject_reasons": dict(reject_reasons),
        "source_counts": dict(source_counts),
        "train": len(train),
        "val": len(val),
        "val_ratio": val_ratio,
        "seed": seed,
    }]

    _save_json_records(output_path, records)
    _save_json_records(train_output, train)
    _save_json_records(val_output, val)
    _save_json_records(report_output, report)
    logger.info("表达式 clean 数据: kept=%s rejected=%s", len(records), sum(reject_reasons.values()))
    return records


def _wrong_candidate_by_id(wrong_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in wrong_records:
        item_id = str(item.get("id", ""))
        ok, reason, raw_eval_result = _safe_expression_check(item)
        if ok:
            continue
        if reason != "answer_mismatch" or not raw_eval_result:
            continue
        result[item_id] = {**item, "raw_eval_result": raw_eval_result}
    return result


def build_expression_dpo_pairs(
    positive_path: str,
    wrong_path: str,
    output_path: str,
    report_output: str,
) -> list[dict]:
    positives = _load_json_records(positive_path)
    wrong_by_id = _wrong_candidate_by_id(_load_json_records(wrong_path))
    pairs = []
    skipped_sources: Counter[str] = Counter()

    for item in positives:
        item_id = str(item.get("id", ""))
        wrong = wrong_by_id.get(item_id)
        if wrong is None:
            skipped_sources[str(item.get("source", ""))] += 1
            continue
        pairs.append({
            "id": item_id,
            "source_id": str(item.get("source_id", item_id)),
            "source": item.get("source", ""),
            "question": normalize_question_text(item.get("question", "")),
            "instruction": EXPR_INSTRUCTION,
            "chosen": f"<expr>{item['expression']}</expr><answer>{item['answer']}</answer>",
            "rejected": f"<expr>{wrong['expression']}</expr><answer>{wrong['raw_eval_result']}</answer>",
        })

    report = [{
        "positive": positive_path,
        "wrong": wrong_path,
        "paired": len(pairs),
        "positive_total": len(positives),
        "wrong_candidates": len(wrong_by_id),
        "skipped_positive_sources": dict(skipped_sources),
    }]
    _save_json_records(output_path, pairs)
    _save_json_records(report_output, report)
    logger.info("表达式 DPO 偏好对: %s -> %s", len(pairs), output_path)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="构建表达式路线 clean SFT/DPO 数据")
    sub = parser.add_subparsers(dest="mode", required=True)

    clean = sub.add_parser("clean")
    clean.add_argument("--input", default="data/processed/train_clean_augmented.json")
    clean.add_argument("--output", default="data/processed/train_expr_clean.json")
    clean.add_argument("--train", default="data/splits/train_expr_clean_train.json")
    clean.add_argument("--val", default="data/splits/train_expr_clean_val.json")
    clean.add_argument("--report", default="data/processed/train_expr_clean_report.json")
    clean.add_argument("--val_ratio", type=float, default=0.1)
    clean.add_argument("--seed", type=int, default=42)

    dpo = sub.add_parser("dpo")
    dpo.add_argument("--positive", default="data/processed/train_expr_clean.json")
    dpo.add_argument("--wrong", default="data/processed/expr_wrong.jsonl")
    dpo.add_argument("--output", default="data/processed/train_expr_dpo_clean.json")
    dpo.add_argument("--report", default="data/processed/train_expr_dpo_clean_report.json")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.mode == "clean":
        build_expression_clean_splits(
            input_path=args.input,
            output_path=args.output,
            train_output=args.train,
            val_output=args.val,
            report_output=args.report,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        build_expression_dpo_pairs(
            positive_path=args.positive,
            wrong_path=args.wrong,
            output_path=args.output,
            report_output=args.report,
        )


if __name__ == "__main__":
    main()
