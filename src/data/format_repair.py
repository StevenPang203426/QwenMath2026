"""
Repair expression records that fail only because answer formatting is wrong.
"""
from __future__ import annotations

import argparse
import logging
from collections import Counter

from src.data.expr_builder import _load_json_records, _save_json_records
from src.utils.answer_normalizer import (
    answers_match_by_question,
    detect_answer_format,
    format_auto,
    normalize_answer_for_question,
    normalize_question_text,
    safe_eval_expression,
)

logger = logging.getLogger("math_solver.format_repair")


def _value_to_raw_answer(value: float) -> str:
    return format_auto(value)


def _repair_one(item: dict) -> tuple[dict | None, dict | None]:
    item_id = str(item.get("id", ""))
    question = normalize_question_text(item.get("question", ""))
    gold_answer = str(item.get("answer", "")).strip()
    expression = str(item.get("expression", "")).strip()
    if not expression:
        return None, _rejected(item, "missing_expression")

    try:
        value = safe_eval_expression(expression)
    except Exception as exc:
        return None, _rejected(item, f"eval_failed: {exc}")

    raw_eval = _value_to_raw_answer(value)
    formatted_eval = normalize_answer_for_question(raw_eval, question)
    decision = detect_answer_format(question)

    if not answers_match_by_question(raw_eval, gold_answer, question):
        return None, {
            **_base_record(item, question, gold_answer, expression),
            "eval_result": raw_eval,
            "formatted_eval_result": formatted_eval,
            "answer_type": decision.answer_type,
            "format_param": decision.param,
            "reject_reason": "formatted_result_mismatch",
        }

    return {
        **_base_record(item, question, gold_answer, expression),
        "id": f"format_repair_{item_id}",
        "source": "expr_format_repair",
        "eval_result": raw_eval,
        "formatted_eval_result": formatted_eval,
        "answer_type": decision.answer_type,
        "format_param": decision.param,
        "repair_reason": f"format_normalized:{decision.reason}",
        "original_status": item.get("status", ""),
        "original_error": item.get("error", ""),
    }, None


def _base_record(item: dict, question: str, answer: str, expression: str) -> dict:
    item_id = str(item.get("id", ""))
    return {
        "source_id": item_id,
        "question": question,
        "answer": answer,
        "expression": expression,
    }


def _rejected(item: dict, reason: str) -> dict:
    return {
        "id": str(item.get("id", "")),
        "source_id": str(item.get("id", "")),
        "question": normalize_question_text(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "expression": str(item.get("expression", "")),
        "reject_reason": reason,
        "original_status": item.get("status", ""),
        "original_error": item.get("error", ""),
    }


def build_format_repair_data(
    expr_path: str,
    repaired_output: str,
    rejected_output: str,
    report_output: str,
    limit: int = 0,
) -> list[dict]:
    records = _load_json_records(expr_path)
    if limit > 0:
        records = records[:limit]

    repaired: list[dict] = []
    rejected: list[dict] = []
    stats: Counter[str] = Counter()

    for item in records:
        if item.get("valid") is True and item.get("status") == "ok":
            stats["already_valid"] += 1
            continue
        stats["candidates"] += 1
        repaired_item, rejected_item = _repair_one(item)
        if repaired_item is not None:
            repaired.append(repaired_item)
            stats["repaired"] += 1
        elif rejected_item is not None:
            rejected.append(rejected_item)
            stats[f"rejected:{rejected_item.get('reject_reason', 'unknown')}"] += 1

    report = [{
        "input": expr_path,
        "total_records": len(records),
        "repaired": len(repaired),
        "rejected": len(rejected),
        "stats": dict(stats),
    }]
    _save_json_records(repaired_output, repaired)
    _save_json_records(rejected_output, rejected)
    _save_json_records(report_output, report)
    logger.info(
        "格式修复完成: repaired=%s rejected=%s -> %s",
        len(repaired),
        len(rejected),
        repaired_output,
    )
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="修复表达式正确但答案格式不符合题意的数据")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument(
        "--output",
        default="data/processed/intermediate/format_repair/expr_format_repaired.json",
    )
    parser.add_argument(
        "--rejected",
        default="data/processed/intermediate/format_repair/expr_format_rejected.json",
    )
    parser.add_argument(
        "--report",
        default="data/processed/intermediate/format_repair/format_repair_report.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_format_repair_data(args.expr, args.output, args.rejected, args.report, args.limit)


if __name__ == "__main__":
    main()
