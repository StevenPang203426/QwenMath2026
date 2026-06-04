"""
Audit and clean rule-augmented data by source replacement consistency.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from src.data.expr_builder import _load_json_records, _save_json_records
from src.utils.answer_normalizer import normalize_answer_for_question, normalize_question_text

logger = logging.getLogger("math_solver.augmentation_auditor")


def _load_json(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {str(item["id"]): item for item in records if "id" in item}


def _replace_number_token(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(rf"(?<![\w.]){re.escape(old)}(?![\w.])")
    return pattern.subn(new, text)


def _apply_changes(text: str, changes: list[dict]) -> tuple[str, list[dict]]:
    current = text
    hits = []
    for change in changes:
        old = str(change.get("old", ""))
        new = str(change.get("new", ""))
        current, count = _replace_number_token(current, old, new)
        hits.append({"old": old, "new": new, "count": count})
    return current, hits


def _audit_one(item: dict, raw_by_id: dict[str, dict], expr_by_id: dict[str, dict]) -> tuple[dict | None, dict | None]:
    source_id = str(item.get("source_id", ""))
    raw_item = raw_by_id.get(source_id)
    expr_item = expr_by_id.get(source_id)
    if raw_item is None:
        return None, _reject(item, "missing_raw_source")
    if expr_item is None:
        return None, _reject(item, "missing_expr_source")

    changes = item.get("changed_numbers") or []
    if not changes:
        return None, _reject(item, "missing_changed_numbers")

    source_question_raw = str(raw_item.get("question", ""))
    source_question_clean = normalize_question_text(raw_item.get("question", ""))
    source_expression = str(expr_item.get("expression", ""))
    expected_question_raw, raw_hits = _apply_changes(source_question_raw, changes)
    expected_question_clean, clean_hits = _apply_changes(source_question_clean, changes)
    expected_expression, expr_hits = _apply_changes(source_expression, changes)

    actual_question = str(item.get("question", ""))
    actual_expression = str(item.get("expression", ""))
    question_matches = actual_question in {expected_question_raw, expected_question_clean}
    expression_matches = actual_expression == expected_expression
    question_hits = raw_hits if actual_question == expected_question_raw else clean_hits
    all_question_hits = all(hit["count"] > 0 for hit in question_hits)
    all_expr_hits = all(hit["count"] > 0 for hit in expr_hits)

    if not (question_matches and expression_matches and all_question_hits and all_expr_hits):
        return None, {
            **_reject(item, "source_replacement_mismatch"),
            "expected_question_raw": expected_question_raw,
            "expected_question_clean": expected_question_clean,
            "expected_expression": expected_expression,
            "question_hits": question_hits,
            "expression_hits": expr_hits,
        }

    clean_question = normalize_question_text(actual_question)
    clean_answer = normalize_answer_for_question(str(item.get("answer", "")), clean_question)
    clean_item = {
        **item,
        "question": clean_question,
        "answer": clean_answer,
        "source": item.get("source", "rule_augment"),
        "audit_status": "clean",
        "correspondence_check": "source_replacement_exact",
    }
    if clean_question != actual_question:
        clean_item["question_cleaned"] = True
    if clean_answer != str(item.get("answer", "")):
        clean_item["answer_format_normalized"] = True
        clean_item["original_answer"] = str(item.get("answer", ""))
    return clean_item, None


def _reject(item: dict, reason: str) -> dict:
    return {
        **item,
        "audit_status": "rejected",
        "reject_reason": reason,
    }


def audit_augmented_data(
    augmented_path: str,
    raw_path: str,
    expr_path: str,
    clean_output: str,
    rejected_output: str,
    report_output: str,
    limit: int = 0,
) -> list[dict]:
    augmented = _load_json(augmented_path)
    if limit > 0:
        augmented = augmented[:limit]
    raw_by_id = _by_id(_load_json(raw_path))
    expr_by_id = _by_id(_load_json_records(expr_path))

    clean: list[dict] = []
    rejected: list[dict] = []
    stats: Counter[str] = Counter()
    for item in augmented:
        stats["total"] += 1
        if str(item.get("question", "")).strip().startswith("["):
            stats["question_list_string_format"] += 1
        clean_item, rejected_item = _audit_one(item, raw_by_id, expr_by_id)
        if clean_item is not None:
            clean.append(clean_item)
            stats["clean"] += 1
            if clean_item.get("question_cleaned"):
                stats["question_cleaned"] += 1
            if clean_item.get("answer_format_normalized"):
                stats["answer_format_normalized"] += 1
        elif rejected_item is not None:
            rejected.append(rejected_item)
            stats["rejected"] += 1
            stats[f"rejected:{rejected_item.get('reject_reason', 'unknown')}"] += 1

    report = [{
        "input": augmented_path,
        "clean": len(clean),
        "rejected": len(rejected),
        "stats": dict(stats),
        "correctness_basis": "question and expression must both be exact source_id + changed_numbers replacements",
    }]
    _save_json_records(clean_output, clean)
    _save_json_records(rejected_output, rejected)
    _save_json_records(report_output, report)
    logger.info("增强审计完成: clean=%s rejected=%s -> %s", len(clean), len(rejected), clean_output)
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description="审计并清洗规则增强数据")
    parser.add_argument("--augmented", default="data/processed/train_augmented.json")
    parser.add_argument("--raw", default="data/raw/train.json")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument(
        "--clean",
        default="data/processed/intermediate/augmentation/train_augmented_clean.json",
    )
    parser.add_argument(
        "--rejected",
        default="data/processed/intermediate/augmentation/train_augmented_rejected.json",
    )
    parser.add_argument(
        "--report",
        default="data/processed/intermediate/augmentation/augmentation_report.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    audit_augmented_data(
        augmented_path=args.augmented,
        raw_path=args.raw,
        expr_path=args.expr,
        clean_output=args.clean,
        rejected_output=args.rejected,
        report_output=args.report,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
