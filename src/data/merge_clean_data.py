"""
Merge raw high-confidence, repaired, and augmented records.
"""
import argparse
import json
import logging
from pathlib import Path
from collections import Counter

from src.data.expr_builder import _load_json_records, _save_json_records
from src.data.rule_augmentor import BAD_AUDIT_LABELS
from src.utils.answer_normalizer import (
    answers_match_by_question,
    format_auto,
    normalize_question_text,
    safe_eval_expression,
)

logger = logging.getLogger("math_solver.merge_clean_data")


def _load_json(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _audit_by_id(audit_path: str) -> dict[str, dict]:
    if not Path(audit_path).exists() and audit_path == "data/processed/intermediate/quality/quality_audit.json":
        legacy = "data/processed/quality_audit.json"
        if Path(legacy).exists():
            audit_path = legacy
    return {str(item["id"]): item for item in _load_json(audit_path) if "id" in item}


def _valid_expr_by_id(expr_path: str) -> dict[str, dict]:
    records = {}
    if not Path(expr_path).exists():
        return records
    for item in _load_json_records(expr_path):
        if item.get("status") == "ok" and item.get("valid") is True and item.get("expression"):
            records[str(item["id"])] = item
    return records


def _expr_by_id(expr_path: str) -> dict[str, dict]:
    if not Path(expr_path).exists():
        return {}
    return {str(item["id"]): item for item in _load_json_records(expr_path) if "id" in item}


def _raw_is_usable(item: dict, audit: dict | None) -> bool:
    if audit is None:
        return True
    return audit.get("label") not in BAD_AUDIT_LABELS


def _dedupe_raw_records(raw_data: list[dict], expr_path: str) -> tuple[list[dict], int]:
    grouped: dict[str, list[dict]] = {}
    order = []
    for item in raw_data:
        item_id = str(item.get("id", ""))
        if item_id not in grouped:
            order.append(item_id)
            grouped[item_id] = []
        grouped[item_id].append(item)

    all_expr = _expr_by_id(expr_path)
    deduped = []
    skipped = 0
    for item_id in order:
        group = grouped[item_id]
        if len(group) == 1:
            deduped.append(group[0])
            continue
        skipped += len(group) - 1
        chosen = _choose_raw_duplicate(item_id, group, all_expr.get(item_id))
        deduped.append(chosen)
    return deduped, skipped


def _choose_raw_duplicate(item_id: str, group: list[dict], expr_item: dict | None) -> dict:
    if expr_item and expr_item.get("expression"):
        try:
            value = safe_eval_expression(str(expr_item["expression"]))
            raw_answer = format_auto(value)
            for item in group:
                question = normalize_question_text(item.get("question", ""))
                if answers_match_by_question(raw_answer, str(item.get("answer", "")), question):
                    return item
        except Exception:
            pass
    return group[0]


def _unify_repairs(repaired: list[dict], format_repaired: list[dict]) -> tuple[list[dict], int]:
    unified = []
    seen_source_ids = set()
    skipped = 0
    for item in repaired:
        source_id = str(item.get("source_id", ""))
        if source_id in seen_source_ids:
            skipped += 1
            continue
        seen_source_ids.add(source_id)
        unified.append({**item, "source": item.get("source", "auto_repair")})
    for item in format_repaired:
        source_id = str(item.get("source_id", ""))
        if source_id in seen_source_ids:
            skipped += 1
            continue
        seen_source_ids.add(source_id)
        unified.append({**item, "source": item.get("source", "expr_format_repair")})
    return unified, skipped


def _source_percentages(source_counts: dict[str, int], total: int) -> dict[str, float]:
    if total == 0:
        return {source: 0.0 for source in source_counts}
    return {
        source: round(count / total * 100, 2)
        for source, count in source_counts.items()
    }


def _write_merge_report(
    report_output: str,
    merged: list[dict],
    skipped_raw: int,
    skipped_augmented: int,
    skipped_repair_duplicates: int,
    skipped_raw_duplicates: int,
) -> None:
    if not report_output:
        return
    source_counts = dict(sorted(Counter(str(item.get("source", "unknown")) for item in merged).items()))
    report = [{
        "total_records": len(merged),
        "source_counts": source_counts,
        "source_percentages": _source_percentages(source_counts, len(merged)),
        "source_descriptions": {
            "raw_ok": "原始合格数据",
            "auto_repair": "题意修复数据",
            "expr_format_repair": "表达式结果规范修复数据",
            "rule_augment": "增强后的合格数据",
        },
        "skipped": {
            "raw": skipped_raw,
            "augmented": skipped_augmented,
            "repair_duplicates": skipped_repair_duplicates,
            "raw_duplicates": skipped_raw_duplicates,
        },
    }]
    _save_json_records(report_output, report)


def merge_clean_data(
    raw_path: str,
    audit_path: str,
    expr_path: str,
    repaired_path: str,
    format_repaired_path: str,
    augmented_path: str,
    output_path: str,
    unified_repairs_output: str = "",
    require_expr_for_raw: bool = False,
    report_output: str = "",
) -> list[dict]:
    raw_data, skipped_raw_duplicates = _dedupe_raw_records(_load_json(raw_path), expr_path)
    audit_map = _audit_by_id(audit_path)
    expr_map = _valid_expr_by_id(expr_path)
    repaired = _load_json(repaired_path)
    format_repaired = _load_json(format_repaired_path)
    augmented = _load_json(augmented_path)
    unified_repairs, skipped_repair_duplicates = _unify_repairs(repaired, format_repaired)
    if unified_repairs_output:
        _save_json_records(unified_repairs_output, unified_repairs)

    merged = []
    repaired_source_ids = {str(item.get("source_id")) for item in unified_repairs}
    skipped_raw = 0
    skipped_augmented = 0

    for item in raw_data:
        item_id = str(item["id"])
        if item_id in repaired_source_ids:
            skipped_raw += 1
            continue
        if not _raw_is_usable(item, audit_map.get(item_id)):
            skipped_raw += 1
            continue
        expr_item = expr_map.get(item_id)
        if require_expr_for_raw and expr_item is None:
            skipped_raw += 1
            continue
        merged_item = {
            "id": item_id,
            "source_id": item_id,
            "source": "raw_ok",
            "question": normalize_question_text(item.get("question", "")),
            "answer": str(item["answer"]),
            "instruction": item.get("instruction", ""),
        }
        if expr_item:
            merged_item["expression"] = expr_item["expression"]
        merged.append(merged_item)

    for item in unified_repairs:
        merged.append(item)
    for item in augmented:
        if str(item.get("source_id", "")) in repaired_source_ids:
            skipped_augmented += 1
            continue
        merged.append({**item, "source": item.get("source", "rule_augment")})

    _save_json_records(output_path, merged)
    _write_merge_report(
        report_output=report_output,
        merged=merged,
        skipped_raw=skipped_raw,
        skipped_augmented=skipped_augmented,
        skipped_repair_duplicates=skipped_repair_duplicates,
        skipped_raw_duplicates=skipped_raw_duplicates,
    )
    logger.info(
        f"合并完成: raw={len(raw_data)-skipped_raw}, repaired={len(unified_repairs)}, "
        f"augmented={len(augmented)-skipped_augmented}, total={len(merged)} -> {output_path} "
        f"(重复修复跳过: {skipped_repair_duplicates}, 重复原始题跳过: {skipped_raw_duplicates})"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="合并清洗与增强后的训练数据")
    parser.add_argument("--raw", default="data/raw/train.json")
    parser.add_argument("--audit", default="data/processed/intermediate/quality/quality_audit.json")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument("--repaired", default="data/processed/train_repaired.json")
    parser.add_argument(
        "--format_repaired",
        default="data/processed/intermediate/format_repair/expr_format_repaired.json",
    )
    parser.add_argument(
        "--augmented",
        default="data/processed/intermediate/augmentation/train_augmented_clean.json",
    )
    parser.add_argument("--output", default="data/processed/train_clean_augmented.json")
    parser.add_argument("--unified_repairs_output", default="data/processed/train_repairs_unified.json")
    parser.add_argument("--require_expr_for_raw", action="store_true")
    parser.add_argument("--report", default="data/processed/train_clean_augmented_report.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    merge_clean_data(
        raw_path=args.raw,
        audit_path=args.audit,
        expr_path=args.expr,
        repaired_path=args.repaired,
        format_repaired_path=args.format_repaired,
        augmented_path=args.augmented,
        output_path=args.output,
        unified_repairs_output=args.unified_repairs_output,
        require_expr_for_raw=args.require_expr_for_raw,
        report_output=args.report,
    )


if __name__ == "__main__":
    main()
