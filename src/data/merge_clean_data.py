"""
Merge raw high-confidence, repaired, and augmented records.
"""
import argparse
import json
import logging
from pathlib import Path

from src.data.expr_builder import _load_json_records, _save_json_records
from src.data.rule_augmentor import BAD_AUDIT_LABELS

logger = logging.getLogger("math_solver.merge_clean_data")


def _load_json(path: str) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _audit_by_id(audit_path: str) -> dict[str, dict]:
    return {str(item["id"]): item for item in _load_json(audit_path) if "id" in item}


def _valid_expr_by_id(expr_path: str) -> dict[str, dict]:
    records = {}
    if not Path(expr_path).exists():
        return records
    for item in _load_json_records(expr_path):
        if item.get("status") == "ok" and item.get("valid") is True and item.get("expression"):
            records[str(item["id"])] = item
    return records


def _raw_is_usable(item: dict, audit: dict | None) -> bool:
    if audit is None:
        return True
    return audit.get("label") not in BAD_AUDIT_LABELS


def merge_clean_data(
    raw_path: str,
    audit_path: str,
    expr_path: str,
    repaired_path: str,
    augmented_path: str,
    output_path: str,
    require_expr_for_raw: bool = False,
) -> list[dict]:
    raw_data = _load_json(raw_path)
    audit_map = _audit_by_id(audit_path)
    expr_map = _valid_expr_by_id(expr_path)
    repaired = _load_json(repaired_path)
    augmented = _load_json(augmented_path)

    merged = []
    repaired_source_ids = {str(item.get("source_id")) for item in repaired}
    skipped_raw = 0

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
            "question": item["question"],
            "answer": str(item["answer"]),
            "instruction": item.get("instruction", ""),
        }
        if expr_item:
            merged_item["expression"] = expr_item["expression"]
        merged.append(merged_item)

    for item in repaired:
        merged.append({**item, "source": item.get("source", "auto_repair")})
    for item in augmented:
        merged.append({**item, "source": item.get("source", "rule_augment")})

    _save_json_records(output_path, merged)
    logger.info(
        f"合并完成: raw={len(raw_data)-skipped_raw}, repaired={len(repaired)}, "
        f"augmented={len(augmented)}, total={len(merged)} -> {output_path}"
    )
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="合并清洗与增强后的训练数据")
    parser.add_argument("--raw", default="data/raw/train.json")
    parser.add_argument("--audit", default="data/processed/quality_audit.json")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument("--repaired", default="data/processed/train_repaired.json")
    parser.add_argument("--augmented", default="data/processed/train_augmented.json")
    parser.add_argument("--output", default="data/processed/train_clean_augmented.json")
    parser.add_argument("--require_expr_for_raw", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    merge_clean_data(
        raw_path=args.raw,
        audit_path=args.audit,
        expr_path=args.expr,
        repaired_path=args.repaired,
        augmented_path=args.augmented,
        output_path=args.output,
        require_expr_for_raw=args.require_expr_for_raw,
    )


if __name__ == "__main__":
    main()
