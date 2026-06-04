"""
Rule-based numeric augmentation for high-confidence expression data.

Only records with verified correct expressions are augmented. Answers are
computed by safe_eval instead of LLM arithmetic.
"""
import argparse
import json
import logging
import random
import re
from pathlib import Path

from src.data.expr_builder import _load_json_records, _save_json_records, safe_eval
from src.utils.answer_normalizer import normalize_answer_for_question, normalize_question_text

logger = logging.getLogger("math_solver.rule_augmentor")

BAD_AUDIT_LABELS = {"ambiguous", "ocr_error", "missing_symbol", "wrong_answer", "unsolvable", "needs_human"}
_NUMBER_PATTERN = re.compile(r'(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])')

# 禁止替换的常数：圆周率、时间/日历单位、百分基数、0/1/2 等基础常数
_FROZEN_NUMBERS = {
    "3.14",   # 圆周率
    "0", "1", "2",  # 基础常数（经常是倍数/比例基准）
    "10", "100", "1000",  # 进制基数
    "12",     # 月数
    "24",     # 小时
    "60",     # 分钟/秒
    "365",    # 天数
    "7",      # 周天数
    "30",     # 月天数（近似）
    "360",    # 角度
    "180",    # 角度
}


def _load_json(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _audit_excluded_ids(audit_path: str) -> set[str]:
    path = Path(audit_path)
    if not path.exists() and audit_path == "data/processed/intermediate/quality/quality_audit.json":
        legacy = Path("data/processed/quality_audit.json")
        if legacy.exists():
            path = legacy
    if not path.exists():
        return set()
    excluded = set()
    for item in _load_json(str(path)):
        label = item.get("label")
        if label in BAD_AUDIT_LABELS:
            excluded.add(str(item["id"]))
    return excluded


def _valid_expr_records(expr_path: str) -> dict[str, dict]:
    records = {}
    for item in _load_json_records(expr_path):
        if item.get("status") == "ok" and item.get("valid") is True and item.get("expression"):
            records[str(item["id"])] = item
    return records


def _extract_numbers(text: str) -> list[str]:
    return [match.group(0) for match in _NUMBER_PATTERN.finditer(text)]


def _format_number(value: float, prefer_int: bool) -> str:
    if prefer_int:
        return str(max(1, int(round(value))))
    value = round(value, 4)
    return str(int(value)) if value == int(value) else str(value).rstrip("0").rstrip(".")


def _perturb_number(num_str: str, rng: random.Random, ratio: float) -> str:
    value = float(num_str)
    if value == 0:
        return str(rng.randint(1, 9))
    prefer_int = "." not in num_str
    scale = rng.uniform(1 - ratio, 1 + ratio)
    new_value = value * scale
    if new_value == value:
        new_value += 1
    if prefer_int:
        new_value = max(1, round(new_value))
        if int(new_value) == int(float(num_str)):
            new_value += 1
    return _format_number(new_value, prefer_int)


def _replace_number_token(text: str, old: str, new: str) -> tuple[str, int]:
    pattern = re.compile(rf'(?<![\w.]){re.escape(old)}(?![\w.])')
    return pattern.subn(new, text)


def _answer_from_expr(expr: str, question: str) -> str | None:
    try:
        value = safe_eval(expr.replace("×", "*").replace("÷", "/").replace("^", "**"))
    except Exception:
        return None
    raw_answer = str(int(value)) if value == int(value) else str(round(value, 6)).rstrip("0").rstrip(".")
    return normalize_answer_for_question(raw_answer, question)


def _augment_one(item: dict, expr_item: dict, rng: random.Random, ratio: float) -> dict | None:
    question = normalize_question_text(item.get("question", ""))
    expression = str(expr_item.get("expression", ""))
    question_nums = set(_extract_numbers(question))
    expr_nums = [
        num for num in _extract_numbers(expression)
        if num in question_nums and float(num) != 0 and num not in _FROZEN_NUMBERS
    ]
    if not expr_nums:
        return None

    rng.shuffle(expr_nums)
    for old_num in expr_nums:
        new_num = _perturb_number(old_num, rng, ratio)
        if new_num == old_num:
            continue
        new_question, q_count = _replace_number_token(question, old_num, new_num)
        new_expression, e_count = _replace_number_token(expression, old_num, new_num)
        if q_count == 0 or e_count == 0 or new_question == question or new_expression == expression:
            continue
        new_answer = _answer_from_expr(new_expression, new_question)
        if new_answer is None:
            continue
        return {
            "question": new_question,
            "answer": new_answer,
            "expression": new_expression,
            "changed_numbers": [{"old": old_num, "new": new_num}],
        }
    return None


def augment_high_confidence_data(
    input_path: str,
    expr_path: str,
    audit_path: str,
    output_path: str,
    num_augments_per_sample: int = 1,
    ratio: float = 0.3,
    seed: int = 42,
    limit: int = 0,
) -> list[dict]:
    data = _load_json(input_path)
    if limit > 0:
        data = data[:limit]
    expr_by_id = _valid_expr_records(expr_path)
    excluded_ids = _audit_excluded_ids(audit_path)
    rng = random.Random(seed)

    augmented = []
    skipped = 0
    for item in data:
        item_id = str(item["id"])
        if item_id in excluded_ids or item_id not in expr_by_id:
            skipped += 1
            continue
        for aug_idx in range(num_augments_per_sample):
            aug = _augment_one(item, expr_by_id[item_id], rng, ratio)
            if aug is None:
                skipped += 1
                continue
            augmented.append({
                "id": f"aug_{item_id}_{aug_idx}",
                "source_id": item_id,
                "source": "rule_augment",
                **aug,
            })

    _save_json_records(output_path, augmented)
    logger.info(f"规则增强: {len(augmented)} 条，跳过: {skipped} 条 -> {output_path}")
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="规则优先数据增强")
    parser.add_argument("--input", default="data/raw/train.json")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument("--audit", default="data/processed/intermediate/quality/quality_audit.json")
    parser.add_argument("--output", default="data/processed/train_augmented.json")
    parser.add_argument("--num_augments", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    augment_high_confidence_data(
        input_path=args.input,
        expr_path=args.expr,
        audit_path=args.audit,
        output_path=args.output,
        num_augments_per_sample=args.num_augments,
        ratio=args.ratio,
        seed=args.seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
