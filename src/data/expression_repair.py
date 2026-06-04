"""
Idempotent repair pipeline for expression data.

This module keeps the DeepSeek-facing repair flow behind a small interface:
given raw records and existing expression records, write safe/repaired/rejected
artifacts without mutating the original files.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from collections import Counter
from typing import Any, Callable

from src.data.expr_builder import DEFAULT_MODEL, DEEPSEEK_BASE_URL, _load_json_records, _save_json_records
from src.utils.answer_normalizer import (
    answers_match_by_question,
    format_auto,
    normalize_answer_for_question,
    normalize_question_text,
    safe_eval_expression,
)
from src.utils.expression_policy import validate_expression

logger = logging.getLogger("math_solver.expression_repair")

RepairRequestFn = Callable[[dict, int, list[dict[str, Any]]], dict[str, Any] | None]


_REPAIR_SYSTEM_PROMPT = "你是小学数学应用题的规范列式修复员。只输出一个可解析表达式。"

_REPAIR_CACHE_PREFIX = """
你正在修复小学数学应用题的表达式数据。请根据题目和标注答案，重新给出一个规范表达式。

【规范表达式要求】
- 只允许数字、+、-、*、/、**、英文括号，以及一元正负号。
- 禁止方程或等号，例如 x+3=7。
- 禁止 if/else、三元表达式、比较表达式，例如 >、<、>=、<=、==。
- 禁止任何函数调用，例如 abs、round、ceil、floor、max、min。
- 禁止 %、//、变量、中文、单位、LaTeX、列表、字典、字符串。
- 百分数写成除法或小数，例如 25/100。
- 分数直接写除法，例如 2/3。
- 至少、至多、保留小数等题意格式不要写成函数；只写能得到原始数值的纯算术表达式，最终格式由后处理完成。
- 圆周率使用 3.14。

【输出格式】
只输出 <expr>...</expr>，不要解释，不要推理过程，不要 Markdown。
""".strip()

_CANDIDATE_SEPARATOR = "\n\n【候选题目 JSON】\n"


def build_repair_messages(item: dict) -> list[dict[str, Any]]:
    payload = {
        "id": str(item.get("id", "")),
        "question": normalize_question_text(item.get("question", "")),
        "answer": str(item.get("answer", "")),
        "previous_expression": str(item.get("expression", "")),
        "previous_error": str(item.get("error", "")),
        "policy_reasons": list(item.get("policy_reasons", [])),
    }
    user_text = (
        _REPAIR_CACHE_PREFIX
        + _CANDIDATE_SEPARATOR
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return [
        {"role": "system", "content": _REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def repair_expression_records(
    raw_path: str,
    expr_path: str,
    safe_output: str,
    repaired_output: str,
    rejected_output: str,
    report_output: str,
    sft_output: str = "",
    request_fn: RepairRequestFn | None = None,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    max_attempts: int = 3,
    limit: int = 0,
    checkpoint_every: int = 0,
) -> list[dict]:
    raw_records = _load_json_records(raw_path)
    if limit > 0:
        raw_records = raw_records[:limit]
    total_records = len(raw_records)
    existing_by_id = {str(item["id"]): item for item in _load_json_records(expr_path) if "id" in item}
    if request_fn is not None:
        request = request_fn
    elif max_attempts <= 0:
        request = lambda item, attempt, messages: None
    else:
        request = _build_deepseek_request(api_key=api_key, model=model)

    safe: list[dict] = []
    repaired: list[dict] = []
    rejected: list[dict] = []
    stats: Counter[str] = Counter()
    cache_stats: Counter[str] = Counter()

    def write_checkpoint(processed: int, complete: bool) -> None:
        report = _build_repair_report(
            raw_path=raw_path,
            expr_path=expr_path,
            safe=safe,
            repaired=repaired,
            rejected=rejected,
            stats=stats,
            cache_stats=cache_stats,
            max_attempts=max_attempts,
            processed=processed,
            total=total_records,
            complete=complete,
        )
        _write_repair_outputs(
            safe_output=safe_output,
            repaired_output=repaired_output,
            rejected_output=rejected_output,
            report_output=report_output,
            sft_output=sft_output,
            safe=safe,
            repaired=repaired,
            rejected=rejected,
            report=report,
        )
        if not complete:
            logger.info(
                "表达式修复进度 %s/%s safe=%s repaired=%s rejected=%s",
                processed,
                total_records,
                len(safe),
                len(repaired),
                len(rejected),
            )

    for processed, raw_item in enumerate(raw_records, start=1):
        item_id = str(raw_item.get("id", ""))
        existing = existing_by_id.get(item_id)
        candidate = _candidate_from(raw_item, existing)
        check = verify_safe_expression(
            candidate.get("expression", ""),
            candidate.get("answer", ""),
            candidate.get("question", ""),
        )
        if check["safe"]:
            safe.append(_safe_record(candidate, check, source=candidate.get("source", "expr_existing")))
            stats["safe_existing"] += 1
            if checkpoint_every > 0 and processed % checkpoint_every == 0:
                write_checkpoint(processed, complete=False)
            continue

        candidate["policy_reasons"] = check.get("policy_reasons", [])
        last_check = check
        last_expr = str(candidate.get("expression", ""))
        raw_response = ""
        for attempt in range(1, max_attempts + 1):
            messages = build_repair_messages(candidate)
            response = request(candidate, attempt, messages)
            _accumulate_cache_stats(cache_stats, response)
            expr = _extract_expression_response(response)
            raw_response = str((response or {}).get("raw_response", ""))
            last_expr = expr
            last_check = verify_safe_expression(expr, candidate.get("answer", ""), candidate.get("question", ""))
            if last_check["safe"]:
                fixed = {
                    **candidate,
                    "expression": expr,
                    "raw_response": raw_response,
                    "attempts": attempt,
                }
                repaired_item = _safe_record(fixed, last_check, source="expr_api_repair")
                safe.append(repaired_item)
                repaired.append(repaired_item)
                stats["repaired"] += 1
                break
            candidate = {**candidate, "expression": expr, "error": last_check["error"], "policy_reasons": last_check["policy_reasons"]}
        else:
            rejected.append(_rejected_record(candidate, last_expr, last_check, raw_response, max_attempts))
            stats["rejected"] += 1

        if checkpoint_every > 0 and processed % checkpoint_every == 0:
            write_checkpoint(processed, complete=False)

    write_checkpoint(total_records, complete=True)
    return safe


def _build_repair_report(
    raw_path: str,
    expr_path: str,
    safe: list[dict],
    repaired: list[dict],
    rejected: list[dict],
    stats: Counter[str],
    cache_stats: Counter[str],
    max_attempts: int,
    processed: int,
    total: int,
    complete: bool,
) -> list[dict]:
    return [{
        "input": raw_path,
        "existing": expr_path,
        "safe": len(safe),
        "repaired": len(repaired),
        "rejected": len(rejected),
        "stats": dict(stats),
        "cache": dict(cache_stats),
        "max_attempts": max_attempts,
        "processed": processed,
        "total": total,
        "complete": complete,
        "safety_contract": "pure_arithmetic_eval_then_question_normalization",
    }]


def _write_repair_outputs(
    safe_output: str,
    repaired_output: str,
    rejected_output: str,
    report_output: str,
    sft_output: str,
    safe: list[dict],
    repaired: list[dict],
    rejected: list[dict],
    report: list[dict],
) -> None:
    _save_json_records(safe_output, safe)
    _save_json_records(repaired_output, repaired)
    _save_json_records(rejected_output, rejected)
    _save_json_records(report_output, report)
    if sft_output:
        _save_json_records(sft_output, _to_sft_records(safe))


def _to_sft_records(records: list[dict]) -> list[dict]:
    return [
        {
            "id": item["id"],
            "question": item["question"],
            "expression": item["expression"],
            "answer": item["answer"],
        }
        for item in records
    ]


def verify_safe_expression(expr: str, gold_answer: str, question: Any) -> dict:
    policy = validate_expression(expr)
    if not policy.ok:
        return {
            "safe": False,
            "eval_result": None,
            "error": "policy_violation",
            "policy_reasons": list(policy.reasons),
        }
    try:
        value = safe_eval_expression(policy.normalized)
    except Exception as exc:
        return {
            "safe": False,
            "eval_result": None,
            "error": f"eval_failed: {exc}",
            "policy_reasons": [],
        }

    raw_answer = format_auto(value)
    formatted = normalize_answer_for_question(raw_answer, question)
    if answers_match_by_question(raw_answer, str(gold_answer), question):
        return {
            "safe": True,
            "eval_result": formatted,
            "raw_eval_result": raw_answer,
            "error": None,
            "policy_reasons": [],
        }
    return {
        "safe": False,
        "eval_result": formatted,
        "raw_eval_result": raw_answer,
        "error": "answer_mismatch",
        "policy_reasons": [],
    }


def _candidate_from(raw_item: dict, existing: dict | None) -> dict:
    existing = existing or {}
    question = normalize_question_text(raw_item.get("question", existing.get("question", "")))
    return {
        "id": str(raw_item.get("id", existing.get("id", ""))),
        "source_id": str(raw_item.get("id", existing.get("id", ""))),
        "source": existing.get("source", "expr_existing"),
        "question": question,
        "answer": str(raw_item.get("answer", existing.get("answer", ""))),
        "expression": str(existing.get("expression", "")),
        "error": str(existing.get("error", "")),
        "original_status": existing.get("status", ""),
        "original_valid": existing.get("valid", None),
    }


def _safe_record(candidate: dict, check: dict, source: str) -> dict:
    return {
        "id": str(candidate["id"]),
        "source_id": str(candidate.get("source_id", candidate["id"])),
        "source": source,
        "question": candidate["question"],
        "answer": str(candidate["answer"]),
        "expression": str(candidate["expression"]),
        "eval_result": check["eval_result"],
        "raw_eval_result": check.get("raw_eval_result"),
        "valid": True,
        "status": "ok",
        "attempts": candidate.get("attempts", 0),
    }


def _rejected_record(candidate: dict, expr: str, check: dict, raw_response: str, attempts: int) -> dict:
    return {
        "id": str(candidate.get("id", "")),
        "source_id": str(candidate.get("source_id", candidate.get("id", ""))),
        "question": candidate.get("question", ""),
        "answer": str(candidate.get("answer", "")),
        "expression": expr,
        "eval_result": check.get("eval_result"),
        "valid": False,
        "status": "rejected",
        "reject_reason": check.get("error") or "unknown",
        "policy_reasons": check.get("policy_reasons", []),
        "raw_response": raw_response,
        "attempts": attempts,
    }


def _build_deepseek_request(api_key: str, model: str) -> RepairRequestFn:
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required for real API repair")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

    def request(item: dict, attempt: int, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
        for retry in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=256,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                usage = getattr(resp, "usage", None)
                content = resp.choices[0].message.content
                return {
                    "expression": _extract_expr_text(content),
                    "raw_response": content,
                    "prompt_cache_hit_tokens": _usage_value(usage, "prompt_cache_hit_tokens"),
                    "prompt_cache_miss_tokens": _usage_value(usage, "prompt_cache_miss_tokens"),
                }
            except Exception as exc:
                logger.warning("表达式修复 API 失败 id=%s attempt=%s retry=%s: %s", item.get("id"), attempt, retry + 1, exc)
                time.sleep(2 ** retry)
        return None

    return request


def _extract_expression_response(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    expr = response.get("expression")
    if expr:
        return str(expr).strip()
    return _extract_expr_text(str(response.get("raw_response", "")))


def _extract_expr_text(text: str) -> str:
    match = re.search(r"<expr>\s*(.*?)\s*</expr>", text or "", re.DOTALL)
    if match:
        return match.group(1).strip()
    return str(text or "").strip()


def _usage_value(usage: Any, key: str) -> int:
    if isinstance(usage, dict):
        return int(usage.get(key, 0) or 0)
    return int(getattr(usage, key, 0) or 0)


def _accumulate_cache_stats(cache_stats: Counter[str], response: dict[str, Any] | None) -> None:
    if not response:
        return
    cache_stats["requests"] += 1
    cache_stats["prompt_cache_hit_tokens"] += int(response.get("prompt_cache_hit_tokens", 0) or 0)
    cache_stats["prompt_cache_miss_tokens"] += int(response.get("prompt_cache_miss_tokens", 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="幂等修复不规范表达式数据")
    parser.add_argument("--raw", default="data/raw/train.json")
    parser.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    parser.add_argument("--safe", default="data/processed/intermediate/expression_repair/expr_correct_safe.json")
    parser.add_argument("--repaired", default="data/processed/intermediate/expression_repair/expr_repaired.json")
    parser.add_argument("--rejected", default="data/processed/intermediate/expression_repair/expr_rejected.json")
    parser.add_argument("--report", default="data/processed/intermediate/expression_repair/expr_repair_report.json")
    parser.add_argument("--sft", default="data/processed/train_expr_safe.json")
    parser.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--no_api", action="store_true", help="只生成当前安全数据与拒绝报告，不请求外部 API")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    repair_expression_records(
        raw_path=args.raw,
        expr_path=args.expr,
        safe_output=args.safe,
        repaired_output=args.repaired,
        rejected_output=args.rejected,
        report_output=args.report,
        sft_output=args.sft,
        request_fn=(lambda item, attempt, messages: None) if args.no_api else None,
        api_key=args.api_key,
        model=args.model,
        max_attempts=0 if args.no_api else args.max_attempts,
        limit=args.limit,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
