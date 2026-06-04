"""
Data quality audit and high-confidence repair pipeline.

The module first performs cheap rule-based pre-screening, then optionally asks
an LLM to audit only high-risk records. Repairs are written only when the LLM
proposal passes deterministic expression verification.
"""
import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from src.data.expr_builder import (
    DEFAULT_MODEL,
    DEEPSEEK_BASE_URL,
    _answers_match,
    _load_json_records,
    _save_json_records,
    safe_eval,
)
from src.utils.answer_normalizer import answers_match_by_question, normalize_question_text

logger = logging.getLogger("math_solver.quality_auditor")

AUDIT_LABELS = {
    "ok",
    "ambiguous",
    "ocr_error",
    "missing_symbol",
    "wrong_answer",
    "unsolvable",
    "needs_human",
}
AUTO_REPAIR_LABELS = {"ocr_error", "missing_symbol", "wrong_answer"}

_AUDIT_SYSTEM_PROMPT = "你是小学数学题数据质量审计员。严格按用户消息中的固定规则审计，只输出 JSON。"

# DeepSeek context cache only matches persisted input prefixes. Keep the full
# rubric in the user message before the per-question JSON so consecutive audit
# calls share a long, identical prefix.
_AUDIT_CACHE_PREFIX = """
你正在审计 CCF BDCI 小学数学应用题训练数据。任务是判断题目是否清晰、标注答案是否可信、是否存在 OCR 或漏符号问题。

【输入说明】
每次请求最后都会给出一个候选题目 JSON，字段含义固定：
- id: 原始题目编号
- question: 题干，可能包含 OCR 残留、缺失符号或歧义表达
- answer: 当前标注答案，可能正确也可能错误
- signals: 规则预筛信号，例如 expr_correct_failed、cot_answer_disagreement、suspected_ocr_fraction、suspected_missing_percent、possible_ambiguity、empty_answer
- risk_score: 规则预筛风险分数，只作为参考，不要机械服从

【输出要求】
只输出一个 JSON 对象，不要输出 Markdown，不要输出解释性正文，不要包裹代码块。JSON 必须包含以下字段：
- label: ok | ambiguous | ocr_error | missing_symbol | wrong_answer | unsolvable | needs_human
- confidence: 0 到 1 的数字
- reason: 简短中文原因
- can_auto_repair: true 或 false
- repaired_question: 如果能高置信修复，给修复后的题干，否则空字符串
- repaired_answer: 如果能高置信修复，给修复后的答案，否则空字符串
- repair_expression: 如果能高置信修复，给 Python 可计算表达式，否则空字符串

【标签定义】
- ok: 题目清晰，标注答案可信，不需要修复
- ambiguous: 题意有歧义，存在多个合理解释
- ocr_error: 题干明显有 OCR 识别错误，例如分数线丢失、数字粘连、字符错识别
- missing_symbol: 题干明显漏了百分号、小数点、分数线、括号等关键符号
- wrong_answer: 题干清晰，但标注答案明显错误
- unsolvable: 信息不足、条件矛盾或无法唯一求解
- needs_human: 不能高置信判断，或修复会改变题目核心含义

【自动修复原则】
自动修复必须非常保守。只有明显 OCR 错、漏百分号/分数线/小数点、或标注答案明显错时才允许 can_auto_repair=true。
如果只是模型表达式生成失败，不能直接判定题目错误。
如果修复需要猜测题意，必须 label=needs_human 或 ambiguous，can_auto_repair=false。
如果题目可解但当前 signals 只是模型生成不稳定，通常 label=ok 或 needs_human。

【修复写回门槛】
当 can_auto_repair=true 时，必须同时给出 repaired_question、repaired_answer、repair_expression。
repair_expression 必须是 Python 可直接计算的纯数学表达式，只允许数字和 + - * / ** ( )，不要使用变量、中文、单位、函数或 LaTeX。
repair_expression 的计算结果必须与 repaired_answer 一致。
修复后的题干只能修正明显字符/符号/标注问题，不得更换题型或重新创作新题。

【判断细节】
百分率、合格率、成活率、发芽率、命中率等通常要求百分数答案。
几分之几、分率、占比、比例、比重等通常要求分数答案。
至少、最少、起码并搭配车辆、船、箱、人等离散量时，通常需要向上取整。
至多、最多、顶多并搭配能、可以、装、分、做、买等语义时，通常需要向下取整。
圆周率按 3.14 处理。单位不要写进 repaired_answer。

下面是候选题目 JSON。请只根据固定规则和该 JSON 审计：
""".strip()

_AUDIT_CANDIDATE_SEPARATOR = "\n\n【候选题目 JSON】\n"


def _get_client(api_key: str):
    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _load_json(path: str) -> list[dict]:
    if not Path(path).exists() and path == "data/processed/intermediate/quality/quality_audit.json":
        legacy = Path("data/processed/quality_audit.json")
        if legacy.exists():
            path = str(legacy)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _by_id(records: list[dict]) -> dict[str, dict]:
    return {str(item["id"]): item for item in records if "id" in item}


def _expr_is_valid(item: dict | None) -> bool:
    return bool(item and item.get("status") == "ok" and item.get("valid") is True)


def _looks_like_ocr_fraction(text: str) -> bool:
    # Conservative patterns: common fraction OCR collapses in elementary math.
    return bool(re.search(r'(?<!\d)(1|2|3|4|5|6|7|8|9)1(2|3|4|5|6|7|8|9)(?!\d)', text))


def _looks_like_missing_percent(text: str) -> bool:
    has_percent_context = any(key in text for key in ["百分率", "百分比", "成活率", "合格率", "命中率", "利率"])
    return has_percent_context and "%" not in text and "百分之" not in text


def _question_has_ambiguity_signal(text: str) -> bool:
    return any(key in text for key in ["几分之几", "比", "多", "少"]) and len(re.findall(r'\d+', text)) < 2


def _candidate_signals(item: dict, expr_item: dict | None, cot_item: dict | None) -> list[str]:
    question = normalize_question_text(item.get("question", ""))
    answer = str(item.get("answer", ""))
    signals = []

    if expr_item is not None and not _expr_is_valid(expr_item):
        signals.append("expr_correct_failed")
    if cot_item:
        api_answer = cot_item.get("api_answer")
        if api_answer and not _answers_match(api_answer, answer):
            signals.append("cot_answer_disagreement")
    if expr_item and expr_item.get("eval_result") and not answers_match_by_question(expr_item["eval_result"], answer, question):
        signals.append("expr_answer_disagreement")
    if _looks_like_ocr_fraction(question):
        signals.append("suspected_ocr_fraction")
    if _looks_like_missing_percent(question):
        signals.append("suspected_missing_percent")
    if _question_has_ambiguity_signal(question):
        signals.append("possible_ambiguity")
    if not answer.strip():
        signals.append("empty_answer")

    return signals


def _risk_score(signals: list[str]) -> float:
    weights = {
        "expr_correct_failed": 0.35,
        "cot_answer_disagreement": 0.45,
        "expr_answer_disagreement": 0.50,
        "suspected_ocr_fraction": 0.45,
        "suspected_missing_percent": 0.40,
        "possible_ambiguity": 0.25,
        "empty_answer": 0.80,
    }
    return min(1.0, sum(weights.get(signal, 0.2) for signal in signals))


def build_quality_candidates(
    input_path: str,
    output_path: str,
    expr_path: str = "data/processed/expr_correct.jsonl",
    cot_path: str = "data/processed/train_cot_raw.json",
    threshold: float = 0.25,
    limit: int = 0,
) -> list[dict]:
    data = _load_json(input_path)
    if limit > 0:
        data = data[:limit]

    expr_by_id = _by_id(_load_json_records(expr_path)) if os.path.exists(expr_path) else {}
    cot_by_id = _by_id(_load_json_records(cot_path)) if os.path.exists(cot_path) else {}

    candidates = []
    for item in data:
        item_id = str(item["id"])
        signals = _candidate_signals(item, expr_by_id.get(item_id), cot_by_id.get(item_id))
        score = _risk_score(signals)
        if score >= threshold:
            candidates.append({
                "id": item_id,
                "question": normalize_question_text(item.get("question", "")),
                "answer": str(item.get("answer", "")),
                "signals": signals,
                "risk_score": round(score, 4),
            })

    _save_json(output_path, candidates)
    logger.info(f"质量候选: {len(candidates)}/{len(data)} -> {output_path}")
    return candidates


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _candidate_payload(candidate: dict) -> dict:
    """Return the dynamic candidate payload kept after the cacheable prefix."""
    return {
        "id": candidate["id"],
        "question": candidate["question"],
        "answer": candidate["answer"],
        "signals": candidate.get("signals", []),
        "risk_score": candidate.get("risk_score", 0),
    }


def _build_audit_messages(candidate: dict) -> list[dict[str, Any]]:
    """Build a cache-friendly DeepSeek chat request for one audit candidate."""
    user_text = (
        _AUDIT_CACHE_PREFIX
        + _AUDIT_CANDIDATE_SEPARATOR
        + json.dumps(
            _candidate_payload(candidate),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return [
        {"role": "system", "content": _AUDIT_SYSTEM_PROMPT},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def _record_cache_usage(resp, cache_stats: dict[str, int] | None) -> None:
    """Accumulate DeepSeek prompt cache hit/miss tokens from a response."""
    if cache_stats is None:
        return
    usage = getattr(resp, "usage", None)
    if isinstance(usage, dict):
        hit = usage.get("prompt_cache_hit_tokens", 0) or 0
        miss = usage.get("prompt_cache_miss_tokens", 0) or 0
    else:
        hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        miss = getattr(usage, "prompt_cache_miss_tokens", 0) or 0
    cache_stats["requests"] = cache_stats.get("requests", 0) + 1
    cache_stats["hit_tokens"] = cache_stats.get("hit_tokens", 0) + int(hit)
    cache_stats["miss_tokens"] = cache_stats.get("miss_tokens", 0) + int(miss)


def _call_audit_api(
    candidate: dict,
    client,
    model: str,
    max_retries: int = 3,
    cache_stats: dict[str, int] | None = None,
) -> dict:
    messages = _build_audit_messages(candidate)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            _record_cache_usage(resp, cache_stats)
            result = _extract_json_object(resp.choices[0].message.content)
            result["id"] = str(candidate["id"])
            return _normalize_audit_result(result)
        except Exception as e:
            logger.warning(f"审计 API 失败 id={candidate['id']} ({attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)

    return {
        "id": str(candidate["id"]),
        "label": "needs_human",
        "confidence": 0.0,
        "reason": "audit_api_failed",
        "can_auto_repair": False,
        "repaired_question": "",
        "repaired_answer": "",
        "repair_expression": "",
    }


def _normalize_audit_result(result: dict) -> dict:
    label = str(result.get("label", "needs_human"))
    if label not in AUDIT_LABELS:
        label = "needs_human"
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return {
        "id": str(result.get("id", "")),
        "label": label,
        "confidence": confidence,
        "reason": str(result.get("reason", "")),
        "can_auto_repair": bool(result.get("can_auto_repair", False)),
        "repaired_question": str(result.get("repaired_question", "")),
        "repaired_answer": str(result.get("repaired_answer", "")),
        "repair_expression": str(result.get("repair_expression", "")),
    }


def audit_candidates(
    candidates_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    limit: int = 0,
) -> list[dict]:
    candidates = _load_json(candidates_path)
    if limit > 0:
        candidates = candidates[:limit]
    client = _get_client(api_key)

    audits = []
    cache_stats: dict[str, int] = {"requests": 0, "hit_tokens": 0, "miss_tokens": 0}
    for idx, candidate in enumerate(candidates, 1):
        audits.append(_call_audit_api(candidate, client, model, cache_stats=cache_stats))
        if idx % 50 == 0:
            _save_json(output_path, audits)
            hit = cache_stats["hit_tokens"]
            miss = cache_stats["miss_tokens"]
            total = hit + miss
            hit_rate = hit / total * 100 if total else 0.0
            logger.info(
                f"审计进度: {idx}/{len(candidates)} | "
                f"DeepSeek缓存 hit={hit}, miss={miss}, hit_rate={hit_rate:.1f}%"
            )

    _save_json(output_path, audits)
    logger.info(f"审计完成: {len(audits)} -> {output_path}")
    hit = cache_stats["hit_tokens"]
    miss = cache_stats["miss_tokens"]
    total = hit + miss
    hit_rate = hit / total * 100 if total else 0.0
    logger.info(
        f"DeepSeek缓存汇总: requests={cache_stats['requests']}, "
        f"hit={hit}, miss={miss}, hit_rate={hit_rate:.1f}%"
    )
    return audits


def _repair_is_valid(audit: dict) -> tuple[bool, str | None]:
    if audit.get("label") not in AUTO_REPAIR_LABELS:
        return False, "label_not_repairable"
    if not audit.get("can_auto_repair"):
        return False, "can_auto_repair_false"
    if float(audit.get("confidence", 0.0)) < 0.85:
        return False, "low_confidence"
    question = str(audit.get("repaired_question", "")).strip()
    answer = str(audit.get("repaired_answer", "")).strip()
    expr = str(audit.get("repair_expression", "")).strip()
    if not question or not answer or not expr:
        return False, "missing_repair_fields"
    try:
        value = safe_eval(expr.replace("×", "*").replace("÷", "/").replace("^", "**"))
    except Exception as e:
        return False, f"repair_eval_failed: {e}"
    value_str = str(int(value)) if value == int(value) else str(value)
    if not answers_match_by_question(value_str, answer, question):
        return False, f"repair_answer_mismatch: eval={value_str}, answer={answer}"
    return True, None


def build_repaired_data(audit_path: str, output_path: str) -> list[dict]:
    audits = _load_json(audit_path)
    repaired = []
    rejected = 0

    for audit in audits:
        is_valid, reason = _repair_is_valid(audit)
        if not is_valid:
            rejected += 1
            continue
        repaired.append({
            "id": f"repair_{audit['id']}",
            "source_id": str(audit["id"]),
            "source": "auto_repair",
            "question": normalize_question_text(audit["repaired_question"]),
            "answer": str(audit["repaired_answer"]),
            "expression": audit["repair_expression"],
            "audit_label": audit["label"],
            "audit_confidence": audit["confidence"],
            "repair_reason": audit.get("reason", reason or ""),
        })

    _save_json(output_path, repaired)
    logger.info(f"自动修复写回: {len(repaired)} 条，拒绝: {rejected} 条 -> {output_path}")
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description="数据质量审计与自动修复")
    sub = parser.add_subparsers(dest="action", required=True)

    cand = sub.add_parser("candidates", help="规则预筛质量候选")
    cand.add_argument("--input", default="data/raw/train.json")
    cand.add_argument("--output", default="data/processed/intermediate/quality/quality_candidates.json")
    cand.add_argument("--expr", default="data/processed/expr_correct.jsonl")
    cand.add_argument("--cot", default="data/processed/train_cot_raw.json")
    cand.add_argument("--threshold", type=float, default=0.25)
    cand.add_argument("--limit", type=int, default=0)

    audit = sub.add_parser("audit", help="调用大模型审计候选")
    audit.add_argument("--candidates", default="data/processed/intermediate/quality/quality_candidates.json")
    audit.add_argument("--output", default="data/processed/intermediate/quality/quality_audit.json")
    audit.add_argument("--api_key", default=os.environ.get("DEEPSEEK_API_KEY", ""))
    audit.add_argument("--model", default=DEFAULT_MODEL)
    audit.add_argument("--limit", type=int, default=0)

    repair = sub.add_parser("repair", help="根据审计结果生成自动修复数据")
    repair.add_argument("--audit", default="data/processed/intermediate/quality/quality_audit.json")
    repair.add_argument("--output", default="data/processed/train_repaired.json")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.action == "candidates":
        build_quality_candidates(args.input, args.output, args.expr, args.cot, args.threshold, args.limit)
    elif args.action == "audit":
        if not args.api_key:
            raise SystemExit("请提供 --api_key 或设置 DEEPSEEK_API_KEY")
        audit_candidates(args.candidates, args.output, args.api_key, args.model, args.limit)
    elif args.action == "repair":
        build_repaired_data(args.audit, args.output)


if __name__ == "__main__":
    main()
