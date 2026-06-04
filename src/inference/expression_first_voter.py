"""
Expression-first voting for inference candidates.

Safe expression candidates form the base answer. CoT answers can verify ties or
serve as fallback, but they do not override a safe expression majority.
"""
from __future__ import annotations

from typing import Any

from src.inference.answer_postprocessor import postprocess_answer
from src.utils.answer_normalizer import format_auto, safe_eval_expression
from src.utils.expression_policy import validate_expression
from src.utils.metrics import normalize_number


def vote_expression_first(
    question: str,
    expr_candidates: list[dict[str, Any]],
    cot_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cot_candidates = cot_candidates or []
    safe_expr = []
    rejected_expr = []

    for candidate in expr_candidates:
        normalized = _answer_from_expression(candidate.get("expression", ""), question)
        if normalized is None:
            rejected_expr.append(candidate)
            continue
        safe_expr.append({
            **candidate,
            "answer": normalized,
            "vote_key": _vote_key(normalized),
            "weight": float(candidate.get("weight", 1.0)),
        })

    if safe_expr:
        winner = _expression_winner(safe_expr, cot_candidates, question)
        return {
            **winner,
            "safe_expression_candidates": len(safe_expr),
            "rejected_expression_candidates": len(rejected_expr),
        }

    fallback = _cot_fallback(cot_candidates, question)
    return {
        **fallback,
        "safe_expression_candidates": 0,
        "rejected_expression_candidates": len(rejected_expr),
    }


def _answer_from_expression(expression: Any, question: str) -> str | None:
    policy = validate_expression(expression)
    if not policy.ok:
        return None
    try:
        value = safe_eval_expression(policy.normalized)
    except Exception:
        return None
    return postprocess_answer(format_auto(value), question)


def _expression_winner(expr_candidates: list[dict[str, Any]], cot_candidates: list[dict[str, Any]], question: str) -> dict[str, Any]:
    groups = _group_candidates(expr_candidates)
    ordered = sorted(groups.values(), key=lambda group: (group["count"], group["weight"]), reverse=True)
    top = ordered[0]
    tied = [group for group in ordered if group["count"] == top["count"] and group["weight"] == top["weight"]]

    if len(tied) == 1:
        return {"answer": top["answer"], "source": "expression_majority", "details": top}

    cot_keys = {_vote_key(postprocess_answer(str(item.get("answer", "")), question)) for item in cot_candidates}
    for group in tied:
        if group["key"] in cot_keys:
            return {"answer": group["answer"], "source": "cot_verified_expression", "details": group}

    return {"answer": top["answer"], "source": "expression_tie", "details": top}


def _cot_fallback(cot_candidates: list[dict[str, Any]], question: str) -> dict[str, Any]:
    if not cot_candidates:
        return {"answer": "0", "source": "no_candidate", "details": {}}
    normalized = []
    for item in cot_candidates:
        answer = postprocess_answer(str(item.get("answer", "")), question)
        normalized.append({
            **item,
            "answer": answer,
            "vote_key": _vote_key(answer),
            "weight": float(item.get("weight", 1.0)),
        })
    winner = sorted(_group_candidates(normalized).values(), key=lambda group: (group["count"], group["weight"]), reverse=True)[0]
    return {"answer": winner["answer"], "source": "cot_fallback", "details": winner}


def _group_candidates(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["vote_key"]
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
        groups[key]["sources"].append(candidate.get("source", "unknown"))
    return groups


def _vote_key(answer: str) -> str:
    normalized = normalize_number(str(answer))
    return normalized if normalized is not None else str(answer).strip()
