"""
CSV/JSON entrypoint for expression-first voting.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.inference.expression_first_voter import vote_expression_first
from src.utils.answer_normalizer import normalize_question_text


def run_expression_first_vote(
    test_path: str,
    output_path: str,
    expr_specs: list[str],
    cot_specs: list[str],
    report_path: str = "",
) -> str:
    questions = _load_questions(test_path)
    expr_by_id = _load_expr_candidates(expr_specs)
    cot_by_id = _load_cot_candidates(cot_specs)

    rows = []
    report = []
    all_ids = sorted(questions.keys(), key=lambda item: int(item) if str(item).isdigit() else str(item))
    for item_id in all_ids:
        result = vote_expression_first(
            question=questions[item_id],
            expr_candidates=expr_by_id.get(item_id, []),
            cot_candidates=cot_by_id.get(item_id, []),
        )
        rows.append([item_id, result["answer"]])
        report.append({
            "id": item_id,
            "answer": result["answer"],
            "source": result["source"],
            "safe_expression_candidates": result["safe_expression_candidates"],
            "rejected_expression_candidates": result["rejected_expression_candidates"],
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return output_path


def _load_questions(test_path: str) -> dict[str, str]:
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(item["id"]): normalize_question_text(item.get("question", "")) for item in data}


def _load_expr_candidates(specs: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        path, source, weight = _parse_spec(spec)
        if not Path(path).exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        for item in records:
            item_id = str(item.get("id", ""))
            if not item_id:
                continue
            result.setdefault(item_id, []).append({
                "source": source,
                "weight": weight,
                "expression": str(item.get("expression", "")),
            })
    return result


def _load_cot_candidates(specs: list[str]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for spec in specs:
        path, source, weight = _parse_spec(spec)
        if not Path(path).exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                result.setdefault(str(row[0]), []).append({
                    "source": source,
                    "weight": weight,
                    "answer": row[1],
                })
    return result


def _parse_spec(spec: str) -> tuple[str, str, float]:
    parts = spec.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"spec must be path:source:weight, got {spec!r}")
    return parts[0], parts[1], float(parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="表达式优先投票融合")
    parser.add_argument("--test", default="data/raw/test.json")
    parser.add_argument("--output", default="outputs/submissions/submit_voted.csv")
    parser.add_argument("--report", default="outputs/submissions/submit_voted_report.json")
    parser.add_argument("--expr", action="append", default=[], help="path:source:weight")
    parser.add_argument("--cot", action="append", default=[], help="path:source:weight")
    args = parser.parse_args()
    run_expression_first_vote(args.test, args.output, args.expr, args.cot, args.report)


if __name__ == "__main__":
    main()
