"""
Compare expression GRPO endpoints with and without a DPO initialization step.

The ablation keeps generation, expression safety, answer normalization, and
temperature budget fixed. The only intended variable is whether GRPO starts
from SFT directly or from the DPO adapter.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.inference.expr4_eval_submit import (
    TEMPERATURES,
    answer_close_by_question,
    answer_from_expression,
    answer_vote_key,
    details_complete_for_records,
    generate_expression_details,
    load_records,
    write_csv,
    write_json,
)
from src.utils.answer_normalizer import normalize_question_text

logger = logging.getLogger("math_solver.expr_grpo_ablation")

SINGLE_GRPO = {
    "name": "grpo_expr_clean_vllm",
    "adapter_path": "outputs/checkpoints/grpo_expr_clean_vllm/best",
    "label": "SFT->GRPO",
}
DPO_GRPO = {
    "name": "grpo_expr_from_dpo_clean_vllm",
    "adapter_path": "outputs/checkpoints/grpo_expr_from_dpo_clean_vllm/best",
    "label": "SFT->DPO->GRPO",
}
COMPARE_MODELS = [SINGLE_GRPO, DPO_GRPO]
MODEL_ALIASES = {
    SINGLE_GRPO["name"]: "single_grpo",
    DPO_GRPO["name"]: "dpo_grpo",
}


def load_complete_expression_details(
    records: list[dict[str, Any]],
    details_dir: str,
    prefix: str,
    models: list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for spec in models:
        model_name = spec["name"]
        path = Path(details_dir) / f"{prefix}_{model_name}_details.json"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            details = json.load(f)
        matching_details = filter_complete_details_for_records(details, records)
        if matching_details is not None:
            result[model_name] = matching_details
        else:
            logger.warning("候选明细不完整或与当前记录不匹配，将重跑: %s", path)
    return result


def filter_complete_details_for_records(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return details for records if a full cached file covers the requested subset."""
    expected_ids = {str(item.get("id", "")) for item in records}
    matching = [item for item in details if str(item.get("id", "")) in expected_ids]
    if details_complete_for_records(matching, records, len(TEMPERATURES)):
        return matching
    return None


def prepare_expression_details(
    records: list[dict[str, Any]],
    details_dir: str,
    prefix: str,
    batch_size: int,
    reuse_existing: bool = True,
    generate_missing: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    if reuse_existing:
        details_by_model = load_complete_expression_details(records, details_dir, prefix, COMPARE_MODELS)
    else:
        details_by_model = {}

    missing_specs = [spec for spec in COMPARE_MODELS if spec["name"] not in details_by_model]
    if missing_specs and not generate_missing:
        missing = ", ".join(spec["name"] for spec in missing_specs)
        raise FileNotFoundError(f"缺少完整候选明细: {missing}")

    if missing_specs:
        logger.info("生成缺失表达式候选: %s", ", ".join(spec["name"] for spec in missing_specs))
        generated = generate_expression_details(
            records=records,
            output_dir=details_dir,
            prefix=prefix,
            batch_size=batch_size,
            models=missing_specs,
        )
        details_by_model.update(generated)
    elif reuse_existing:
        logger.info("复用完整表达式候选: %s", ", ".join(sorted(details_by_model)))

    return details_by_model


def _safe_candidate(detail: dict[str, Any], question: str) -> dict[str, Any] | None:
    answer = answer_from_expression(detail.get("expression", ""), question)
    if answer is None:
        return None
    temperature = float(detail.get("temperature", min(TEMPERATURES)))
    return {
        "id": str(detail.get("id", "")),
        "answer": answer,
        "vote_key": answer_vote_key(answer, question),
        "expression": str(detail.get("expression", "")),
        "temperature": temperature,
        "source": str(detail.get("source", "")),
        "model": str(detail.get("model", "")),
    }


def vote_temperature_candidates(question: str, details: list[dict[str, Any]]) -> dict[str, Any]:
    safe_candidates = []
    rejected = []
    for detail in details:
        candidate = _safe_candidate(detail, question)
        if candidate is None:
            rejected.append(detail)
        else:
            safe_candidates.append(candidate)

    if not safe_candidates:
        return {
            "answer": "0",
            "source": "no_safe_expression",
            "safe_expression_candidates": 0,
            "rejected_expression_candidates": len(rejected),
            "fallback": True,
            "details": {
                "groups": [],
                "rejected_sources": [str(item.get("source", "")) for item in rejected],
            },
        }

    groups: dict[str, dict[str, Any]] = {}
    for candidate in safe_candidates:
        key = str(candidate["vote_key"])
        if key not in groups:
            groups[key] = {
                "key": key,
                "answer": candidate["answer"],
                "count": 0,
                "temperatures": [],
                "sources": [],
                "expressions": [],
            }
        groups[key]["count"] += 1
        groups[key]["temperatures"].append(candidate["temperature"])
        groups[key]["sources"].append(candidate["source"])
        groups[key]["expressions"].append(candidate["expression"])

    min_temperature = min(TEMPERATURES)

    def sort_key(group: dict[str, Any]) -> tuple[int, int, str]:
        has_min_temp = any(math.isclose(float(temp), min_temperature) for temp in group["temperatures"])
        return (-int(group["count"]), 0 if has_min_temp else 1, str(group["key"]))

    ordered = sorted(groups.values(), key=sort_key)
    winner = ordered[0]
    return {
        "answer": winner["answer"],
        "source": "expression_temperature_vote",
        "safe_expression_candidates": len(safe_candidates),
        "rejected_expression_candidates": len(rejected),
        "fallback": False,
        "details": {
            "winner": winner,
            "groups": ordered,
            "rejected_sources": [str(item.get("source", "")) for item in rejected],
        },
    }


def _details_by_id(details: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in details:
        item_id = str(item.get("id", ""))
        result.setdefault(item_id, []).append(item)
    return result


def predict_records(
    records: list[dict[str, Any]],
    model_name: str,
    details: list[dict[str, Any]],
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    grouped = _details_by_id(details)
    rows: list[list[str]] = []
    report: list[dict[str, Any]] = []
    for item in records:
        item_id = str(item["id"])
        question = normalize_question_text(item.get("question", ""))
        result = vote_temperature_candidates(question, grouped.get(item_id, []))
        rows.append([item_id, result["answer"]])
        report.append({
            "id": item_id,
            "model": model_name,
            "question": question,
            "answer": result["answer"],
            "source": result["source"],
            "safe_expression_candidates": result["safe_expression_candidates"],
            "rejected_expression_candidates": result["rejected_expression_candidates"],
            "fallback": result["fallback"],
            "details": result["details"],
        })
    return rows, report


def score_model_report(records: list[dict[str, Any]], report: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_id = {str(item["id"]): item for item in records}
    correct = 0
    source_counts: Counter[str] = Counter()
    no_safe = 0
    safe_candidates = 0
    rejected_candidates = 0

    for item in report:
        item_id = str(item["id"])
        gold_item = gold_by_id[item_id]
        if answer_close_by_question(item["answer"], gold_item.get("answer", ""), gold_item.get("question", "")):
            correct += 1
        source_counts[str(item.get("source", "unknown"))] += 1
        if item.get("fallback"):
            no_safe += 1
        safe_candidates += int(item.get("safe_expression_candidates", 0))
        rejected_candidates += int(item.get("rejected_expression_candidates", 0))

    total = len(report)
    expected_candidates = total * len(TEMPERATURES)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "no_safe_count": no_safe,
        "safe_expression_candidates": safe_candidates,
        "rejected_expression_candidates": rejected_candidates,
        "safe_expression_coverage": safe_candidates / expected_candidates if expected_candidates else 0.0,
        "source_counts": dict(source_counts),
    }


def build_disagreements(
    records: list[dict[str, Any]],
    single_report: list[dict[str, Any]],
    dpo_report: list[dict[str, Any]],
    has_gold: bool,
) -> list[dict[str, Any]]:
    records_by_id = {str(item["id"]): item for item in records}
    single_by_id = {str(item["id"]): item for item in single_report}
    dpo_by_id = {str(item["id"]): item for item in dpo_report}
    rows = []

    for item_id in sorted(records_by_id, key=record_id_sort_key):
        record = records_by_id[item_id]
        question = normalize_question_text(record.get("question", ""))
        single = single_by_id[item_id]
        dpo = dpo_by_id[item_id]
        single_answer = str(single["answer"])
        dpo_answer = str(dpo["answer"])
        if answer_close_by_question(single_answer, dpo_answer, question):
            continue

        row = {
            "id": item_id,
            "question": question,
            "single_grpo_answer": single_answer,
            "dpo_grpo_answer": dpo_answer,
            "single_grpo_source": single["source"],
            "dpo_grpo_source": dpo["source"],
        }
        if has_gold:
            gold = str(record.get("answer", ""))
            single_correct = answer_close_by_question(single_answer, gold, question)
            dpo_correct = answer_close_by_question(dpo_answer, gold, question)
            if dpo_correct and not single_correct:
                outcome = "dpo_win"
            elif single_correct and not dpo_correct:
                outcome = "single_grpo_win"
            elif dpo_correct and single_correct:
                outcome = "both_correct_disagree"
            else:
                outcome = "both_wrong"
            row.update({
                "gold_answer": gold,
                "single_grpo_correct": single_correct,
                "dpo_grpo_correct": dpo_correct,
                "outcome": outcome,
            })
        else:
            row["outcome"] = "unlabeled_disagreement"
        rows.append(row)
    return rows


def record_id_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def discordance_summary(disagreements: list[dict[str, Any]]) -> dict[str, Any]:
    dpo_wins = sum(1 for item in disagreements if item.get("outcome") == "dpo_win")
    single_wins = sum(1 for item in disagreements if item.get("outcome") == "single_grpo_win")
    both_wrong = sum(1 for item in disagreements if item.get("outcome") == "both_wrong")
    both_correct = sum(1 for item in disagreements if item.get("outcome") == "both_correct_disagree")
    discordant = dpo_wins + single_wins
    return {
        "dpo_wins": dpo_wins,
        "single_grpo_wins": single_wins,
        "net_dpo_wins": dpo_wins - single_wins,
        "both_wrong_disagreements": both_wrong,
        "both_correct_disagreements": both_correct,
        "discordant_with_one_correct": discordant,
        "exact_sign_test_p_value": exact_sign_test_p_value(dpo_wins, single_wins),
    }


def exact_sign_test_p_value(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = sum(math.comb(trials, k) for k in range(tail + 1)) / (2 ** trials)
    return min(1.0, 2 * probability)


def _write_dict_csv(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: str, report: dict[str, Any]) -> None:
    val = report["validation"]
    single = val["models"][SINGLE_GRPO["name"]]
    dpo = val["models"][DPO_GRPO["name"]]
    discordance = val["discordance"]
    lines = [
        "# Expression GRPO DPO Ablation",
        "",
        "## Model Chains",
        "",
        f"- single GRPO: `{SINGLE_GRPO['name']}` = {SINGLE_GRPO['label']}",
        f"- DPO -> GRPO: `{DPO_GRPO['name']}` = {DPO_GRPO['label']}",
        "",
        "All expression RL endpoints depend on expression SFT. This ablation keeps the GRPO recipe fixed and changes only whether DPO is used before GRPO.",
        "",
        "## Validation Result",
        "",
        "| Model | Accuracy | Correct / Total | Safe coverage | No-safe count |",
        "|---|---:|---:|---:|---:|",
        (
            f"| SFT->GRPO | {single['accuracy']:.4f} | "
            f"{single['correct']} / {single['total']} | "
            f"{single['safe_expression_coverage']:.4f} | {single['no_safe_count']} |"
        ),
        (
            f"| SFT->DPO->GRPO | {dpo['accuracy']:.4f} | "
            f"{dpo['correct']} / {dpo['total']} | "
            f"{dpo['safe_expression_coverage']:.4f} | {dpo['no_safe_count']} |"
        ),
        "",
        "## DPO Effect",
        "",
        f"- DPO wins: {discordance['dpo_wins']}",
        f"- single GRPO wins: {discordance['single_grpo_wins']}",
        f"- net DPO wins: {discordance['net_dpo_wins']}",
        f"- exact sign-test p-value: {discordance['exact_sign_test_p_value']:.6g}",
        "",
        "## Artifacts",
        "",
        "- `val_single_grpo.csv`, `val_dpo_grpo.csv`",
        "- `test_single_grpo.csv`, `test_dpo_grpo.csv`",
        "- `val_disagreements.csv`, `test_disagreements.csv`",
        "- `ablation_report.json`",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    val_limit = args.val_limit if args.val_limit is not None else args.limit
    test_limit = args.test_limit if args.test_limit is not None else args.limit
    val_records = load_records(args.val, val_limit)
    test_records = load_records(args.test, test_limit)

    reuse_existing = not args.no_reuse_existing
    generate_missing = not args.only_reuse_existing
    if args.no_reuse_existing and args.only_reuse_existing:
        raise ValueError("--no_reuse_existing 与 --only_reuse_existing 不能同时使用")

    val_details = prepare_expression_details(
        records=val_records,
        details_dir=args.val_details_dir,
        prefix="val",
        batch_size=args.expr_batch_size,
        reuse_existing=reuse_existing,
        generate_missing=generate_missing,
    )
    test_details = prepare_expression_details(
        records=test_records,
        details_dir=args.test_details_dir,
        prefix="test",
        batch_size=args.expr_batch_size,
        reuse_existing=reuse_existing,
        generate_missing=generate_missing,
    )

    val_rows: dict[str, list[list[str]]] = {}
    val_reports: dict[str, list[dict[str, Any]]] = {}
    test_rows: dict[str, list[list[str]]] = {}
    test_reports: dict[str, list[dict[str, Any]]] = {}

    for spec in COMPARE_MODELS:
        model_name = spec["name"]
        alias = MODEL_ALIASES[model_name]
        val_rows[model_name], val_reports[model_name] = predict_records(
            val_records,
            model_name,
            val_details[model_name],
        )
        test_rows[model_name], test_reports[model_name] = predict_records(
            test_records,
            model_name,
            test_details[model_name],
        )
        write_csv(str(output_dir / f"val_{alias}.csv"), val_rows[model_name])
        write_csv(str(output_dir / f"test_{alias}.csv"), test_rows[model_name])

    val_disagreements = build_disagreements(
        val_records,
        val_reports[SINGLE_GRPO["name"]],
        val_reports[DPO_GRPO["name"]],
        has_gold=True,
    )
    test_disagreements = build_disagreements(
        test_records,
        test_reports[SINGLE_GRPO["name"]],
        test_reports[DPO_GRPO["name"]],
        has_gold=False,
    )
    _write_dict_csv(str(output_dir / "val_disagreements.csv"), val_disagreements)
    _write_dict_csv(str(output_dir / "test_disagreements.csv"), test_disagreements)

    report = {
        "model_chains": {
            SINGLE_GRPO["name"]: SINGLE_GRPO["label"],
            DPO_GRPO["name"]: DPO_GRPO["label"],
        },
        "inputs": {
            "val": args.val,
            "test": args.test,
            "val_limit": val_limit,
            "test_limit": test_limit,
            "val_details_dir": args.val_details_dir,
            "test_details_dir": args.test_details_dir,
            "temperatures": TEMPERATURES,
            "aggregation": "safe_expression_temperature_vote",
        },
        "validation": {
            "models": {
                model: score_model_report(val_records, val_reports[model])
                for model in [SINGLE_GRPO["name"], DPO_GRPO["name"]]
            },
            "disagreement_count": len(val_disagreements),
            "discordance": discordance_summary(val_disagreements),
            "disagreements": val_disagreements,
        },
        "test": {
            "rows": len(test_records),
            "disagreement_count": len(test_disagreements),
            "models": {
                model: {
                    "rows": len(test_reports[model]),
                    "no_safe_count": sum(1 for item in test_reports[model] if item.get("fallback")),
                    "safe_expression_candidates": sum(
                        int(item.get("safe_expression_candidates", 0)) for item in test_reports[model]
                    ),
                    "rejected_expression_candidates": sum(
                        int(item.get("rejected_expression_candidates", 0)) for item in test_reports[model]
                    ),
                }
                for model in [SINGLE_GRPO["name"], DPO_GRPO["name"]]
            },
            "disagreements": test_disagreements,
        },
    }
    write_json(str(output_dir / "ablation_report.json"), report)
    _write_summary_markdown(str(output_dir / "ablation_summary.md"), report)
    logger.info("消融实验完成: %s", output_dir)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="表达式 GRPO DPO 消融实验")
    parser.add_argument("--val", default="data/splits/train_expr_clean_val.json")
    parser.add_argument("--test", default="data/raw/test.json")
    parser.add_argument("--val_details_dir", default="outputs/evaluation/expr4")
    parser.add_argument("--test_details_dir", default="outputs/submissions")
    parser.add_argument("--output_dir", default="outputs/evaluation/expr_dpo_ablation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    parser.add_argument("--expr_batch_size", type=int, default=16)
    parser.add_argument("--no_reuse_existing", action="store_true")
    parser.add_argument("--only_reuse_existing", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = build_parser().parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
