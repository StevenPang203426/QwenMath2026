"""Offline weighted vote over CoT prompt ablation detail files."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.answer_normalizer import answers_match_by_question, normalize_answer_for_question, normalize_question_text

DEFAULT_DETAILS_DIR = "outputs/evaluation/cot_prompt_ablation_fixed_base"
DEFAULT_OUTPUT_DIR = "outputs/evaluation/cot_prompt_ensemble_offline"
DEFAULT_CANDIDATES = "sft_cot:direct=1.0,grpo:zero_shot_cot=0.35,grpo:few_shot_cot=0.30,dpo:few_shot_cot=0.0"
SFT_DIRECT_BASELINE = 0.7376
DEFAULT_GATE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class CandidateSpec:
    model: str
    prompt: str
    weight: float

    @property
    def label(self) -> str:
        return f"{self.model}:{self.prompt}"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_candidate_specs(raw: str) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk or ":" not in chunk.split("=", 1)[0]:
            raise ValueError(f"Invalid candidate spec: {chunk}; expected model:prompt=weight")
        left, weight_raw = chunk.split("=", 1)
        model, prompt = left.split(":", 1)
        specs.append(CandidateSpec(model=model.strip(), prompt=prompt.strip(), weight=float(weight_raw)))
    if not specs:
        raise ValueError("At least one candidate is required")
    return specs


def detail_path(details_dir: str | Path, prefix: str, spec: CandidateSpec) -> Path:
    return Path(details_dir) / f"{prefix}_{spec.model}_{spec.prompt}_details.json"


def load_details(details_dir: str | Path, prefix: str, specs: list[CandidateSpec]) -> dict[str, dict[str, dict[str, Any]]]:
    details_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        path = detail_path(details_dir, prefix, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing ablation details for {spec.label}: {path}")
        rows = load_json(path)
        details_by_label[spec.label] = {str(item.get("id", "")): item for item in rows}
    return details_by_label


def vote_key(answer: Any, question: Any) -> str:
    try:
        return normalize_answer_for_question(str(answer), question)
    except Exception:
        return str(answer).strip()


def weighted_vote_for_record(
    record: dict[str, Any],
    details_by_label: dict[str, dict[str, dict[str, Any]]],
    specs: list[CandidateSpec],
) -> dict[str, Any]:
    item_id = str(record.get("id", ""))
    question = record.get("question", "")
    groups: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []

    for order, spec in enumerate(specs):
        if spec.weight <= 0:
            continue
        detail = details_by_label.get(spec.label, {}).get(item_id)
        if detail is None:
            continue
        answer = str(detail.get("answer", ""))
        key = vote_key(answer, question)
        candidate = {
            "source": spec.label,
            "answer": answer,
            "vote_key": key,
            "weight": spec.weight,
            "order": order,
        }
        candidates.append(candidate)
        group = groups.setdefault(key, {
            "vote_key": key,
            "weight_sum": 0.0,
            "votes": 0,
            "best_weight": -1.0,
            "first_order": order,
            "answers": [],
        })
        group["weight_sum"] += spec.weight
        group["votes"] += 1
        group["best_weight"] = max(group["best_weight"], spec.weight)
        group["first_order"] = min(group["first_order"], order)
        group["answers"].append(candidate)

    if not groups:
        return {
            "id": item_id,
            "answer": "0",
            "winner_key": "0",
            "selected_source": "",
            "winner_sources": [],
            "candidates": candidates,
        }

    winner = sorted(
        groups.values(),
        key=lambda item: (item["weight_sum"], item["votes"], item["best_weight"], -item["first_order"]),
        reverse=True,
    )[0]
    best_answer = sorted(winner["answers"], key=lambda item: (item["weight"], -item["order"]), reverse=True)[0]
    return {
        "id": item_id,
        "answer": best_answer["answer"],
        "winner_key": winner["vote_key"],
        "selected_source": best_answer["source"],
        "winner_sources": [item["source"] for item in winner["answers"]],
        "candidates": candidates,
    }


def evaluate_weighted_vote(
    records: list[dict[str, Any]],
    details_by_label: dict[str, dict[str, dict[str, Any]]],
    specs: list[CandidateSpec],
) -> dict[str, Any]:
    predictions = []
    correct = 0
    oracle_correct = 0
    source_usage: Counter[str] = Counter()

    for record in records:
        vote = weighted_vote_for_record(record, details_by_label, specs)
        question = record.get("question", "")
        gold = record.get("answer", "")
        is_correct = answers_match_by_question(vote["answer"], gold, question)
        correct += int(is_correct)
        if vote["selected_source"]:
            source_usage.update([vote["selected_source"]])

        candidate_correct = False
        for candidate in vote["candidates"]:
            if answers_match_by_question(candidate["answer"], gold, question):
                candidate_correct = True
                break
        oracle_correct += int(candidate_correct)

        predictions.append({
            "id": str(record.get("id", "")),
            "question": normalize_question_text(question),
            "answer": vote["answer"],
            "gold_answer": str(gold),
            "correct": is_correct,
            "winner_key": vote["winner_key"],
            "selected_source": vote["selected_source"],
            "winner_sources": ";".join(vote["winner_sources"]),
        })

    total = len(records)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "oracle_accuracy": oracle_correct / total if total else 0.0,
        "oracle_correct": oracle_correct,
        "source_usage": dict(source_usage),
        "predictions": predictions,
    }


def write_predictions_csv(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "answer", "gold_answer", "correct", "winner_key", "selected_source", "winner_sources", "question"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def write_summary_markdown(path: str | Path, report: dict[str, Any]) -> None:
    lines = [
        "# Offline CoT Weighted Vote",
        "",
        "This report reads cached CoT prompt-ablation details and does not run model inference.",
        "",
        "## Validation",
        "",
        f"- Accuracy: {report['validation']['accuracy']:.4f} ({report['validation']['correct']} / {report['validation']['total']})",
        f"- Oracle accuracy: {report['validation']['oracle_accuracy']:.4f} ({report['validation']['oracle_correct']} / {report['validation']['total']})",
        f"- Baseline gate (`sft_cot:direct`): {report['baseline_gate']:.4f}",
        f"- Gate tolerance: {report['gate_tolerance']:.6f}",
        f"- Passes gate: {report['passes_baseline_gate']}",
        "",
        "## Candidates",
        "",
        "| Candidate | Weight |",
        "|---|---:|",
    ]
    for item in report["candidates"]:
        lines.append(f"| {item['label']} | {item['weight']:.3f} |")
    lines.extend([
        "",
        "## Selected Source Usage",
        "",
        "| Source | Selected count |",
        "|---|---:|",
    ])
    for source, count in sorted(report["validation"]["source_usage"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {source} | {count} |")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_offline_vote(args: argparse.Namespace) -> dict[str, Any]:
    specs = parse_candidate_specs(args.candidates)
    records = load_json(args.val)
    if args.limit and args.limit > 0:
        records = records[:args.limit]
    details_by_label = load_details(args.details_dir, args.prefix, specs)
    validation = evaluate_weighted_vote(records, details_by_label, specs)
    report = {
        "scope": "cot_prompt_ablation_offline_weighted_vote",
        "details_dir": args.details_dir,
        "val": args.val,
        "prefix": args.prefix,
        "limit": args.limit,
        "baseline_gate": args.baseline_gate,
        "gate_tolerance": args.gate_tolerance,
        "passes_baseline_gate": validation["accuracy"] + args.gate_tolerance >= args.baseline_gate,
        "candidates": [{"label": spec.label, "weight": spec.weight} for spec in specs],
        "validation": {key: value for key, value in validation.items() if key != "predictions"},
    }

    output_dir = Path(args.output_dir)
    write_json(output_dir / "ensemble_report.json", report)
    write_summary_markdown(output_dir / "ensemble_summary.md", report)
    write_predictions_csv(output_dir / "ensemble_predictions.csv", validation["predictions"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline weighted vote over CoT prompt-ablation details")
    parser.add_argument("--val", default="data/splits/train_expr_clean_val.json")
    parser.add_argument("--details_dir", default=DEFAULT_DETAILS_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="val")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--baseline_gate", type=float, default=SFT_DIRECT_BASELINE)
    parser.add_argument("--gate_tolerance", type=float, default=DEFAULT_GATE_TOLERANCE)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_offline_vote(args)


if __name__ == "__main__":
    main()
