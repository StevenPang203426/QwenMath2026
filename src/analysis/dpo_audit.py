"""Audit CoT DPO pair quality, truncation risk, and trainer states."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from src.data.answer_extractor import extract_answer
from src.training.rl_utils import render_chat_prompt, to_text
from src.utils.answer_normalizer import answers_match_by_question, normalize_question_text
from src.utils.config import load_config
from src.utils.metrics import normalize_number

DEFAULT_OUTPUT_DIR = "outputs/evaluation/dpo_audit"


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _stats(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "avg": 0}
    sorted_values = sorted(values)
    p95_idx = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
    return {
        "min": float(sorted_values[0]),
        "p50": float(statistics.median(sorted_values)),
        "p95": float(sorted_values[p95_idx]),
        "max": float(sorted_values[-1]),
        "avg": float(sum(sorted_values) / len(sorted_values)),
    }


def _same_normalized_answer(left: str, right: str) -> bool:
    left_norm = normalize_number(left)
    right_norm = normalize_number(right)
    if left_norm is not None and right_norm is not None:
        return left_norm == right_norm
    return str(left).strip() == str(right).strip()


def build_reference_by_id(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    raw_path = Path(path)
    if not raw_path.exists():
        return {}
    return {str(item.get("id", "")): item for item in load_json(raw_path)}


def _render_dpo_prompt(item: dict[str, Any], tokenizer: Any) -> str:
    instruction = to_text(item.get("instruction", "请一步一步思考，然后给出数字答案。"))
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": to_text(item.get("question", ""))},
    ]
    return render_chat_prompt(tokenizer, messages)


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(input_ids)


def audit_dpo_pairs(
    pairs: list[dict[str, Any]],
    reference_by_id: dict[str, dict[str, Any]] | None = None,
    tokenizer: Any | None = None,
    max_prompt_length: int = 256,
    max_length: int = 768,
) -> dict[str, Any]:
    reference_by_id = reference_by_id or {}
    chosen_chars: list[int] = []
    rejected_chars: list[int] = []
    prompt_chars: list[int] = []
    prompt_tokens: list[int] = []
    chosen_total_tokens: list[int] = []
    rejected_total_tokens: list[int] = []

    counts = {
        "missing_chosen_answer_tag": 0,
        "missing_rejected_answer_tag": 0,
        "same_extracted_answer": 0,
        "chosen_matches_reference": 0,
        "rejected_matches_reference": 0,
        "rejected_correct_chosen_wrong": 0,
        "prompt_over_max_prompt_length": 0,
        "chosen_over_max_length": 0,
        "rejected_over_max_length": 0,
    }
    risky_examples: list[dict[str, Any]] = []

    for item in pairs:
        item_id = str(item.get("id", ""))
        question = normalize_question_text(item.get("question", ""))
        chosen = to_text(item.get("chosen", ""))
        rejected = to_text(item.get("rejected", ""))
        chosen_answer = extract_answer(chosen)
        rejected_answer = extract_answer(rejected)
        chosen_chars.append(len(chosen))
        rejected_chars.append(len(rejected))
        prompt_chars.append(len(question))

        if "<answer>" not in chosen.lower() or "</answer>" not in chosen.lower():
            counts["missing_chosen_answer_tag"] += 1
        if "<answer>" not in rejected.lower() or "</answer>" not in rejected.lower():
            counts["missing_rejected_answer_tag"] += 1
        if _same_normalized_answer(chosen_answer, rejected_answer):
            counts["same_extracted_answer"] += 1
            if len(risky_examples) < 20:
                risky_examples.append({
                    "id": item_id,
                    "reason": "same_extracted_answer",
                    "chosen_answer": chosen_answer,
                    "rejected_answer": rejected_answer,
                    "question": question,
                })

        reference = reference_by_id.get(item_id)
        if reference is not None:
            gold = reference.get("answer", "")
            chosen_ok = answers_match_by_question(chosen_answer, gold, question)
            rejected_ok = answers_match_by_question(rejected_answer, gold, question)
            counts["chosen_matches_reference"] += int(chosen_ok)
            counts["rejected_matches_reference"] += int(rejected_ok)
            counts["rejected_correct_chosen_wrong"] += int(rejected_ok and not chosen_ok)

        if tokenizer is not None:
            prompt = _render_dpo_prompt(item, tokenizer)
            p_tokens = _token_count(tokenizer, prompt)
            c_total = p_tokens + _token_count(tokenizer, chosen)
            r_total = p_tokens + _token_count(tokenizer, rejected)
            prompt_tokens.append(p_tokens)
            chosen_total_tokens.append(c_total)
            rejected_total_tokens.append(r_total)
            counts["prompt_over_max_prompt_length"] += int(p_tokens > max_prompt_length)
            counts["chosen_over_max_length"] += int(c_total > max_length)
            counts["rejected_over_max_length"] += int(r_total > max_length)
            if (c_total > max_length or r_total > max_length) and len(risky_examples) < 20:
                risky_examples.append({
                    "id": item_id,
                    "reason": "token_truncation_risk",
                    "prompt_tokens": p_tokens,
                    "chosen_total_tokens": c_total,
                    "rejected_total_tokens": r_total,
                    "question": question,
                })

    total = len(pairs)
    return {
        "total_pairs": total,
        "counts": counts,
        "rates": {key: (value / total if total else 0.0) for key, value in counts.items()},
        "char_stats": {
            "prompt": _stats(prompt_chars),
            "chosen": _stats(chosen_chars),
            "rejected": _stats(rejected_chars),
        },
        "token_stats": {
            "enabled": tokenizer is not None,
            "prompt": _stats(prompt_tokens),
            "chosen_total": _stats(chosen_total_tokens),
            "rejected_total": _stats(rejected_total_tokens),
            "max_prompt_length": max_prompt_length,
            "max_length": max_length,
        },
        "risky_examples": risky_examples,
    }


def summarize_trainer_states(checkpoint_root: str | Path) -> list[dict[str, Any]]:
    root = Path(checkpoint_root)
    summaries: list[dict[str, Any]] = []
    for path in sorted(root.glob("checkpoint-*/trainer_state.json"), key=lambda p: int(p.parent.name.split("-")[-1])):
        state = load_json(path)
        logs = [item for item in state.get("log_history", []) if "loss" in item]
        losses = [float(item["loss"]) for item in logs]
        tail = logs[-5:]
        summaries.append({
            "checkpoint": str(path.parent),
            "global_step": state.get("global_step"),
            "epoch": state.get("epoch"),
            "log_rows": len(logs),
            "loss_min": min(losses) if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_tail_avg": (sum(losses[-5:]) / min(5, len(losses))) if losses else None,
            "learning_rate_last": tail[-1].get("learning_rate") if tail else None,
            "rewards_margin_last": tail[-1].get("rewards/margins") if tail else None,
            "mean_token_accuracy_last": tail[-1].get("mean_token_accuracy") if tail else None,
        })
    return summaries


def checkpoint_validation_commands(checkpoint_root: str | Path, output_root: str = "outputs/evaluation/dpo_checkpoint_ablation") -> list[str]:
    commands = []
    root = Path(checkpoint_root)
    for checkpoint in sorted(root.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        commands.append(
            f"OUTPUT_DIR={output_root}/{checkpoint.name} "
            "MODELS=dpo PROMPTS=few_shot_cot "
            f"bash scripts/run_cot_prompt_ablation.sh val --dpo_adapter_path {checkpoint}"
        )
    return commands


def write_summary_markdown(path: str | Path, report: dict[str, Any]) -> None:
    pair = report["pair_audit"]
    lines = [
        "# DPO Audit",
        "",
        "This audit does not retrain DPO. It checks pair quality, truncation risk, and trainer_state logs.",
        "",
        "## Pair Quality",
        "",
        f"- Total pairs: {pair['total_pairs']}",
        f"- Same extracted answer: {pair['counts']['same_extracted_answer']} ({pair['rates']['same_extracted_answer']:.2%})",
        f"- Missing chosen <answer>: {pair['counts']['missing_chosen_answer_tag']} ({pair['rates']['missing_chosen_answer_tag']:.2%})",
        f"- Missing rejected <answer>: {pair['counts']['missing_rejected_answer_tag']} ({pair['rates']['missing_rejected_answer_tag']:.2%})",
        f"- Rejected correct while chosen wrong: {pair['counts']['rejected_correct_chosen_wrong']} ({pair['rates']['rejected_correct_chosen_wrong']:.2%})",
        "",
        "## Truncation Risk",
        "",
        f"- Tokenizer enabled: {pair['token_stats']['enabled']}",
        f"- Prompt over max_prompt_length: {pair['counts']['prompt_over_max_prompt_length']} ({pair['rates']['prompt_over_max_prompt_length']:.2%})",
        f"- Chosen over max_length: {pair['counts']['chosen_over_max_length']} ({pair['rates']['chosen_over_max_length']:.2%})",
        f"- Rejected over max_length: {pair['counts']['rejected_over_max_length']} ({pair['rates']['rejected_over_max_length']:.2%})",
        "",
        "## Trainer States",
        "",
        "| Checkpoint | Step | Loss last | Loss tail avg | LR last | Margin last | Token acc last |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["trainer_states"]:
        lines.append(
            f"| {Path(item['checkpoint']).name} | {item['global_step']} | "
            f"{item['loss_last'] if item['loss_last'] is not None else ''} | "
            f"{item['loss_tail_avg'] if item['loss_tail_avg'] is not None else ''} | "
            f"{item['learning_rate_last'] if item['learning_rate_last'] is not None else ''} | "
            f"{item['rewards_margin_last'] if item['rewards_margin_last'] is not None else ''} | "
            f"{item['mean_token_accuracy_last'] if item['mean_token_accuracy_last'] is not None else ''} |"
        )
    lines.extend([
        "",
        "## Next Commands",
        "",
    ])
    lines.extend(f"- `{cmd}`" for cmd in report["checkpoint_validation_commands"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    pairs = load_json(args.data)
    reference_by_id = build_reference_by_id(args.reference_data)
    tokenizer = None
    if not args.skip_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model.name, use_fast=False, trust_remote_code=True)

    pair_audit = audit_dpo_pairs(
        pairs=pairs,
        reference_by_id=reference_by_id,
        tokenizer=tokenizer,
        max_prompt_length=int(getattr(config.dpo, "max_prompt_length", 256)),
        max_length=int(getattr(config.dpo, "max_length", 768)),
    )
    report = {
        "scope": "dpo_pair_and_training_audit",
        "config": args.config,
        "data": args.data,
        "reference_data": args.reference_data,
        "checkpoint_root": args.checkpoint_root,
        "dpo_config": {
            "beta": getattr(config.dpo, "beta", None),
            "loss_type": getattr(config.dpo, "loss_type", None),
            "max_prompt_length": getattr(config.dpo, "max_prompt_length", None),
            "max_length": getattr(config.dpo, "max_length", None),
            "learning_rate": getattr(config.training, "learning_rate", None),
            "num_train_epochs": getattr(config.training, "num_train_epochs", None),
        },
        "pair_audit": pair_audit,
        "trainer_states": summarize_trainer_states(args.checkpoint_root),
        "checkpoint_validation_commands": checkpoint_validation_commands(args.checkpoint_root),
    }
    output_dir = Path(args.output_dir)
    write_json(output_dir / "dpo_audit_report.json", report)
    write_summary_markdown(output_dir / "dpo_audit_summary.md", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit DPO pair quality, truncation risk, and trainer_state logs")
    parser.add_argument("--config", default="configs/dpo.yaml")
    parser.add_argument("--data", default="data/processed/train_dpo.json")
    parser.add_argument("--reference_data", default="data/raw/train.json")
    parser.add_argument("--checkpoint_root", default="outputs/checkpoints/dpo")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip_tokenizer", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
