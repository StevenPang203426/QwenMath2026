"""Generate report figures from existing experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"


def load_json(relative_path: str):
    path = ROOT / relative_path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(path)


def annotate_bars(values: list[float], fmt: str = "{:.1f}") -> None:
    ax = plt.gca()
    ymax = max(values) if values else 1
    for index, value in enumerate(values):
        ax.text(index, value + ymax * 0.015, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_data_source_composition() -> None:
    report = load_json("data/processed/train_clean_augmented_report.json")[0]
    counts = report["source_counts"]
    labels = ["raw_ok", "auto_repair", "expr_format_repair", "rule_augment"]
    values = [counts[label] for label in labels]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    plt.figure(figsize=(8.5, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Records")
    plt.title("Clean Augmented Training Data Composition")
    plt.xticks(rotation=18, ha="right")
    annotate_bars(values, "{:.0f}")
    savefig("data_source_composition.png")


def plot_cot_prompt_ablation() -> None:
    report = load_json("outputs/evaluation/cot_prompt_ablation_fixed_base/ablation_report.json")
    metrics = report["validation"]["metrics"]
    entries = [
        ("SFT direct", metrics["sft_cot"]["direct"]["accuracy"]),
        ("SFT zero-shot", metrics["sft_cot"]["zero_shot_cot"]["accuracy"]),
        ("SFT few-shot", metrics["sft_cot"]["few_shot_cot"]["accuracy"]),
        ("DPO few-shot", metrics["dpo"]["few_shot_cot"]["accuracy"]),
        ("GRPO zero-shot", metrics["grpo"]["zero_shot_cot"]["accuracy"]),
        ("GRPO few-shot", metrics["grpo"]["few_shot_cot"]["accuracy"]),
    ]
    labels = [item[0] for item in entries]
    values = [item[1] * 100 for item in entries]

    plt.figure(figsize=(9.2, 4.8))
    plt.bar(labels, values, color=["#4C78A8", "#72B7B2", "#54A24B", "#E45756", "#F58518", "#B279A2"])
    plt.ylabel("Validation accuracy (%)")
    plt.title("CoT Prompt Ablation (Fixed Base)")
    plt.ylim(0, max(values) * 1.18)
    plt.xticks(rotation=22, ha="right")
    annotate_bars(values, "{:.1f}")
    savefig("cot_prompt_ablation.png")


def plot_expression_dpo_ablation() -> None:
    report = load_json("outputs/evaluation/expr_dpo_ablation/ablation_report.json")
    models = report["validation"]["models"]
    entries = [
        ("SFT->GRPO", models["grpo_expr_clean_vllm"]["accuracy"], models["grpo_expr_clean_vllm"]["safe_expression_coverage"]),
        (
            "SFT->DPO->GRPO",
            models["grpo_expr_from_dpo_clean_vllm"]["accuracy"],
            models["grpo_expr_from_dpo_clean_vllm"]["safe_expression_coverage"],
        ),
    ]
    labels = [item[0] for item in entries]
    accuracy = [item[1] * 100 for item in entries]
    coverage = [item[2] * 100 for item in entries]
    x_positions = range(len(labels))
    width = 0.36

    plt.figure(figsize=(7.4, 4.8))
    ax = plt.gca()
    ax.bar([x - width / 2 for x in x_positions], accuracy, width, label="Accuracy", color="#4C78A8")
    ax.bar([x + width / 2 for x in x_positions], coverage, width, label="Safe coverage", color="#F58518")
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Percent (%)")
    ax.set_title("Expression DPO Ablation")
    ax.legend()
    ymax = max(accuracy + coverage)
    ax.set_ylim(0, ymax * 1.18)
    for index, value in enumerate(accuracy):
        ax.text(index - width / 2, value + ymax * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    for index, value in enumerate(coverage):
        ax.text(index + width / 2, value + ymax * 0.015, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    savefig("expression_dpo_ablation.png")


def plot_model_accuracy_comparison() -> None:
    cot = load_json("outputs/evaluation/cot_prompt_ablation_fixed_base/ablation_report.json")["validation"]["metrics"]
    smoke = load_json("outputs/evaluation/grpo_cot_reward_balanced_smoke/ablation_report.json")["validation"]["metrics"]
    expr4 = load_json("outputs/evaluation/expr4/val_report.json")
    expr_single = expr4["single_model_metrics"]
    entries = [
        ("CoT SFT\ndirect", cot["sft_cot"]["direct"]["accuracy"]),
        ("CoT DPO\nfew-shot", cot["dpo"]["few_shot_cot"]["accuracy"]),
        ("Old CoT GRPO\nzero-shot", cot["grpo"]["zero_shot_cot"]["accuracy"]),
        ("Balanced GRPO\nsmoke", smoke["grpo"]["direct"]["accuracy"]),
        ("Expr SFT\nclean", expr_single["sft_expr_clean"]["accuracy"]),
        ("Expr vote\nmajority", expr4["expression_vote_metrics"]["accuracy"]),
        ("Expr4\nselected", expr4["selection"]["accuracy"]),
    ]
    labels = [item[0] for item in entries]
    values = [item[1] * 100 for item in entries]

    plt.figure(figsize=(10, 5.2))
    colors = ["#4C78A8", "#E45756", "#F58518", "#54A24B", "#72B7B2", "#B279A2", "#FF9DA6"]
    plt.bar(labels, values, color=colors)
    plt.ylabel("Validation accuracy (%)")
    plt.title("Key Validation Results")
    plt.ylim(0, max(values) * 1.15)
    annotate_bars(values, "{:.1f}")
    savefig("model_accuracy_comparison.png")


def plot_vote_source_distribution() -> None:
    report = load_json("outputs/submissions/expr4_submit_report.json")
    counts = report["source_counts"]
    labels = ["expression_cot_calibrated", "expression_majority", "cot_fallback"]
    values = [counts.get(label, 0) for label in labels]
    colors = ["#4C78A8", "#54A24B", "#F58518"]

    plt.figure(figsize=(8.8, 4.8))
    plt.bar(labels, values, color=colors)
    plt.ylabel("Test rows")
    plt.title("Final Test Submission Source Distribution")
    plt.xticks(rotation=18, ha="right")
    annotate_bars(values, "{:.0f}")
    savefig("vote_source_distribution.png")


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    plot_data_source_composition()
    plot_cot_prompt_ablation()
    plot_expression_dpo_ablation()
    plot_model_accuracy_comparison()
    plot_vote_source_distribution()


if __name__ == "__main__":
    main()
