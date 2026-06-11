# Evaluation Output Index

Generated evaluation artifacts live under this directory and are ignored by git except this index. Keep durable conclusions in `docs/issues/` or `docs/notes/`; use this file as a lightweight map to local result folders.

| Directory | Status | Summary | Notes |
|---|---|---|---|
| `cot_prompt_ablation/` | historical | `ablation_summary.md` | Old CoT ablation with incorrect RL adapter base; keep only for contrast. |
| `cot_prompt_ablation_fixed_base/` | active reference | `ablation_summary.md` | Fixed-base CoT ablation; current single-model reference is `sft_cot:direct`. |
| `cot_prompt_ensemble_offline/` | active experiment | `ensemble_summary.md` | Offline weighted vote from cached CoT details. |
| `dpo_audit/` | active audit | `dpo_audit_summary.md` | DPO pair/truncation/checkpoint audit output. |
| `expr4/` | active reference | `val_report.json` | Expression-first voting validation and selection output. |
| `expr_dpo_ablation/` | active experiment | `ablation_summary.md` | Expression DPO/GRPO ablation output. |
| `grpo_cot_reward_*` | experimental | `ablation_summary.md` | CoT GRPO reward smoke/full validation outputs. |

Do not commit generated CSV/JSON details from these folders.
