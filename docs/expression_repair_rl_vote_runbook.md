# 表达式修复、强化学习与投票运行说明

## 数据修复

安全表达式契约是：表达式只能包含数字、`+ - * / ** ( )` 和一元正负号；不允许方程、比较、if/else、函数、`%`、`//`、变量、中文单位或 LaTeX。表达式先 eval 得到原始值，再按题意归一成最终答案。

当前已执行无外部 API 的安全合并：

```bash
bash scripts/run_expr_repair.sh no_api
```

产物：

- `data/processed/intermediate/expression_repair/expr_correct_safe.json`
- `data/processed/intermediate/expression_repair/expr_rejected.json`
- `data/processed/intermediate/expression_repair/expr_repair_report.json`
- `data/processed/train_expr_safe.json`

当前 no-api 结果：安全样本 10702 条，rejected 1297 条，安全样本复验 0 违规。

真实 DeepSeek 修复会把 rejected 题目发送到外部 API。只有在明确接受外部数据披露风险后再运行：

```bash
export DEEPSEEK_API_KEY=your_key
bash scripts/run_expr_repair.sh
```

## 表达式实验矩阵

```bash
bash scripts/run_sft_expr_safe.sh
bash scripts/run_dpo_expr.sh
bash scripts/run_grpo_expr.sh
bash scripts/run_grpo_expr_from_dpo.sh
```

- `sft_expr_safe` 使用安全表达式数据冷启动。
- `dpo_expr` 使用表达式偏好对。
- `grpo_expr` 从安全 SFT 开始。
- `grpo_expr_from_dpo` 从 Expr-DPO checkpoint 开始。

## 表达式优先投票

```bash
bash scripts/run_infer_vote.sh
```

表达式模型每题生成多个候选；安全表达式候选是主答案来源，CoT 答案只用于验证、破局或兜底。最终输出：

- `outputs/submissions/submit_voted.csv`
- `outputs/submissions/submit_voted_report.json`
