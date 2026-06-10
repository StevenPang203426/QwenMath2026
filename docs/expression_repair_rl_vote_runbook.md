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

脚本默认每 100 条写一次 checkpoint，可通过环境变量调整：

```bash
CHECKPOINT_EVERY=50 bash scripts/run_expr_repair.sh
```

如果本机 `no_proxy` 环境变量含有异常字符，OpenAI/httpx 初始化可能报 `Invalid port`。本次运行通过覆盖干净代理例外列表解决：

```bash
env no_proxy='localhost,127.0.0.1,::1' NO_PROXY='localhost,127.0.0.1,::1' bash scripts/run_expr_repair.sh
```

本次真实 API 修复结果：原安全样本 10702 条，API 修复成功 245 条，最终安全样本 10947 条，剩余 rejected 1052 条；安全样本复验 0 违规。报告记录 API 请求 3420 次，prompt cache hit tokens 801664，miss tokens 333258。

本次数据产物 SHA256：

- `data/processed/intermediate/expression_repair/expr_correct_safe.json`: `b141b716ff76c8000c7988d917cb0a86d717ca9bd472e3dfff34df2e65dbdd95`
- `data/processed/intermediate/expression_repair/expr_repaired.json`: `4401aa1f3949a25b807f423c02b2b82e6112f6162f69489d493873c4deddc12c`
- `data/processed/intermediate/expression_repair/expr_rejected.json`: `9c1827e2150020eb67b8a3d1130ffcf5579806f9d0f3215c95e6233388379841`
- `data/processed/intermediate/expression_repair/expr_repair_report.json`: `f98522b1c7631691645c0ec65d5a386d01c10351332d08b2df4caf9146b9c45a`
- `data/processed/train_expr_safe.json`: `cd8df49bab3b67523890fdc0310a08480deee4623bf6b27917e5c6da05c0d502`

## 清洗合并结果

已执行：

```bash
bash scripts/run_clean_data_merge.sh
```

最终产物：

- `data/processed/train_clean_augmented.json`
- `data/processed/train_repairs_unified.json`
- `data/processed/train_clean_augmented_report.json`

当前合并总量 11891 条：

| 来源 | 含义 | 条数 | 占比 |
|------|------|------|------|
| `raw_ok` | 原始合格数据 | 10599 | 89.13% |
| `auto_repair` | 题意修复数据 | 347 | 2.92% |
| `expr_format_repair` | 表达式结果规范修复数据 | 279 | 2.35% |
| `rule_augment` | 增强后的合格数据 | 666 | 5.60% |

剩余 reject 不进入训练集，保留为人工抽样和后续规则补充材料；当前建议先使用合并后的 clean 数据进入训练阶段。

## 表达式训练数据

已执行：

```bash
bash scripts/run_expr_training_data.sh all
```

当前表达式 clean 数据来自 `data/processed/train_clean_augmented.json`，只保留通过规范表达式检测、可 eval，且题意归一后与标注答案一致的样本。

| 产物 | 路径 | 条数 |
|------|------|------|
| clean 表达式全集 | `data/processed/train_expr_clean.json` | 11478 |
| SFT/GRPO train split | `data/splits/train_expr_clean_train.json` | 10331 |
| SFT/GRPO val split | `data/splits/train_expr_clean_val.json` | 1147 |
| clean DPO pairs | `data/processed/train_expr_dpo_clean.json` | 9095 |

clean 表达式全集来源分布：

| 来源 | 条数 |
|------|------|
| `raw_ok` | 10214 |
| `rule_augment` | 665 |
| `auto_repair` | 320 |
| `expr_format_repair` | 279 |

过滤拒绝 413 条：`policy_violation` 256 条，`missing_expression` 101 条，`answer_mismatch` 56 条。这些样本不进入表达式路线训练。

## 表达式实验矩阵

```bash
bash scripts/run_sft_expr_clean.sh
bash scripts/run_grpo_expr_clean.sh
bash scripts/run_grpo_expr_clean_vllm.sh
bash scripts/run_grpo_expr_clean_fast.sh
bash scripts/run_dpo_expr_clean.sh
bash scripts/run_grpo_expr_from_dpo_clean.sh
bash scripts/run_grpo_expr_from_dpo_clean_vllm.sh
bash scripts/run_grpo_expr_from_dpo_clean_fast.sh
```

| 实验 | 初始化 | 配置 | 说明 |
|------|------|------|------|
| `sft_expr_clean` | base model | `configs/sft_expr_clean.yaml` | 表达式 SFT 主线 |
| `grpo_expr_clean` | `sft_expr_clean/best` | `configs/grpo_expr_clean.yaml` | GRPO 主线 |
| `grpo_expr_clean_vllm` | `sft_expr_clean/best` | `configs/grpo_expr_clean_vllm.yaml` | vLLM rollout 加速 |
| `grpo_expr_clean_fast` | `sft_expr_clean/best` | `configs/grpo_expr_clean_fast.yaml` | 不依赖 vLLM 的保守加速 |
| `dpo_expr_clean` | `sft_expr_clean/best` | `configs/dpo_expr_clean.yaml` | DPO 消融 |
| `grpo_expr_from_dpo_clean` | `dpo_expr_clean/best` | `configs/grpo_expr_from_dpo_clean.yaml` | DPO→GRPO 消融 |
| `grpo_expr_from_dpo_clean_vllm` | `dpo_expr_clean/best` | `configs/grpo_expr_from_dpo_clean_vllm.yaml` | DPO→GRPO vLLM 加速 |
| `grpo_expr_from_dpo_clean_fast` | `dpo_expr_clean/best` | `configs/grpo_expr_from_dpo_clean_fast.yaml` | DPO→GRPO 保守加速 |

GRPO clean 原始配置默认 `use_vllm: false`。当前环境的 TRL 声明支持 vLLM 0.12-0.18，本机 vLLM 是 0.21.0；`*_vllm.sh` 会给出警告但允许运行，因为已通过本机 1-step smoke。若需强制版本检查，可设置 `STRICT_VLLM_VERSION=1`。`*_fast` 不使用 vLLM，保留同样的 `num_generations=8`，将 `max_completion_length` 收紧到 96、`max_prompt_length` 收紧到覆盖全训练集的 192，关闭训练中 eval，并把保存间隔放宽到 500 step。

## Reward 口径

表达式 GRPO 的正确性只信任 `<expr>`：表达式通过安全策略后 eval，再按题意归一，与 gold answer 比较。`<answer>` 标签不兜底，只作为一致性约束。

当前 6 维 reward：

| 维度 | 范围 | 说明 |
|------|------|------|
| eval 正确性 | -0.6 ~ +1.0 | 题意归一后与 gold 匹配 |
| 可解析性 | -0.3 ~ +0.3 | 规范表达式且可 eval |
| 格式标签 | 0.0 ~ +0.2 | 同时有 `<expr>` 和 `<answer>` |
| 无非法字符 | -0.3 ~ 0.0 | 策略违规表达式惩罚 |
| answer 一致性 | -0.2 ~ +0.2 | `<answer>` 与表达式归一结果一致 |
| 输出洁净 | -0.2 ~ 0.0 | 标签外解释或超长输出惩罚 |

## 表达式优先投票

```bash
bash scripts/run_infer_vote.sh
```

表达式模型每题生成多个候选；安全表达式候选是主答案来源，CoT 答案只用于验证、破局或兜底。最终输出：

- `outputs/submissions/submit_voted.csv`
- `outputs/submissions/submit_voted_report.json`
