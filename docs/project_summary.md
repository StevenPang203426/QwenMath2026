# 项目整体方案与新增内容总览

> 项目：CCF BDCI 小学数学应用题自动解题
> 模型：Qwen2.5-0.5B-Instruct + LoRA
> 更新日期：2026-06-11

---

## 一、项目整体架构

本项目采用 **双路线并行 + 4 模型投票融合** 的方案：

| 路线 | 思路 | 模型输出 | 答案来源 |
|------|------|----------|----------|
| CoT 路线 | 让模型学会逐步推理 | 推理过程 + 数字答案 | 模型直接输出 |
| 表达式路线 | 让模型学会写数学表达式 | 中缀数学表达式 | Python eval() 精确计算 |

两条路线共享 Qwen2.5-0.5B 基座模型，各自独立训练，最终 4 模型投票融合输出最终答案。CoT 路线中的 DPO/GRPO 是二阶段 adapter：训练时先把 `sft_cot` adapter merge 到 base，再训练新的 DPO/GRPO LoRA；推理和评测时必须使用 `SFT-merged base + DPO/GRPO adapter`。

---

## 二、训练流程总览

```
数据构造（DeepSeek V4 Flash API）
├── CoT 数据：推理过程 + 答案（data_builder.py）
├── DPO 数据：正确推理 vs 错误推理（data_builder.py + data_repair.py）
├── 表达式数据：中缀表达式 + eval 验证（expr_builder.py）  [新增]
└── 数据修复：失败数据修复、Type B/C 回收（data_repair.py）

训练阶段
├── CoT 路线：SFT → DPO / GRPO（DPO/GRPO 均基于 SFT-merged base；GRPO 使用 5 维奖励函数）
└── 表达式路线：SFT → GRPO（6 维奖励函数）            [新增]

推理阶段
├── CoT 二阶段 adapter 加载（DPO/GRPO = SFT-merged base + adapter）
├── 自适应 Prompt（题目类型检测 → 格式约束注入）        [新增]
├── 答案后处理（百分数/分数/取整/保留位数规则引擎）      [新增]
├── 表达式推理（safe_eval + fallback）                  [新增]
└── 4 模型投票融合                                      [新增]
```

---

## 三、本轮新增内容

### 3.1 GRPO 五维奖励函数重新设计

**文件：** `src/models/reward.py`，`src/training/grpo_trainer.py`，`configs/grpo.yaml`

参考 DeepSeek-R1、GRPO-LEAD（EMNLP 2025）设计 5 个独立奖励函数：

| 维度 | 范围 | 说明 |
|------|------|------|
| R1 正确性 | -0.5 ~ +1.0 | 精确匹配 +1.0，四舍五入等价 +0.3，错误 -0.5 |
| R2 格式标签 | 0.0 ~ +0.2 | `<think></think><answer></answer>` 全有 +0.2 |
| R3 逻辑词 | -0.3 ~ +0.95 | 基础组 ×0.05 + 结构组 ×0.10，0 命中 -0.3 |
| R4 长度正则 | -0.3 ~ 0.0 | 过短 -0.3，过长 -0.2 |
| R5 无 LaTeX | -0.3 ~ 0.0 | 含 LaTeX -0.3 |

关键设计：四舍五入等价判断（`_round_match`）、逻辑词按种类数计分防刷词、数据驱动的长度阈值。

### 3.2 Fixed-base CoT 消融结论（2026-06-11）

修复 DPO/GRPO 推理时的 SFT-merged base 加载后，二阶段模型比错误加载版明显恢复，但当前最强单模型仍是 `sft_cot:direct`。

| 模型/Prompt | Validation Accuracy | Missing answer tag | 当前定位 |
|---|---:|---:|---|
| `sft_cot:direct` | 0.7376 | 16 | CoT 单模型首选 |
| `grpo:zero_shot_cot` | 0.6347 | 185 | 可作为投票辅助 |
| `grpo:few_shot_cot` | 0.6312 | 5 | 格式稳定但准确率不足 |
| `dpo:few_shot_cot` | 0.4333 | 93 | 暂不作为主投票模型 |

当前结论：不重训 SFT；GRPO 需要先调整 reward 后再考虑从 `sft_cot` 重训；DPO 先审计数据与训练日志，再决定是否重训。投票阶段应以 `sft_cot:direct` 为主权重，GRPO 仅作为互补候选。

### 3.2.1 CoT 修复实验计划与工具（2026-06-11）

当前提交策略先不替换 CoT 主模型：单模型继续使用 `sft_cot:direct`；prompt-specific ensemble 先通过离线脚本验证，推荐初始权重为 `sft_cot:direct=1.0`、`grpo:zero_shot_cot=0.35`、`grpo:few_shot_cot=0.30`、`dpo:few_shot_cot=0.0`。如果离线 weighted vote 低于 0.7376，则当前提交只保留 `sft_cot:direct`。

GRPO 下一步采用 strict/balanced 双 reward smoke：两版都从 `sft_cot/best` 开始，写入独立 checkpoint；通过 `training.max_steps` 做小步数筛选。成功进入全量训练的门槛是 validation accuracy 超过旧 GRPO 0.6347 且 missing answer tag <= 5%；最终替换门槛是达到或超过 `sft_cot:direct = 0.7376` 且 missing answer tag <= 5%。DPO 只做 pair、截断、loss 和中间 checkpoint 审计，不进入本轮重训。

Smoke 结果已完成：`balanced:direct` 达到 0.7358（844/1147，missing tag 15），`strict:direct` 达到 0.7350（843/1147，missing tag 17）。两版均通过 smoke 门槛，balanced 更高且格式更稳，因此进入全量 GRPO。

Balanced full 结果没有保持 smoke 收益：最佳为 `zero_shot_cot = 0.7010`（804/1147，missing tag 49），`direct = 0.6922` 且 missing tag 165，`few_shot_cot = 0.6888`。因此 full checkpoint 未达到 `sft_cot:direct = 0.7376` 的替换门槛，当前主模型仍固定为 `sft_cot:direct`。下一轮 GRPO 不应直接跑完整 1 epoch，应采用 capped/staged full（例如 max_steps checkpoint sweep）来避免过训练和格式退化。

新增入口：`scripts/run_cot_offline_ensemble.sh`、`scripts/run_grpo_cot_reward_smoke.sh`、`scripts/run_dpo_audit.sh`、`scripts/run_grpo_cot_reward_full.sh`。全量训练配置为 `configs/grpo_cot_reward_balanced_full.yaml`，输出到 `outputs/checkpoints/grpo_cot_reward_balanced_full`，不会覆盖旧 GRPO checkpoint。

### 3.3 GRPO 训练加速

**文件：** `configs/grpo.yaml`，`src/training/grpo_trainer.py`

| 变更 | 原值 → 新值 | 效果 |
|------|-------------|------|
| vLLM generation | 无 → 开启 | generation 阶段 5-8x |
| batch_size | 2 → 8 | GPU 利用率 14% → ~50% |
| grad_accum | 8 → 2 | effective batch 不变 |
| gradient_checkpointing | true → false | forward 快 ~30% |
| num_generations | 16 → 8 | generation 量减半 |
| max_completion_length | 512 → 256 | 每个 completion 短一半 |

预计训练时间从 82 小时降至 2-5 小时。

### 3.4 答案后处理规则引擎（P0）

**文件：** `src/inference/answer_postprocessor.py`

借鉴 Math2 方案（CCF-BDCI 2020 决赛三等奖，2163 行规则引擎）的思想，根据题目上下文对模型输出答案进行格式化：

| 题目特征 | 后处理 | 示例 |
|----------|--------|------|
| "百分之几" / "率" | → X% | 0.25 → 25% |
| "几分之几" / "分率" | → a/b | 0.6 → 3/5 |
| "至少" + 量词 | → ceil(x) | 3.2 → 4 |
| "至多" + "能/可以" | → floor(x) | 3.8 → 3 |
| "保留N位小数" | → round(x, N) | 3.14159 → 3.14 |
| "多少辆/人/个" | → 整数 | 12.0 → 12 |

优先级：格式（百分数/分数/保留位数）> 取整（ceil/floor）> 整数 > 自动。

测试集覆盖：8000 题中 612 题（7.7%）有明确格式约束。

### 3.5 题目类型分类器 + 自适应 Prompt（P2）

**文件：** `src/inference/question_classifier.py`

推理前分析题目类型，将格式约束动态注入 system prompt，与后处理形成双保险：

```
基础 prompt + 【格式约束】答案写成百分数形式，如25%。
```

### 3.6 多路推理投票融合（P1）

**文件：** `src/inference/ensemble_infer.py`

同模型多温度采样（5 次）+ 多 checkpoint 投票，本地 majority vote。不使用外部 API（竞赛不允许）。

### 3.7 表达式方案（完整新路线）

**设计文档：** `docs/expr_plan.md`

#### 3.6.1 数据构造

**文件：** `src/data/expr_builder.py`

- 调用 DeepSeek API 生成中缀表达式
- safe_eval（AST 白名单）验证：eval(表达式) == gold answer 则采用
- 自动重算机制：不合规记录最多重算 3 次
- 正确表达式 → SFT 训练数据；错误表达式 → DPO rejected

#### 3.6.2 表达式 GRPO 6 维奖励函数

**文件：** `src/models/reward_expr.py`

| 维度 | 范围 | 说明 |
|------|------|------|
| R1 eval 正确性 | -0.6 ~ +1.0 | eval(expr)==gold → +1.0 |
| R2 可解析性 | -0.3 ~ +0.3 | eval 不报错 → +0.3 |
| R3 格式标签 | 0.0 ~ +0.2 | `<expr></expr><answer></answer>` 全有 |
| R4 无非法字符 | -0.3 ~ 0.0 | 含中文/字母/LaTeX → -0.3 |
| R5 answer 一致性 | -0.2 ~ +0.2 | `<answer>` 与表达式题意归一结果一致 |
| R6 输出洁净 | -0.2 ~ 0.0 | 标签外解释或超长输出 → -0.2 |

#### 3.6.3 表达式推理

**文件：** `src/inference/expr_predictor.py`

模型输出 → 提取 `<expr>` → safe_eval() → postprocess → 如果 eval 失败则 fallback 到 CoT 答案。

#### 3.6.4 表达式 GRPO 训练器

**文件：** `src/training/grpo_expr_trainer.py`，`configs/grpo_expr.yaml`

`max_new_tokens=128`（表达式远短于 CoT），训练速度比 CoT 路线快 2-3 倍。

### 3.7 4 模型投票融合

**文件：** `scripts/run_infer_vote.sh`

| 模型 | 路线 | 权重 |
|------|------|------|
| CoT-GRPO | CoT | 1.5 |
| CoT-SFT | CoT | 1.0 |
| Expr-GRPO | 表达式 | 1.5 |
| Expr-SFT | 表达式 | 1.0 |

核心规则：**跨路线一致性优先**——当 CoT 和表达式两条路线给出相同答案时，该答案几乎一定正确。

### 3.8 数据质量审计与修复

**文件：** `src/data/quality_auditor.py`，`docs/data_quality_augmentation_plan.md`

流程：expr_builder 3 次重算仍失败的记录 → 自动导出到 `quality_candidates.json` → quality_auditor 预筛 + 大模型审计 → 高门槛修复（safe_eval 验证 repair_expression）。

触发条件：题目歧义、OCR 错误、漏百分号/小数点、标注答案错误。审计修复后直接使用 repair_expression，不需要重新调用 API 生成。

### 3.9 规则数据增强

**文件：** `src/data/rule_augmentor.py`

对高可信题（审计通过 + 表达式可验证）做数字替换增强：提取题干和表达式中共有的数字 → 扰动替换 → safe_eval 重算新答案。

关键约束：禁替换常数黑名单（3.14 / 0 / 1 / 2 / 12 / 24 / 60 / 100 / 365 等）。每题最多 1 个增强样本，增强数据仅用于 GRPO 阶段（不进入 SFT）。

---

## 四、完整文件清单

### 新建文件

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/models/reward.py` | 重写 | CoT 5 维 GRPO 奖励函数 |
| `src/models/reward_expr.py` | 新建 | 表达式 6 维 GRPO 奖励函数 |
| `src/data/expr_builder.py` | 新建 | 表达式数据构建 + safe_eval + API 调用 |
| `src/inference/answer_postprocessor.py` | 新建 | 答案后处理规则引擎 |
| `src/inference/question_classifier.py` | 新建 | 题目类型分类器 + 自适应 Prompt |
| `src/inference/ensemble_infer.py` | 新建 | 多路推理投票融合 |
| `src/inference/expr_predictor.py` | 新建 | 表达式推理 + safe_eval + fallback |
| `src/training/grpo_expr_trainer.py` | 新建 | 表达式 GRPO 训练器 |
| `configs/grpo_expr.yaml` | 新建 | 表达式 GRPO 配置 |
| `configs/sft_expr.yaml` | 新建 | 表达式 SFT 配置 |
| `tests/test_postprocessor.py` | 新建 | 后处理单元测试 |
| `docs/optimization_plan.md` | 新建 | 优化方案文档（借鉴 Math2） |
| `docs/expr_plan.md` | 新建 | 表达式方案设计文档 |

### 新建脚本

| 脚本 | 命令 |
|------|------|
| `scripts/run_expr_data_build.sh` | `bash scripts/run_expr_data_build.sh correct <api_key>` |
| `scripts/run_sft_expr.sh` | `bash scripts/run_sft_expr.sh` |
| `scripts/run_grpo_expr.sh` | `bash scripts/run_grpo_expr.sh` |
| `scripts/run_infer_vote.sh` | `bash scripts/run_infer_vote.sh` |
| `scripts/run_infer_ensemble.sh` | `bash scripts/run_infer_ensemble.sh` |
| `scripts/run_classify_questions.sh` | `bash scripts/run_classify_questions.sh data/raw/test.json` |
| `scripts/run_postprocess_csv.sh` | `bash scripts/run_postprocess_csv.sh <input.csv>` |
| `scripts/run_test_postprocessor.sh` | `bash scripts/run_test_postprocessor.sh` |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/training/grpo_trainer.py` | 集成新奖励函数 + vLLM + 加速参数 |
| `src/inference/batch_infer.py` | 集成后处理器 + 分类器 + 去表头 |
| `src/models/__init__.py` | 更新导入（移除已删除的旧函数） |
| `configs/grpo.yaml` | 训练加速参数 + 5 维奖励注释 |

---

## 五、执行顺序

### CoT 路线（现有，已部分完成）

```
1. bash scripts/run_data_build.sh           # CoT 数据构建
2. bash scripts/run_sft_cot.sh              # CoT SFT 训练
3. bash scripts/run_dpo.sh                  # CoT DPO 训练
4. bash scripts/run_grpo.sh                 # CoT GRPO 训练
```

### 表达式路线（新增）

```
5. bash scripts/run_expr_data_build.sh correct <api_key> 50   # 小批量测试
6. bash scripts/run_expr_data_build.sh correct <api_key>       # 全量正确表达式
7. bash scripts/run_expr_data_build.sh wrong <api_key>         # 错误表达式
8. bash scripts/run_expr_data_build.sh convert_sft             # 转换 SFT 格式
9. bash scripts/run_sft_expr.sh                                # 表达式 SFT
10. bash scripts/run_grpo_expr.sh                              # 表达式 GRPO
```

### 推理 + 提交

```
11. bash scripts/run_infer_vote.sh          # 4 模型投票推理 → submit_voted.csv
```

### 辅助工具

```
bash scripts/run_classify_questions.sh data/raw/test.json   # 查看题目类型分布
bash scripts/run_postprocess_csv.sh <旧csv>                  # 对旧结果做后处理
bash scripts/run_test_postprocessor.sh                       # 后处理单元测试
```

---

## 六、设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 是否引入 Graph2Tree/BERT-Seq2Seq | 不引入 | 性能天花板低（77-85%），工程成本极高，环境不兼容 |
| 表达式方案与 CoT 方案的关系 | 完全独立 | 两条路线互补，表达式保精度，CoT 保泛化 |
| 后处理 ceil/floor 对整数的处理 | 已是整数则直接通过 | 避免 ceil(4)=4 的无意义操作 |
| 百分数判断（模型输出 25 + 题目问百分率） | 25 视为 25%，直接加 % | 模型大概率已理解题意 |
| 分数转换精度 | Fraction.limit_denominator(100) | 覆盖小学常见分数 |
| 规则优先级 | 格式 > 取整 > 整数 > 自动 | 避免冲突 |
| 竞赛是否允许外部 API | 不允许 | 仅本地模型投票，无 fallback |
| 投票模型数 | 4 个 | CoT-GRPO + CoT-SFT + Expr-GRPO + Expr-SFT |
| 四舍五入方式 | Decimal ROUND_HALF_UP | 避免 Python 默认 banker's rounding |
