# 数据质量审计与增强方案

> 项目：CCF BDCI 小学数学应用题自动解题
> 编写日期：2026-06-02
> 状态：已实现，包含题目级修复、表达式格式修复、增强数据对应性审计

---

## 1. 背景

当前训练数据存在两类风险：

- 原始题目质量问题：OCR 错误、漏百分号、分数识别错误、题意歧义、标注答案错误等。
- 生成数据质量问题：正确表达式算错、错误表达式反而算对、错误表达式不可解析等。
- 表达式本身列式正确，但裸 `eval_result` 没有按题意输出百分数、分数、保留小数或离散取整格式，导致 `valid=false`。
- 规则增强样本需要验证题干和表达式是否由同一个源题、同一组 `changed_numbers` 同步替换得到，不能只看表达式计算值是否等于答案。

如果直接基于这些数据做 SFT/DPO/GRPO 或数据增强，坏题会被复制放大，训练集噪声增加。

因此新增一条数据治理流程：先做质量审计和高门槛自动修复，再只对高可信题做数据增强。

---

## 2. 总体策略

执行顺序：

1. 规则预筛：用现有表达式验证、答案一致性、异常文本模式筛出高风险题。
2. 大模型审计：只把高风险题交给商业大模型，降低 token 成本。
3. 高门槛自动修复：只有满足强验证条件的修复才写入 repaired 数据。
4. 表达式格式修复：对 `expr_correct.jsonl` 中 `valid=false` 的记录做题意格式化验证，独立输出格式修复数据。
5. 高可信数据增强：只增强审计通过且表达式可验证的题，并对增强结果做题干-表达式同源替换审计。
6. 合并训练集：原始高可信题 + 题目级 repaired 题 + 格式 repaired 题 + clean augmented 题，保留来源字段。

核心原则：

- 不覆盖 `data/raw/train.json`。
- 坏题不进入增强流程。
- 大模型不直接作为最终答案来源；能 eval 的答案必须由程序计算。
- 所有自动修复和增强都要保留 provenance，方便回滚和统计。

---

## 3. 质量审计

### 3.1 审计目标

识别以下问题：

- 题目有歧义，无法唯一确定答案。
- OCR 错误，例如 `1/4` 被识别为 `114`。
- 漏符号，例如漏 `%`、漏小数点、漏括号。
- 标注答案疑似错误。
- 题干信息不足，无法求解。
- 题目可解但当前生成表达式/CoT 不可靠。

### 3.2 标签体系

第一版使用小标签集，避免类别过细导致大模型输出不稳定。

| 标签 | 含义 | 处理方式 |
|------|------|----------|
| `ok` | 题目清晰，答案可信 | 可进入训练和增强 |
| `ambiguous` | 题意有歧义，存在多个解释 | 不自动增强，进入人工池 |
| `ocr_error` | 疑似 OCR 识别错误 | 可尝试高门槛修复 |
| `missing_symbol` | 漏 `%`、小数点、分数线等符号 | 可尝试高门槛修复 |
| `wrong_answer` | 题干清晰但标注答案疑似错误 | 可尝试高门槛修复 |
| `unsolvable` | 信息不足或矛盾，无法求解 | 过滤，不增强 |
| `needs_human` | 模型不确定，需要人工判断 | 过滤，不增强 |

### 3.3 规则预筛

不建议全量直接调用大模型审计。先用规则筛出高风险题。

预筛信号：

- `expr_builder.py correct` 多次生成正确表达式失败。
- `expr_builder.py wrong` 多次生成错误表达式失败或总是算对。
- DeepSeek CoT 的 `api_answer`、表达式 `eval_result`、标注 `answer` 三者不一致。
- 题干中出现疑似 OCR 模式：连续数字异常、`114`、`34` 等可能由分数丢失 `/` 产生的片段。
- 题干包含百分率语义但无 `%` 或“百分之”。
- 题干出现单位/条件缺失模式，例如“比多几分之几”但缺少比较对象。
- 多次生成得到不同答案。

预筛输出：`data/processed/quality_candidates.json`
实际落盘路径：`data/processed/intermediate/quality/quality_candidates.json`

```json
[
  {
    "id": "123",
    "question": "...",
    "answer": "...",
    "signals": ["expr_correct_failed", "answer_disagreement"],
    "risk_score": 0.82
  }
]
```

### 3.4 大模型审计输出

大模型只审计预筛出的候选题。

输出文件：`data/processed/intermediate/quality/quality_audit.json`

```json
[
  {
    "id": "123",
    "label": "missing_symbol",
    "confidence": 0.91,
    "reason": "题干问百分率，但数值表达缺少百分号，原答案与常规解法不一致。",
    "can_auto_repair": true,
    "repaired_question": "...",
    "repaired_answer": "...",
    "repair_expression": "..."
  }
]
```

---

## 4. 自动修复

### 4.1 写回门槛

自动修复采用高门槛写回。必须同时满足：

- 大模型明确指出问题类型，不是模糊猜测。
- `confidence >= 0.85`。
- 给出 `repaired_question`、`repaired_answer`、`repair_expression`。
- `repair_expression` 能通过 `safe_eval()`。
- `safe_eval(repair_expression)` 与 `repaired_answer` 一致。
- 修复后的题干不改变原题核心题型，只修正 OCR/符号/标注问题。

不满足以上条件的记录只保留审计结果，不写入 repaired 训练数据。

### 4.2 修复输出

输出文件：`data/processed/train_repaired.json`

```json
[
  {
    "id": "123",
    "source_id": "123",
    "source": "auto_repair",
    "question": "修复后的题干",
    "answer": "修复后的答案",
    "expression": "可计算表达式",
    "audit_label": "missing_symbol",
    "audit_confidence": 0.91,
    "repair_reason": "漏百分号"
  }
]
```

### 4.3 表达式格式修复

格式修复解决第二类问题：表达式列式正确，但 `safe_eval(expression)` 的裸结果没有满足题目要求。例如：

- 题目问“几分之几”，表达式结果是 `0.6`，标注答案是 `3/5`。
- 题目问“合格率/百分率”，表达式结果是 `0.25`，标注答案是 `25%`。
- 题目问“至少需要几辆车”，表达式结果是 `3.2`，标注答案是 `4`。
- 题目要求“保留两位小数”，表达式结果需要按题意四舍五入。

实现入口：

- 共享规则：`src/utils/answer_normalizer.py`
- 格式修复：`src/data/format_repair.py`
- 推理兼容入口：`src/inference/answer_postprocessor.py`

输出文件：

- 修复成功：`data/processed/intermediate/format_repair/expr_format_repaired.json`
- 修复拒绝：`data/processed/intermediate/format_repair/expr_format_rejected.json`
- 统计报告：`data/processed/intermediate/format_repair/format_repair_report.json`

格式修复不覆盖 `expr_correct.jsonl`，最终由合并脚本消费独立产物。

---

## 5. 数据增强

### 5.1 增强范围

只增强高可信题：

- 质量审计标签为 `ok`，或未进入高风险候选池。
- 已有正确表达式，且 `valid=True`。
- 题干可识别出可替换数字。
- 替换数字后表达式可重算，答案形式可规范化。

不增强：

- `ambiguous`、`unsolvable`、`needs_human`。
- 自动修复未通过写回门槛的题。
- 表达式不可解析或答案验证失败的题。

### 5.2 增强方式

第一版采用规则优先，不让大模型逐题改写。

规则增强流程：

1. 从题干和表达式中抽取数字。
2. 选择 1 个或多个可替换数字。
3. 按题型约束生成新数字，避免负数、非整数、不合理比例。
4. 同步替换题干中的数字和表达式中的对应数字。
5. 使用 `safe_eval()` 计算新答案，并按题意格式化。
6. 检查增强题干和增强表达式是否都能由 `source_id + changed_numbers` 精确推出。
7. 输出增强样本。

每道高可信题第一版最多生成 1 个增强样本。

### 5.3 大模型回退

只有规则无法安全替换但题目价值较高时，才交给大模型批量改写。

大模型回退仍不直接决定最终答案：

- 大模型输出新题干和表达式。
- 程序用 `safe_eval()` 计算答案。
- 验证通过才写入增强数据。

### 5.4 增强输出

输出文件：`data/processed/train_augmented.json`

```json
[
  {
    "id": "aug_123_0",
    "source_id": "123",
    "source": "rule_augment",
    "question": "增强后的题干",
    "answer": "新答案",
    "expression": "新表达式",
    "changed_numbers": [
      {"old": "105", "new": "128"}
    ]
  }
]
```

### 5.5 增强审计与清洗

增强审计的正确性口径不是“表达式值等于答案”，而是：

- 用 `source_id` 找回原始题干和原始表达式。
- 将 `changed_numbers` 同步应用到原始题干和原始表达式。
- 增强题干必须等于替换后的题干；增强表达式必须等于替换后的表达式。
- 每个替换数字必须同时命中题干和表达式。

当前检查结论：

- 当前重跑后的 `data/processed/train_augmented.json` 共 666 条。
- 666/666 条题干与表达式都满足同源替换对应关系。
- `train_augmented_clean.json` 中 0 条题干保留 Python list 字符串污染；本轮有 2 条题干在审计阶段被清洗。

输出文件：

- clean 增强数据：`data/processed/intermediate/augmentation/train_augmented_clean.json`
- rejected 增强数据：`data/processed/intermediate/augmentation/train_augmented_rejected.json`
- 审计报告：`data/processed/intermediate/augmentation/augmentation_report.json`

---

## 6. Token 成本控制

### 6.1 质量审计成本

全量 12000 题直接审计会消耗大量 token。第一版使用预筛后审计：

- 规则预筛全量，零 API 成本。
- 只审计高风险题，预计 5%-20%。
- 每题审计 prompt 控制为短 JSON 输入，要求短 JSON 输出。

如果高风险题为 1000 条，每题输入输出合计约 500-1000 tokens，则总量约 50-100 万 tokens，成本可控。

### 6.2 数据增强成本

规则增强几乎不消耗大模型 token。

token 主要来自大模型回退增强，应作为可选流程：

- 默认不开启。
- 只对规则增强失败但题型有价值的样本启用。
- 批量 prompt 一次处理多题，降低 system prompt 重复成本。

---

## 7. 训练集合并

建议最终合并文件：`data/processed/train_clean_augmented.json`

合并统计报告：`data/processed/train_clean_augmented_report.json`

合并来源：

- 原始高可信题：`source="raw_ok"`
- 自动修复题：`source="auto_repair"`
- 格式修复题：`source="expr_format_repair"`
- 规则增强题：`source="rule_augment"`
- 大模型回退增强题：`source="llm_augment"`

训练时可以按来源加权：

| 来源 | 建议权重 |
|------|----------|
| `raw_ok` | 1.0 |
| `auto_repair` | 0.7 |
| `expr_format_repair` | 0.8 |
| `rule_augment` | 0.8 |
| `llm_augment` | 0.6 |

第一版如果训练代码不支持样本权重，则只保留 `source` 字段用于后续统计。

当前已执行合并结果：

| 来源 | 含义 | 条数 | 占比 |
|------|------|------|------|
| `raw_ok` | 原始合格数据 | 10599 | 89.13% |
| `auto_repair` | 题意修复数据 | 347 | 2.92% |
| `expr_format_repair` | 表达式结果规范修复数据 | 279 | 2.35% |
| `rule_augment` | 增强后的合格数据 | 666 | 5.60% |
| **合计** | 最终清洗增强训练集 | **11891** | **100.00%** |

统一修复数据 `data/processed/train_repairs_unified.json` 共 626 条，其中题意修复 347 条、表达式结果规范修复 279 条。合并时修复数据优先于原始题，增强数据只保留通过同源替换审计的 `rule_augment` 样本。

拒绝数据处置：

- `expr_format_rejected.json` 中未通过题意归一匹配的数据不进入训练集，只保留为人工抽样审阅池。
- `expr_rejected.json` 中安全表达式修复失败的数据不进入 `train_expr_safe.json`，也不进入最终清洗增强训练集。
- 不继续大规模自动修复长尾 reject；下一步优先使用当前 clean merge 产物训练，并按 reject 原因抽样补规则。

---

## 8. 待实现模块

建议新增：

- `src/data/quality_auditor.py`：预筛 + 大模型审计 + 自动修复输出。
- `src/data/rule_augmentor.py`：规则改数 + 表达式重算。
- `src/data/merge_clean_data.py`：合并 raw_ok / repaired / augmented。
- `src/data/format_repair.py`：表达式正确但答案格式不规范的独立修复。
- `src/data/augmentation_auditor.py`：增强题干和表达式的同源替换审计。
- `src/utils/answer_normalizer.py`：数据修复和推理共用的答案格式规则。

建议新增脚本：

- `scripts/data.sh quality-audit`
- `scripts/data.sh format-repair`
- `scripts/data.sh augment`
- `scripts/data.sh clean-merge`

---

## 9. 第一版验收标准

- 质量审计能输出 `intermediate/quality/quality_candidates.json` 和 `intermediate/quality/quality_audit.json`。
- 自动修复只写回满足高门槛验证的记录。
- 格式修复不覆盖 `expr_correct.jsonl`，只生成独立修复/拒绝/报告产物。
- 数据增强默认不调用大模型，每道高可信题最多生成 1 个增强样本。
- 所有 clean 增强样本都通过 `source_id + changed_numbers` 的题干-表达式同源替换检查。
- 所有输出文件保留 `source` / `source_id`，不覆盖原始数据。
- 能统计各阶段数量：候选数、审计数、修复写回数、增强数、过滤数。
