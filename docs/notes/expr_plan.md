# 表达式方案设计文档

> 项目：CCF BDCI 小学数学应用题自动解题
> 编写日期：2026-06-01
> 状态：待实施

---

## 1. 方案概述

新增一条独立训练路线：让 Qwen2.5-0.5B 学会为每道题生成**可计算的中缀数学表达式**，然后用 Python `eval()` 得到精确答案。与现有 CoT 路线（模型直接输出数字答案）互补，最终 4 模型投票融合。

核心优势：表达式经 eval() 计算后答案精度无损，不存在模型"算错最后一步"的问题。只要表达式对了，答案一定对。

### 方案对比

| 维度 | CoT 路线（现有） | 表达式路线（新增） |
|------|------------------|-------------------|
| 模型输出 | 推理过程 + 数字答案 | 中缀数学表达式 |
| 答案来源 | 模型直接输出 | eval(表达式) 计算 |
| 精度风险 | 模型可能算错 | 表达式对则答案必对 |
| 泛化风险 | 强（自由文本） | 弱（必须是合法表达式） |
| 格式标签 | `<think>` `<answer>` | `<expr>` `<answer>` |

### 最终投票方案

4 模型参与 majority vote：

| 模型 | 路线 | 权重 |
|------|------|------|
| CoT-GRPO | CoT | 1.5 |
| CoT-SFT | CoT | 1.0 |
| Expr-GRPO | 表达式 | 1.5 |
| Expr-SFT | 表达式 | 1.0 |

---

## 2. 数据构造

表达式路线的数据构造应接入数据质量审计与增强流程，详见 `docs/notes/data_quality_augmentation_plan.md`。核心顺序是：先审计和高门槛自动修复，再只对高可信且表达式可验证的题做规则优先增强。

### 2.1 DeepSeek API 生成表达式

为训练集中每道题调用 DeepSeek V4 Flash API，生成中缀表达式。

**Prompt 设计：**

```
你是一位小学数学老师。请为以下数学题写出一个Python可直接计算的中缀数学表达式。

【格式要求】
- 只输出一个数学表达式，不要输出推理过程
- 使用纯数字和运算符：+  -  *  /  **  (  )
- 圆周率用3.14代替
- 不要使用任何变量名、中文、单位
- 分数直接写除法，如三分之二写成 2/3
- 百分数写成小数，如25%写成 0.25 或 25/100
- 表达式必须能被Python的eval()直接计算

请严格按以下格式回答：
<expr>你的表达式</expr>
```

**负样本 Prompt（用于 DPO）：**

```
你是一个数学学生。请为以下数学题写出一个Python可计算的数学表达式，
但请故意在某个数字或运算符上犯一个合理的错误。
语气自然，不要用"故意""错误地"等词。

请严格按以下格式回答：
<expr>你的表达式</expr>
```

### 2.2 自动验证

对 DeepSeek 返回的表达式做自动验证：

```python
def verify_expression(expr_str: str, gold_answer: str) -> bool:
    """验证表达式是否正确"""
    try:
        result = eval(expr_str)
        gold_norm = normalize_number(gold_answer)
        pred_norm = normalize_number(str(result))
        return pred_norm is not None and gold_norm is not None and pred_norm == gold_norm
    except:
        return False
```

验证通过 → 采用为正样本（SFT + DPO chosen）
验证失败 → 丢弃（或用作 DPO rejected，如果表达式本身可解析但结果错误）

### 2.2.1 合规筛查与自动重算

表达式数据构建启用默认筛查机制，断点续传时不再简单按 `id` 跳过已有记录，而是先判断记录是否合规。

**正确表达式模式（`correct`）合规条件：**

```python
status == "ok" and valid is True and expression 非空
```

如果已有记录出现以下情况，会自动重算：

- 表达式可解析但结果与标准答案不匹配
- 表达式为空
- 表达式不可解析
- API 失败
- 状态字段异常

**错误表达式模式（`wrong`）合规条件：**

```python
status == "ok" and valid is False and eval_result is not None and expression 非空
```

如果已有记录出现以下情况，会自动重算：

- 错误表达式反而算对（`accidentally_correct`）
- 错误表达式不可解析（`unparseable`）
- 表达式为空
- API 失败
- 状态字段异常

**重算策略：**

- 默认启用，不需要额外 `--repair` 参数
- 每条不合规记录最多重算 3 次（`MAX_REPAIR_ATTEMPTS = 3`）
- 3 次后仍不合规，则保留最后一次结果并写入诊断字段
- 转换 SFT/DPO 数据时仍只使用合规记录，不合规记录不会进入训练集

**诊断字段：**

```json
{
  "attempts": 3,
  "repair_reason": "wrong_accidentally_correct"
}
```

常见 `repair_reason`：

- `missing`：没有旧记录或新样本
- `correct_invalid`：正确表达式模式下结果不匹配
- `wrong_accidentally_correct`：错误表达式模式下反而算对
- `wrong_unparseable`：错误表达式不可解析
- `api_failed`：API 调用失败
- `empty_expression`：表达式为空
- `wrong_invalid` / `invalid_status`：其他状态异常

### 2.3 数据格式

**SFT 训练数据（train_expr.json）：**

```json
{
    "id": 0,
    "question": "食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？",
    "expression": "105*3",
    "answer": "315",
    "eval_result": "315"
}
```

**DPO 训练数据（train_expr_dpo.json）：**

```json
{
    "id": 0,
    "question": "食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？",
    "chosen": "<expr>105*3</expr>",
    "rejected": "<expr>105+3</expr>"
}
```

### 2.4 缓存策略

复用现有 data_builder.py 的幂等缓存机制，并增加质量门控：

- 按 prompt 类型分批（表达式正样本 / 表达式负样本）
- 相同前缀的 prompt 排列在一起，提高 DeepSeek context cache 命中率
- 已有且合规的结果跳过，断点续传
- 已有但不合规的结果自动进入重算队列，避免坏数据被缓存固化

### 2.5 预计数据量

| 数据类型 | 预计条数 | 说明 |
|----------|----------|------|
| 表达式正样本 | ~9000-10000 | 12000 条中约 75-85% 能生成正确表达式 |
| 表达式负样本 | ~5000-7000 | 多次采样或故意生错 |
| DPO 偏好对 | ~5000-7000 | chosen + rejected 配对 |

---

## 3. 训练流程

### 3.1 SFT 阶段

**输入格式（chat template）：**

```
system: 请为以下数学题写出一个Python可直接计算的数学表达式。用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。
user: 食堂运来105千克的萝卜，运来的青菜是萝卜的3倍，运来青菜多少千克？
assistant: <expr>105*3</expr><answer>315</answer>
```

**配置（configs/expr/sft_expr_clean.yaml）：**

```yaml
inherit: base
data:
  train_path: "data/splits/train_expr_clean_train.json"
  val_path: "data/splits/train_expr_clean_val.json"
  target_format: "expression"
training:
  output_dir: "outputs/checkpoints/sft_expr_clean"
  num_train_epochs: 3
  per_device_train_batch_size: 8
  learning_rate: 2.0e-4
```

### 3.2 GRPO 阶段

**奖励函数（6 维，专为表达式设计）：**

| 维度 | 函数 | 范围 | 说明 |
|------|------|------|------|
| R1 eval 正确性 | `expr_correctness_fn` | -0.6 ~ +1.0 | eval(expr) 后题意归一与 gold 匹配 |
| R2 可解析性 | `expr_parseable_fn` | -0.3 ~ +0.3 | 规范表达式且可 eval |
| R3 格式标签 | `expr_format_fn` | 0.0 ~ +0.2 | `<expr></expr><answer></answer>` 全有 → +0.2 |
| R4 无非法字符 | `expr_clean_fn` | -0.3 ~ 0.0 | 含中文/字母/LaTeX → -0.3，否则 0.0 |
| R5 answer 一致性 | `expr_answer_consistency_fn` | -0.2 ~ +0.2 | `<answer>` 与表达式题意归一结果一致 |
| R6 输出洁净 | `expr_output_cleanliness_fn` | -0.2 ~ 0.0 | 标签外解释或超长输出 → -0.2 |

理论总分范围：-1.6 ~ +1.7

**关键设计点：**

- R1 的 eval 正确性是最重要的维度，只信任 `<expr>`，`<answer>` 不兜底
- R2/R4 约束规范表达式，避免函数、比较、变量、LaTeX 等取巧写法
- R5/R6 是防守项，约束 answer 标签自洽并减少标签外解释

**配置（configs/expr/grpo_expr_clean.yaml）：**

```yaml
inherit: base
model:
  sft_checkpoint: "outputs/checkpoints/sft_expr_clean/best"
grpo:
  num_generations: 8
  max_new_tokens: 128        # 表达式比 CoT 短得多
  temperature: 0.8
  max_prompt_length: 256
  max_completion_length: 128
  use_vllm: false
training:
  output_dir: "outputs/checkpoints/grpo_expr_clean"
  num_train_epochs: 1
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  gradient_checkpointing: false
  learning_rate: 5.0e-6
  max_grad_norm: 0.1
  dataloader_num_workers: 4
```

注意 `max_new_tokens=128`：表达式通常只有 10-50 个 token，远短于 CoT 的 256，GRPO 训练速度会快很多。

---

## 4. 推理流程

### 4.1 表达式推理 + eval

```python
def expr_predict(model, tokenizer, question):
    """表达式路线推理"""
    # 1. 模型生成
    output = model.generate(question, instruction=EXPR_INSTRUCTION)
    
    # 2. 提取表达式
    expr = extract_between_tags(output, "expr")
    
    # 3. 安全 eval
    try:
        result = safe_eval(expr)  # 限制只允许数字和运算符
        return str(result)
    except:
        return None  # eval 失败，标记为需要 fallback
```

### 4.2 安全 eval

不使用裸 `eval()`，限制只允许数学运算：

```python
import ast
import operator

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def safe_eval(expr_str: str) -> float:
    """安全计算数学表达式，不允许任何函数调用或变量访问"""
    tree = ast.parse(expr_str, mode='eval')
    return _eval_node(tree.body)

def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _SAFE_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        return _SAFE_OPS[type(node.op)](operand)
    else:
        raise ValueError(f"Unsupported node: {type(node)}")
```

### 4.3 Fallback 策略

当 eval 失败时，fallback 到 CoT 模型的答案：

```
表达式模型输出 → 提取 <expr> → safe_eval()
    ├─ 成功 → postprocess_answer(result, question) → 最终答案
    └─ 失败 → 使用 CoT 模型的答案作为替代
```

---

## 5. 投票融合

### 5.1 融合逻辑

```python
def final_vote(cot_grpo_ans, cot_sft_ans, expr_grpo_ans, expr_sft_ans, question):
    """4 模型投票"""
    candidates = [
        ("cot_grpo", cot_grpo_ans, 1.5),
        ("cot_sft", cot_sft_ans, 1.0),
        ("expr_grpo", expr_grpo_ans, 1.5),   # None if eval failed
        ("expr_sft", expr_sft_ans, 1.0),     # None if eval failed
    ]
    # 过滤 eval 失败的
    valid = [(name, ans, w) for name, ans, w in candidates if ans is not None]
    return majority_vote(valid)
```

### 5.2 跨路线一致性优先

借鉴 Math2 的核心经验：**当 CoT 路线和表达式路线给出相同答案时，这个答案几乎一定是对的**。

优先级：
1. CoT + Expr 两路线一致 → 最高置信度，直接采用
2. 同路线多模型一致 → 次高置信度
3. 全部不同 → 选权重最高的（GRPO 模型）

---

## 6. 文件清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `src/data/expr_builder.py` | 表达式数据构建（调用 DeepSeek API + 验证） |
| `src/models/reward_expr.py` | 表达式 GRPO 6 维奖励函数 |
| `src/training/sft_expr_trainer.py` | 表达式 SFT 训练（或复用现有 sft_trainer） |
| `src/training/grpo_expr_trainer.py` | 表达式 GRPO 训练（使用表达式专用奖励） |
| `src/inference/expr_predictor.py` | 表达式推理 + safe_eval |
| `configs/expr/sft_expr.yaml` | 表达式 SFT 配置 |
| `configs/expr/grpo_expr.yaml` | 表达式 GRPO 配置 |
| `scripts/data.sh expr-build` | 表达式数据构建脚本 |
| `scripts/train.sh sft expr legacy` | 表达式 SFT 训练脚本 |
| `scripts/train.sh grpo expr legacy` | 表达式 GRPO 训练脚本 |
| `scripts/submit.sh vote` | 4 模型投票推理脚本 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/inference/ensemble_infer.py` | 扩展支持表达式路线 + 跨路线投票 |
| `src/inference/answer_postprocessor.py` | 支持 eval 结果的后处理 |

---

## 7. 实施顺序

```
第 1 步: expr_builder.py — 数据构造 + API 调用 + 验证
    ↓
第 2 步: sft_expr_trainer.py + configs/expr/sft_expr.yaml — SFT 训练
    ↓
第 3 步: reward_expr.py — 表达式专用奖励函数
    ↓
第 4 步: grpo_expr_trainer.py + configs/expr/grpo_expr.yaml — GRPO 训练
    ↓
第 5 步: expr_predictor.py — 推理 + safe_eval + fallback
    ↓
第 6 步: 扩展 ensemble_infer.py — 4 模型投票融合
    ↓
第 7 步: 集成测试 → 提交
```

### 时间预估

| 步骤 | 预计耗时 | 说明 |
|------|----------|------|
| 数据构造 | 2-4 小时 | DeepSeek API 调用 12000 题 |
| SFT 训练 | 1-2 小时 | 表达式短，训练快 |
| GRPO 训练 | 2-4 小时 | max_new_tokens=128，比 CoT 快 2-3 倍 |
| 推理 + 投票 | 1-2 小时 | 4 模型各推理一次 |
| 总计 | 6-12 小时 | |

---

## 8. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 复杂题（多步骤）表达式过长 | 中 | 限制 max_new_tokens=128，超长截断后 eval 失败则 fallback |
| 表达式正确率低（<70%） | 低 | DeepSeek V4 Flash 能力强，预计 >80%；不够则换 V4 Pro |
| 0.5B 模型学不会表达式 | 低 | 表达式比 CoT 短且结构化，对小模型更友好 |
| eval 安全风险 | 低 | 使用 safe_eval（AST 白名单），不允许函数调用 |
| 百分数/分数格式问题 | 中 | eval 结果经 postprocessor 格式化，已有规则覆盖 |
