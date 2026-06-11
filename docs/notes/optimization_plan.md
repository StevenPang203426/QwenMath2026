# 优化方案：借鉴 Math2 决赛方案提升推理准确率

> 参考项目：CCF-BDCI 2020 小学数学应用题自动解题 · 决赛三等奖（Delores 队）
> 编写日期：2026-06-01

---

## 背景分析

Math2 方案（Graph2Tree + BERT-Seq2Seq 双方案）取得好成绩的核心原因不是模型本身强大，而是：

1. **生成表达式 → eval 计算**：模型生成数学表达式而非直接输出数字，用 Python eval() 计算确保精度无损
2. **2163 行答案后处理规则引擎**：根据题目上下文（百分数/分数/取整/保留位数）精确格式化答案
3. **15 个单模加权投票融合**：Graph2Tree ×9 + BERT-Seq2Seq ×8，跨架构一致性优先
4. **爬虫补全训练标注**：从作业帮爬取每道题的解题表达式作为训练监督信号

我们的 LLM 方案（Qwen2.5-0.5B + LoRA + GRPO）可以借鉴其中 1、2、3 点，将"想对"和"写对"职责分离。

---

## 优化方向总览

| 优先级 | 方向 | 预计提升 | 工作量 | 依赖 |
|--------|------|----------|--------|------|
| P0 必做 | 答案后处理规则引擎 | +5~10% | 2-3 小时 | 无，立即可做 |
| P1 推荐 | 多路推理投票融合 | +3~5% | 1-2 小时 | 需要多个训练好的模型 |
| P2 加分 | Prompt 分类 + 格式引导 | +2~3% | 30 分钟 | 无，立即可做 |

预计总提升：+10~18%

---

## P0 答案后处理规则引擎（最高优先级）

### 设计思路

新建 `src/inference/answer_postprocessor.py`，在模型输出答案后、写入 CSV 前，根据原始 question 文本做上下文感知格式化。

核心逻辑：先检测题目要求的答案类型，再对模型输出的数值进行格式转换。

### 核心框架

```python
def postprocess_answer(raw_answer: str, question: str) -> str:
    """根据题目上下文格式化答案"""
    answer_type = detect_answer_type(question)
    numeric_value = parse_numeric(raw_answer)
    
    if numeric_value is None:
        return raw_answer  # 无法解析，原样返回
    
    if answer_type == "percentage":
        return format_percentage(numeric_value, question)
    elif answer_type == "fraction":
        return format_fraction(numeric_value)
    elif answer_type == "ceil_integer":
        return str(math.ceil(numeric_value))
    elif answer_type == "floor_integer":
        return str(math.floor(numeric_value))
    elif answer_type == "round_n":
        n = detect_decimal_places(question)
        return format_rounded(numeric_value, n)
    elif answer_type == "integer":
        return str(round(numeric_value))
    else:
        return format_auto(numeric_value)
```

### 规则清单

#### 模块 A：答案类型检测（从 question 推断）

| 题目特征关键词 | 答案类型 | 输出格式 | 示例 |
|---|---|---|---|
| "百分之几" / "百分率" / "百分比" | percentage | X% | 25% |
| "率"（非"率的"）/ "浓度" / "纯度" / "盐度" | percentage | X% | 12.5% |
| "几分之几" / "分率" / "占" | fraction | a/b | 3/5 |
| "比例" / "比率" / "占比" / "比重" | fraction | a/b | 2/3 |
| "至少" / "最少" / "起码" + 量词 | ceil_integer | ceil(x) | 4 |
| "至多" / "最多" + "能" / "可以" / "装" | floor_integer | floor(x) | 3 |
| "多少辆/人/个/只/条/棵/支/张" | integer | int(x+0.5) | 12 |
| "保留一位小数" / "精确到0.1" / "精确到十分位" | round_1 | round(x,1) | 3.5 |
| "保留两位小数" / "精确到0.01" / "精确到百分位" | round_2 | round(x,2) | 3.14 |
| "保留整数" / "精确到个位" | round_0 | round(x,0) | 7 |
| 无特殊标记 + 结果整除 | auto_int | 整数 | 24 |
| 无特殊标记 + 结果不整除 | auto_float | 保留合理位数 | 3.5 |

#### 模块 B：格式标准化

| 规则 | 说明 | 示例 |
|---|---|---|
| 去除单位/LaTeX 残留 | 已有逻辑，保留 | "25千克" → "25" |
| 日期格式转换 | X月Y日 → X/Y | "4月5日" → "4/5" |
| 百分数标准化 | 当类型=百分数时，小数转百分数 | 0.25 → "25%" |
| 分数标准化 | 当类型=分数时，小数转最简分数 | 0.6 → "3/5" |
| 尾零处理 | 去除无意义尾零 | "3.0" → "3"，"25.50%" → "25.5%" |
| 补零处理 | "保留两位"时补足位数 | "3.1" → "3.10" |
| 非法字符兜底 | 去除任何非数字/小数点/负号/斜杠/百分号字符 | — |

#### 模块 C：特殊情况处理

| 场景 | 处理方式 |
|---|---|
| 模型输出 "无法计算" / 空字符串 | 返回 "0"（避免空行） |
| 模型输出多个数字 | 取最后一个（已有逻辑） |
| 百分数超出 [0, 100] 范围 | 标记为可疑但仍输出 |
| 分数分母为 0 | 返回原始输出 |

### 与 Math2 的差异

Math2 是对 eval(表达式) 的结果做后处理，格式化的输入是精确浮点数。我们是对 LLM 直接输出的答案做后处理，输入可能已经是格式化后的字符串（如模型已输出 "25%"），需要额外判断：如果模型输出已经符合目标格式，直接通过不修改。

---

## P1 多路推理投票融合

### 设计思路

Math2 用 15 个不同配置的模型投票。我们用**同模型多次采样 + 多 checkpoint + API fallback** 模拟类似效果。

### 实现方案

新建 `src/inference/ensemble_infer.py`：

```python
def ensemble_predict(question, models_dict, n_samples=5):
    """多路推理投票"""
    all_answers = []
    
    # 策略 A：同模型多温度采样
    primary_model = models_dict["grpo"]
    for temp in [0.1, 0.3, 0.5, 0.7]:
        answer = primary_model.generate(question, temperature=temp)
        all_answers.append(("grpo", answer))
    
    # 策略 B：多 checkpoint
    for name, model in models_dict.items():
        answer = model.generate(question, temperature=0.1)
        all_answers.append((name, answer))
    
    # 投票
    return weighted_majority_vote(all_answers)
```

### 投票融合规则

借鉴 Math2 的跨架构优先思想：

1. **多数一致**（≥3/5 相同答案）→ 直接采用，高置信度
2. **两种训练方式都投出相同答案** → 优先采用（跨方法一致性 > 单方法重复）
3. **全部不同** → 选 GRPO 模型答案（RL 训练后推理能力最强）
4. **低置信度**（无明显多数）→ fallback 到 DeepSeek API 直接推理

### 投票权重分配

| 模型来源 | 权重 | 理由 |
|---|---|---|
| GRPO 训练后 | 1.5 | RL 强化后推理最强 |
| DPO 训练后 | 1.2 | 偏好对齐，格式更好 |
| SFT CoT | 1.0 | 基础 CoT 能力 |
| DeepSeek API (fallback) | 2.0 | 大模型保底，仅低置信度时使用 |

### 成本控制

- 正常题目：本地模型 5 次采样，零成本
- 低置信度题目（约 10-20%）：调用 DeepSeek API，成本可控
- 预估 12000 题中约 1500-2400 题需要 API fallback

---

## P2 Prompt 分类 + 格式引导

### 设计思路

在推理前分析题目类型，将格式约束动态注入 system prompt，形成"双保险"：prompt 引导模型输出正确格式 + 后处理规则兜底修正。

### 实现方案

新建 `src/inference/question_classifier.py`：

```python
def build_adaptive_prompt(question: str, base_instruction: str) -> str:
    """根据题目类型追加格式约束"""
    constraints = []
    
    if re.search(r'百分之几|百分率|百分比', question):
        constraints.append("答案必须写成百分数形式，如25%")
    elif re.search(r'率|浓度|纯度', question) and '率的' not in question:
        constraints.append("答案写成百分数形式，如12.5%")
    
    if re.search(r'几分之几|分率', question):
        constraints.append("答案必须写成分数形式，如3/5，不要化成小数")
    
    if re.search(r'至少|最少|起码', question) and re.search(r'辆|人|只|条|次|船|车', question):
        constraints.append("不是整数时向上取整")
    elif re.search(r'至多|最多', question) and re.search(r'能|可以|装|分', question):
        constraints.append("不是整数时向下取整")
    
    precision = re.search(r'保留(\S{1,3})位小数', question)
    if precision:
        constraints.append(f"答案保留{precision.group(1)}位小数")
    
    if constraints:
        return base_instruction + "\n【格式约束】" + "；".join(constraints) + "。"
    return base_instruction
```

### 与 GRPO 奖励函数的协同

P2 的分类逻辑可以与 GRPO 训练时的格式标签奖励（R2）形成正反馈：训练时奖励正确格式，推理时在 prompt 中明确格式要求，进一步降低格式错误率。

---

## 实施顺序

```
第 1 步（立即）: P0 answer_postprocessor.py + P2 question_classifier.py
    ↓  不需要等待训练
第 2 步（训练完成后）: P1 ensemble_infer.py
    ↓  需要 SFT/DPO/GRPO 三个 checkpoint
第 3 步: 集成测试 → 提交
```

### 第 1 步详细文件清单

| 新建文件 | 说明 |
|---|---|
| `src/inference/answer_postprocessor.py` | 答案后处理规则引擎 |
| `src/inference/question_classifier.py` | 题目类型分类器 + 自适应 prompt |

| 修改文件 | 说明 |
|---|---|
| `src/inference/batch_infer.py` | 集成后处理器和分类器 |
| `src/inference/predictor.py` | 支持自适应 prompt 注入 |

### 第 2 步详细文件清单

| 新建文件 | 说明 |
|---|---|
| `src/inference/ensemble_infer.py` | 多路推理 + 投票融合 |

| 修改文件 | 说明 |
|---|---|
| `configs/inference/infer.yaml` | 新增 ensemble 配置节 |

---

## 风险与注意事项

1. **过度后处理风险**：如果模型已经输出正确格式（如已经是 "25%"），后处理不应反复转换。需要先检测是否已符合目标格式。
2. **ceil/floor 误判**：不是所有含 "至少" 的题都需要 ceil，如 "至少比小明多多少" 不需要取整。需要结合量词判断。
3. **分数/百分数歧义**："占总数的几分之几" vs "百分之几的折扣"，需要精确的关键词优先级。
4. **API fallback 成本**：需设置调用上限，避免竞赛期间超支。
5. **投票反而降分**：如果弱模型（如 SFT 未经 RL）投出大量错误答案，可能拉低整体。建议只用训练效果好的模型参与投票。

---

## 与 Math2 方案的核心差异

| 维度 | Math2 | 我们的方案 |
|---|---|---|
| 模型输出 | 数学表达式 → eval 计算 | 直接输出数字答案 |
| 后处理输入 | 精确浮点数 | 模型文本输出（可能已格式化） |
| 模型数量 | 15 个独立模型 | 1 个模型多采样 + 多 checkpoint |
| 训练数据 | 爬虫获取表达式标注 | DeepSeek API 生成 CoT |
| 计算精度 | eval 保证无损 | 依赖模型本身计算能力 |
| 推理成本 | 多模型各推理一次 | 单模型多次 + 少量 API |

我们方案的优势在于 LLM 本身的泛化能力远强于 2020 年的 GCN/Seq2Seq，劣势在于计算精度不可控。后处理规则引擎正是弥补这一劣势的关键手段。
