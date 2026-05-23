# 小学数学应用题自动解题 — 计划生成提示词

## 一、项目背景与约束

你正在为一个 **CCF BDCI 小学数学应用题自动解题** 课程实践项目制定详细的实施计划。

### 核心约束
- **推理模型限制**：最终提交答案的模型严格限制为 **Qwen2.5-0.5B-Instruct**（或更小模型），不可使用更大模型做推理。
- **数据限制**：仅允许对 12000 条官方训练数据进行重构和增强，禁止处理测试数据（8000条）。可用更大模型（DeepSeek V4 API）辅助**构建训练数据**，但不可用于最终推理。
- **评测标准**：正确率 = 正确数 / 总数，答案为纯数字（无单位）。
- **时间线**：2 周完成全部工作。
- **GPU 环境**：AutoDL/魔搭云平台，按需租用。
- **交付物**：submit.csv（平台提交） + 课程实验报告 + 完整代码仓库。

### 现有 Baseline
- 来源：`https://github.com/AI-FDU/Math_Solver`
- 方法：Qwen2.5-0.5B-Instruct + LoRA SFT，直接预测数字答案（无推理过程）
- 框架：原生 transformers Trainer + peft + swanlab
- 数据格式：`{"id", "question", "answer", "instruction"}`，answer 为纯数字

---

## 二、需要实现的四个方案

### 方案 1：思维链 CoT（不微调，纯提示工程）

**目标**：不微调模型，直接通过优化 prompt 提升 Qwen2.5-0.5B 的解题能力。

**子任务**：
1. **Zero-shot CoT**：在 instruction 中加入 "让我们一步一步思考" 等触发语
2. **Few-shot CoT**：在 prompt 中提供 3-5 个带推理步骤的示例
3. **答案提取**：从 CoT 输出中用正则提取最终数字
4. **对比实验**：对比不同 prompt 策略的正确率

**参考项目**：
- [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math) — Qwen 官方数学模型，含 CoT 和 TIR 两种推理模式
- [ashkunwar/COT_Finetuning](https://github.com/ashkunwar/COT_Finetuning) — Qwen2.5 CoT 微调示例（参考其 prompt 设计）

**关键注意**：0.5B 模型的 CoT 能力有限，此方案预期正确率可能不高，但作为 baseline 对比有重要实验价值。

---

### 方案 2：数据构建 + CoT SFT

**目标**：用 DeepSeek V4 API 为 12000 条训练数据生成带推理步骤的 CoT 答案，然后用这些高质量数据对 Qwen2.5-0.5B 做 SFT。

**子任务**：
1. **CoT 数据生成**：
   - 调用 DeepSeek V4 API，让其为每道题生成详细的推理过程 + 最终答案
   - 验证生成答案与标注答案是否一致（过滤错误样本）
   - 同时让其生成一些**错误推理路径**（为后续 DPO 方案准备偏好数据）
2. **数据增强**（可选）：
   - 基于现有题目修改数字生成新题（数据增强）
   - 用 DeepSeek V4 验证增强数据的正确性
3. **CoT SFT 训练**：
   - 用带推理步骤的数据训练模型，让模型学会"先推理再回答"
   - 训练格式：`question → <think>推理过程</think><answer>数字答案</answer>`
4. **答案提取**：
   - 正则表达式提取 `<answer>` 标签内容（主方案，推荐）
   - 后备：匹配最后出现的数字

**参考项目**：
- [doublelei/Awesome-Math-LLM](https://github.com/doublelei/Awesome-Math-LLM) — 数学 LLM 资源集合，含大量数据合成方法
- [tongyx361/Awesome-LLM4Math](https://github.com/tongyx361/Awesome-LLM4Math) — 数学推理数据构建方法概览
- JiuZhang3.0 论文 (arxiv 2405.14365) — 小数据合成模型提升数学推理

---

### 方案 3：DPO（直接偏好优化）

**目标**：利用方案 2 构建的正/误 CoT 偏好数据，对 SFT 后的模型进行 DPO 训练。

**子任务**：
1. **偏好数据构建**：
   - chosen：正确推理 + 正确答案
   - rejected：错误推理或错误答案（来自方案 2 的错误路径）
   - 格式：`{"prompt", "chosen", "rejected"}`
2. **DPO 训练**：
   - 在 SFT checkpoint 基础上进行 DPO 训练
   - 使用 trl 的 DPOTrainer 或 LLaMA-Factory 的 DPO 模式
3. **超参调优**：beta 参数、学习率等

**参考项目**：
- [huggingface/trl](https://github.com/huggingface/trl) — HuggingFace 官方 RL 训练库，DPOTrainer 原生支持 Qwen
- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — 统一微调框架，YAML 配置驱动，支持 SFT/DPO/GRPO
- [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — 小模型 RL 推理训练，含 Qwen2.5-Math 实验

---

### 方案 4：GRPO（组相对策略优化）

**目标**：使用 DeepSeek-R1 中提出的 GRPO 算法，通过多响应采样和组内排序来优化模型策略。

**子任务**：
1. **奖励函数设计**：
   - 格式奖励：是否按 `<think>...</think><answer>...</answer>` 格式输出
   - 正确性奖励：提取的答案是否与标注一致
2. **GRPO 训练**：
   - 每个 prompt 采样 G 个响应（如 G=4-8）
   - 计算组内相对优势
   - 更新策略
3. **显存优化**：
   - 0.5B 模型 + 多响应采样，单卡 24GB 应可行但需注意 batch size
   - 可用 gradient checkpointing + 较小的 G 值

**参考项目**：
- [huggingface/open-r1](https://github.com/huggingface/open-r1) — DeepSeek-R1 的完全开源复现，含 `grpo.py`
- [mkantwala/DeepSeek-R1-TrainingSuite](https://github.com/mkantwala/DeepSeek-R1-TrainingSuite) — GRPO + LoRA 微调实现
- [FareedKhan-dev/train-deepseek-r1](https://github.com/FareedKhan-dev/train-deepseek-r1) — 从零构建 DeepSeek-R1，含 GRPO 教程
- [philschmid/deep-learning-pytorch-huggingface](https://github.com/philschmid/deep-learning-pytorch-huggingface/blob/main/training/mini-deepseek-r1-aha-grpo.ipynb) — mini DeepSeek-R1 GRPO notebook
- [verl-project/verl](https://github.com/verl-project/verl) — EuroSys 2025 接收，支持 GRPO/PPO 等多种 RL 方法

---

## 三、项目结构规范

请严格按照以下目录结构组织代码：

```
Math_Solver/
├── README.md                          # 项目说明、环境配置、运行指南
├── requirements.txt                   # Python 依赖
├── setup.py / pyproject.toml          # 可选，包管理
│
├── configs/                           # ★ 所有配置集中管理（YAML）
│   ├── base.yaml                      # 公共配置（模型路径、数据路径、seed等）
│   ├── sft_baseline.yaml              # 方案0：Baseline SFT 配置
│   ├── sft_cot.yaml                   # 方案2：CoT SFT 配置
│   ├── dpo.yaml                       # 方案3：DPO 配置
│   ├── grpo.yaml                      # 方案4：GRPO 配置
│   └── infer.yaml                     # 推理配置（模型路径、生成参数）
│
├── data/                              # 数据目录
│   ├── raw/                           # 原始数据（不可修改）
│   │   ├── train.json                 # 12000 条训练数据
│   │   └── test.json                  # 8000 条测试数据
│   ├── processed/                     # 预处理后的数据
│   │   ├── train_cot.json             # CoT 增强训练数据
│   │   ├── train_dpo.json             # DPO 偏好对数据
│   │   └── train_augmented.json       # 数据增强后的数据
│   └── splits/                        # 可选：训练/验证划分
│       ├── train_split.json
│       └── val_split.json
│
├── src/                               # 核心代码
│   ├── __init__.py
│   ├── data/                          # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py                 # Dataset 类定义
│   │   ├── data_builder.py            # CoT 数据生成（调用 DeepSeek API）
│   │   ├── data_augmentor.py          # 数据增强（改数字等）
│   │   ├── preprocessor.py            # 数据预处理、格式转换
│   │   └── answer_extractor.py        # ★ 答案提取（正则 + 后备方案）
│   │
│   ├── models/                        # 模型相关
│   │   ├── __init__.py
│   │   ├── model_loader.py            # 统一的模型加载（base / LoRA merge）
│   │   └── reward.py                  # GRPO 奖励函数定义
│   │
│   ├── training/                      # 训练脚本
│   │   ├── __init__.py
│   │   ├── sft_trainer.py             # SFT 训练（baseline + CoT SFT）
│   │   ├── dpo_trainer.py             # DPO 训练
│   │   └── grpo_trainer.py            # GRPO 训练
│   │
│   ├── inference/                     # 推理脚本
│   │   ├── __init__.py
│   │   ├── predictor.py               # 统一推理接口
│   │   ├── cot_prompting.py           # 方案1：CoT 提示工程
│   │   └── batch_infer.py             # 批量推理生成 submit.csv
│   │
│   └── utils/                         # 工具函数
│       ├── __init__.py
│       ├── config.py                  # YAML 配置加载与合并
│       ├── metrics.py                 # 评测指标（正确率计算）
│       ├── logger.py                  # 日志配置（wandb 集成）
│       └── seed.py                    # 随机种子固定
│
├── scripts/                           # 一键运行脚本
│   ├── run_baseline.sh                # 跑 baseline
│   ├── run_data_build.sh              # 跑数据构建
│   ├── run_sft_cot.sh                 # 跑 CoT SFT
│   ├── run_dpo.sh                     # 跑 DPO
│   ├── run_grpo.sh                    # 跑 GRPO
│   ├── run_infer.sh                   # 跑推理
│   └── run_all_experiments.sh         # 全流程自动化
│
├── notebooks/                         # 实验分析
│   ├── eda.ipynb                      # 数据探索性分析
│   ├── error_analysis.ipynb           # 错误分析（哪类题目错误率高）
│   └── results_visualization.ipynb    # 实验结果对比可视化
│
├── outputs/                           # 训练产出（.gitignore）
│   ├── checkpoints/                   # 模型 checkpoint
│   │   ├── baseline/
│   │   ├── sft_cot/
│   │   ├── dpo/
│   │   └── grpo/
│   ├── submissions/                   # 提交文件
│   │   ├── submit_baseline.csv
│   │   ├── submit_cot_prompt.csv
│   │   ├── submit_sft_cot.csv
│   │   ├── submit_dpo.csv
│   │   └── submit_grpo.csv
│   └── logs/                          # wandb 本地日志
│
├── report/                            # 课程报告
│   ├── figures/                       # 报告图表
│   └── report.md                      # 实验报告（或 .tex）
│
└── .gitignore                         # 忽略 outputs/checkpoints、模型权重等
```

### 配置文件规范（YAML）

**`configs/base.yaml`** — 所有方案共享的基础配置：
```yaml
# 模型
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  cache_dir: "./model_cache"
  torch_dtype: "bfloat16"

# 数据
data:
  train_path: "data/raw/train.json"
  test_path: "data/raw/test.json"
  max_length: 512

# LoRA
lora:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 通用训练
training:
  seed: 42
  bf16: true
  gradient_checkpointing: true

# 日志
logging:
  tool: "wandb"                        # wandb 而非 swanlab
  project: "math-solver"
  log_steps: 10

# 答案提取
answer_extraction:
  method: "regex"                      # regex | model
  patterns:
    - '<answer>(.*?)</answer>'         # 优先匹配标签
    - '答案[是为：:]\s*(\d+\.?\d*)'    # 中文关键词
    - '(\d+\.?\d*)\s*$'               # 最后一个数字
```

**`configs/sft_cot.yaml`** — CoT SFT 特定配置（继承 base）：
```yaml
inherit: base

data:
  train_path: "data/processed/train_cot.json"
  max_length: 768                      # CoT 需要更长上下文

training:
  output_dir: "outputs/checkpoints/sft_cot"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  warmup_ratio: 0.05
  lr_scheduler_type: "cosine"
  save_strategy: "steps"
  save_steps: 500
  eval_strategy: "steps"
  eval_steps: 500
```

**`configs/dpo.yaml`**：
```yaml
inherit: base

data:
  train_path: "data/processed/train_dpo.json"

model:
  sft_checkpoint: "outputs/checkpoints/sft_cot/best"

dpo:
  beta: 0.1
  loss_type: "sigmoid"

training:
  output_dir: "outputs/checkpoints/dpo"
  num_train_epochs: 2
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5e-5
```

**`configs/grpo.yaml`**：
```yaml
inherit: base

data:
  train_path: "data/raw/train.json"    # GRPO 用原始数据即可

model:
  sft_checkpoint: "outputs/checkpoints/sft_cot/best"

grpo:
  num_generations: 4                   # 每个 prompt 采样数（显存友好）
  max_new_tokens: 512
  temperature: 0.7

reward:
  format_weight: 0.2                   # 格式奖励权重
  correctness_weight: 1.0              # 正确性奖励权重

training:
  output_dir: "outputs/checkpoints/grpo"
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1e-5
```

---

## 四、技术栈与框架选择

### 推荐方案

| 组件 | 选择 | 理由 |
|------|------|------|
| 训练框架 | **trl** (HuggingFace) | DPOTrainer、GRPOTrainer 原生支持，与 transformers/peft 无缝集成 |
| SFT | transformers Trainer + peft | 与 baseline 一致，改动最小 |
| DPO | trl.DPOTrainer | 官方示例直接用 Qwen2-0.5B |
| GRPO | trl.GRPOTrainer | open-r1 使用的就是 trl |
| 日志 | **wandb** | 行业标准，支持实验对比、超参搜索 |
| 配置 | **OmegaConf / PyYAML** | 支持 YAML 继承与覆盖 |
| 数据生成 | **DeepSeek V4 API** | 性价比高，中文数学能力强 |

### 为什么选 trl 而非 LLaMA-Factory
- trl 更轻量，代码透明度高，适合课程报告中解释原理
- DPOTrainer 和 GRPOTrainer 文档完善，示例可直接参考
- 与现有 baseline 代码风格一致（都是 transformers 生态）
- LLaMA-Factory 更适合生产环境，但对课程作业来说过于"黑盒"

---

## 五、答案提取策略（推荐）

推荐采用**正则表达式 + 多级回退**：

```
优先级：
1. 匹配 <answer>数字</answer> 标签
2. 匹配"答案是/为/：" 后的数字
3. 匹配文本中最后一个数字（含小数/分数）
4. 如果以上都失败，返回原始输出（去除非数字字符）
```

原因：
- 对 0.5B 小模型，正则已足够鲁棒
- 无需额外训练提取模型，节省计算资源
- 在报告中可以统计各级回退的命中率，作为分析点

---

## 六、实验记录与报告要求

### wandb 实验追踪
每次训练记录以下信息：
- 方案名称（experiment tag）
- 训练损失曲线
- 验证集正确率曲线（从 train 中划出 10% 作验证集）
- 超参配置（自动从 YAML 同步）
- 最终测试集正确率（提交后回填）

### 课程报告结构建议
1. **问题描述**：任务定义、数据分析
2. **方案设计**：四个方案的原理与实现
3. **实验结果**：
   - 方案对比表（正确率）
   - 消融实验（LoRA 秩、学习率、CoT 长度、GRPO 采样数等）
   - 错误分析（按题目类型分类：四则运算、分数、几何、行程问题等）
4. **分析与讨论**：各方案优劣、0.5B 模型的能力边界
5. **总结**

---

## 七、两周时间线建议

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1-2 | 项目重构、环境搭建、跑通 baseline | 项目结构、baseline submit.csv |
| Day 3-4 | 方案1 CoT 提示工程 + 方案2 数据构建（API 调用） | CoT prompt 实验、train_cot.json |
| Day 5-6 | 方案2 CoT SFT 训练 | sft_cot checkpoint |
| Day 7-8 | 方案3 DPO 数据准备 + 训练 | dpo checkpoint |
| Day 9-10 | 方案4 GRPO 训练 | grpo checkpoint |
| Day 11-12 | 全部方案推理提交 + 超参调优 | 所有 submit.csv |
| Day 13 | 错误分析 + 消融实验 | 分析 notebook |
| Day 14 | 撰写报告 + 整理代码 | 最终交付 |

---

## 八、其他进阶方案（可选，如时间允许）

1. **Self-Consistency（自一致性）**：推理时采样多条路径，投票选择最频繁的答案
2. **表达式树方法**：参考 [MWP-NAS](https://arxiv.org/abs/2305.04556)，让模型生成数学表达式而非直接算数字，用 Python eval 求值
3. **Rejection Sampling + SFT**：多次采样取正确的样本重新做 SFT（类似 STaR）
4. **模型集成**：对多个 checkpoint/方案的输出做投票

---

## 九、代码规范

1. **类型注解**：所有函数参数和返回值使用 type hints
2. **Docstring**：每个模块和关键函数要有中文注释
3. **配置解耦**：硬编码参数一律提取到 YAML，代码中通过 `config.xxx` 访问
4. **可复现性**：固定 seed、记录完整超参、wandb 自动同步
5. **Git 规范**：
   - 每个方案一个 feature branch
   - commit message 格式：`[方案X] 简短描述`
   - `.gitignore` 排除模型权重、checkpoint、wandb 缓存
6. **数据不入库**：`data/raw/` 通过 README 说明获取方式，不上传大文件
