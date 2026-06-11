# 小学数学应用题自动解题

CCF BDCI × 题拍拍 — 小学数学应用题自动解题课程实践项目。

输入小学 1-6 年级数学应用题，模型输出对应的数字答案。推理模型限制为 **Qwen2.5-0.5B-Instruct**。

## 方案概览

| 方案 | 方法 | 说明 |
|------|------|------|
| Baseline | LoRA SFT | 直接预测数字答案 |
| 方案1 | CoT 提示工程 | 不微调，纯 prompt 优化 |
| 方案2 | 数据构建 + CoT SFT | DeepSeek API 生成推理步骤，再微调 |
| 方案3 | DPO | 正/误偏好对齐 |
| 方案4 | GRPO | 组相对策略优化（DeepSeek-R1 风格） |

## 快速开始

### 环境安装（uv）

本项目使用 [uv](https://docs.astral.sh/uv/) 管理依赖，比 pip 快 10-100 倍。

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 初始化项目并安装依赖（使用清华镜像源）
uv init --no-readme
uv add torch transformers peft trl datasets \
       modelscope pyyaml omegaconf wandb \
       numpy pandas tqdm requests \
       matplotlib seaborn ipython jupyter \
       --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

如果 `uv` 不可用，也可用 pip 安装：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 模型下载

首次运行训练/推理时会**自动下载** Qwen2.5-0.5B-Instruct（~1GB）到 `./model_cache`。
也可以提前手动下载：

```bash
# ModelScope（国内推荐）
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct', cache_dir='./model_cache')"
```

### 数据准备

```bash
# 原始数据已在 data/raw/ 中（来自 https://github.com/AI-FDU/Math_Solver）
ls data/raw/train.json data/raw/test.json
```

### 运行实验

```bash
# 统一入口（推荐）
bash scripts/data.sh cot-build 20       # CoT 数据构建小批量，需要 DEEPSEEK_API_KEY
bash scripts/train.sh sft cot           # CoT SFT
bash scripts/train.sh dpo cot           # CoT DPO
bash scripts/train.sh grpo cot legacy   # CoT GRPO

# 推理 / 提交
bash scripts/submit.sh infer sft_cot    # 可选: baseline | cot_prompt | sft_cot | dpo | grpo
bash scripts/submit.sh expr4            # 当前表达式投票提交入口

# 旧 run_*.sh 仍是兼容 wrapper，新实验优先使用上面的统一入口。
```

### 数据构建

调用 DeepSeek V4 Flash API 生成 CoT 推理数据，支持断点续传。

```bash
export DEEPSEEK_API_KEY=your_key

# 小批量测试（默认 20 条，验证 API 调用和答案匹配率）
bash scripts/data.sh cot-build          # 20 条
bash scripts/data.sh cot-build 50       # 50 条
bash scripts/data.sh cot-build all      # 全量 12000 条

# 等价的 python 命令（更多参数控制）
python -m src.data.data_builder \
    --input data/raw/train.json \
    --output data/processed/train_cot_raw.json \
    --api_key "$DEEPSEEK_API_KEY" \
    --model deepseek-v4-flash \
    --workers 4 \
    --limit 20
```

#### 生成错误推理（DPO 用）

添加 `--generate_wrong` 即可同时生成错误推理路径，用于 DPO 偏好训练。
错误推理有两种生成方式，通过 `--wrong_method` 选择：

```bash
# simple（默认）：以"粗心小学生"角色自然犯错，不额外增加 API 请求
python -m src.data.data_builder \
    --input data/raw/train.json \
    --output data/processed/train_cot_raw.json \
    --api_key "$DEEPSEEK_API_KEY" \
    --generate_wrong \
    --wrong_method simple

# scdpo：Step-Controlled DPO，保留正确前半段 + 注入错误后半段
# 可精确控制错误出现的步骤位置，但会多消耗 input token
python -m src.data.data_builder \
    --input data/raw/train.json \
    --output data/processed/train_cot_raw.json \
    --api_key "$DEEPSEEK_API_KEY" \
    --generate_wrong \
    --wrong_method scdpo
```

#### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `deepseek-v4-flash` | API 模型（也可用 `deepseek-v4-pro`） |
| `--workers` | `4` | 并行线程数 |
| `--limit` | `0`（全部） | 限制处理条数，用于测试 |
| `--generate_wrong` | 关闭 | 同时生成错误推理（DPO 偏好对） |
| `--wrong_method` | `simple` | 错误推理方式：`simple`（推荐）或 `scdpo` |

### 数据修复、格式修复与增强清洗

数据治理流程分为两类修复和一类增强审计：

- 题目级修复：处理 OCR、漏符号、题目自身错误、标注答案错误等问题。
- 格式修复：处理 `expr_correct.jsonl` 中表达式列式正确，但结果未按题意输出百分数、分数、保留小数或离散取整格式的问题。
- 增强审计：检查增强题干和增强表达式是否都能由同一个 `source_id + changed_numbers` 同步替换得到；不把“表达式值等于答案”当成题干一致性的证据。

```bash
# 1. 题目级质量审计与自动修复，需要 DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY=your_key
bash scripts/data.sh quality-audit all

# 2. 表达式答案格式修复，不调用 API
bash scripts/data.sh format-repair

# 3. 规则增强 + 增强数据对应性审计
bash scripts/data.sh augment

# 4. 合并 raw_ok、auto_repair、expr_format_repair、clean rule_augment
bash scripts/data.sh clean-merge
```

主要产物：

| 产物 | 路径 |
|------|------|
| 质量候选 | `data/processed/intermediate/quality/quality_candidates.json` |
| 质量审计 | `data/processed/intermediate/quality/quality_audit.json` |
| 题目级修复 | `data/processed/train_repaired.json` |
| 格式修复 | `data/processed/intermediate/format_repair/expr_format_repaired.json` |
| 格式修复报告 | `data/processed/intermediate/format_repair/format_repair_report.json` |
| clean 增强数据 | `data/processed/intermediate/augmentation/train_augmented_clean.json` |
| 增强审计报告 | `data/processed/intermediate/augmentation/augmentation_report.json` |
| 统一修复数据 | `data/processed/train_repairs_unified.json` |
| 最终清洗增强训练集 | `data/processed/train_clean_augmented.json` |
| 最终合并统计报告 | `data/processed/train_clean_augmented_report.json` |

当前已执行合并结果：

| 来源 | 含义 | 条数 | 占比 |
|------|------|------|------|
| `raw_ok` | 原始合格数据 | 10599 | 89.13% |
| `auto_repair` | 题意修复数据 | 347 | 2.92% |
| `expr_format_repair` | 表达式结果规范修复数据 | 279 | 2.35% |
| `rule_augment` | 增强后的合格数据 | 666 | 5.60% |
| **合计** | 最终清洗增强训练集 | **11891** | **100.00%** |

统一修复数据共 626 条，其中题意修复 347 条、表达式结果规范修复 279 条。合并时跳过原始坏题/被修复替代题 1356 条，跳过重复修复 5 条，跳过原始重复 id 44 条。

### 表达式路线 SFT / RL

表达式路线使用 clean 数据中可复验的安全表达式子集，训练模型输出 `<expr>...</expr><answer>...</answer>`，最终答案仍以表达式 eval 后题意归一为准。

```bash
# 1. 从 clean 合并数据中过滤安全表达式，并构建固定 train/val split 与 DPO pairs
bash scripts/data.sh expr-training all

# 2. 主线：Expr-SFT → Expr-GRPO
bash scripts/train.sh sft expr clean
bash scripts/train.sh grpo expr clean

# 3. 加速版 GRPO：优先尝试 vLLM，异常时回退 fast
bash scripts/train.sh grpo expr clean-vllm
bash scripts/train.sh grpo expr clean-fast

# 4. 消融：Expr-DPO → Expr-GRPO
bash scripts/train.sh dpo expr clean
bash scripts/train.sh grpo expr from-dpo-clean
bash scripts/train.sh grpo expr from-dpo-clean-vllm
bash scripts/train.sh grpo expr from-dpo-clean-fast
```

`grpo_expr_clean_vllm` 和 `grpo_expr_from_dpo_clean_vllm` 使用 vLLM rollout 加速；当前环境 `vLLM=0.21.0` 超出 TRL 声明的 `0.12.0~0.18.0` 支持范围，但已通过本机 1-step smoke。若需强制版本检查，可设置 `STRICT_VLLM_VERSION=1`。`*_fast` 不使用 vLLM，主要通过 `max_completion_length=96`、关闭训练中 eval、降低保存频率、调整 microbatch 来加速。

当前表达式训练数据：

| 产物 | 路径 | 条数 |
|------|------|------|
| clean 表达式全集 | `data/processed/train_expr_clean.json` | 11478 |
| SFT/GRPO train split | `data/splits/train_expr_clean_train.json` | 10331 |
| SFT/GRPO val split | `data/splits/train_expr_clean_val.json` | 1147 |
| DPO clean pairs | `data/processed/train_expr_dpo_clean.json` | 9095 |

### 配置管理

所有超参通过 `configs/` 分路线管理，支持继承和命令行覆盖：

```bash
# 命令行覆盖示例
python -m src.training.sft_trainer --config configs/cot/sft_cot.yaml --training.learning_rate 1e-5
```

## 项目结构

```
├── configs/          # YAML 配置文件（cot/expr/inference/experiments）
├── data/             # 数据（raw/processed/splits）
├── src/              # 核心代码
│   ├── data/         # 数据处理、答案提取、API 数据构建
│   ├── models/       # 模型加载、奖励函数
│   ├── training/     # SFT / DPO / GRPO 训练器
│   ├── inference/    # 推理、CoT 提示、批量推理
│   └── utils/        # 配置、指标、日志、种子
├── scripts/          # 统一入口 + 兼容 wrapper
├── notebooks/        # 实验分析
├── outputs/          # checkpoint、提交文件、日志
└── report/           # 课程报告
```

## 实验追踪

使用 wandb 追踪实验：

```bash
wandb login
# 所有训练自动上传至 wandb 项目 "math-solver"
```

## 评测

```python
from src.utils.metrics import evaluate_from_files
result = evaluate_from_files("outputs/submissions/submit_sft_cot.csv", "data/raw/train.json")
print(f"正确率: {result['accuracy']:.4f}")
```
