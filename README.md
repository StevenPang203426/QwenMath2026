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
# 跑单个方案
bash scripts/run_baseline.sh        # Baseline
bash scripts/run_data_build.sh      # 数据构建（需设置 DEEPSEEK_API_KEY）
bash scripts/run_sft_cot.sh         # CoT SFT
bash scripts/run_dpo.sh             # DPO
bash scripts/run_grpo.sh            # GRPO

# 推理
bash scripts/run_infer.sh sft_cot   # 可选: baseline | cot_prompt | sft_cot | dpo | grpo

# 全流程
bash scripts/run_all_experiments.sh
```

### 数据构建

调用 DeepSeek V4 Flash API 生成 CoT 推理数据，支持断点续传。

```bash
export DEEPSEEK_API_KEY=your_key

# 小批量测试（默认 20 条，验证 API 调用和答案匹配率）
bash scripts/run_data_build.sh          # 20 条
bash scripts/run_data_build.sh 50       # 50 条
bash scripts/run_data_build.sh all      # 全量 12000 条

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

### 配置管理

所有超参通过 `configs/*.yaml` 管理，支持继承和命令行覆盖：

```bash
# 命令行覆盖示例
python -m src.training.sft_trainer --config configs/sft_cot.yaml --training.learning_rate 1e-5
```

## 项目结构

```
├── configs/          # YAML 配置文件
├── data/             # 数据（raw/processed/splits）
├── src/              # 核心代码
│   ├── data/         # 数据处理、答案提取、API 数据构建
│   ├── models/       # 模型加载、奖励函数
│   ├── training/     # SFT / DPO / GRPO 训练器
│   ├── inference/    # 推理、CoT 提示、批量推理
│   └── utils/        # 配置、指标、日志、种子
├── scripts/          # Shell 运行脚本
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
