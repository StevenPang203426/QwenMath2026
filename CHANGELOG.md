# CHANGELOG — 小学数学应用题自动解题

本文档记录项目的每次重要迭代，遵循 [Keep a Changelog](https://keepachangelog.com/) 规范。
版本号格式：`vX.Y.Z`（主版本.功能版本.补丁版本）

---

## [v0.1.1] - 2026-05-23 — 工具链切换与数据构建测试模式

### Changed（变更）
- 包管理从 pip 切换至 **uv**（清华镜像源），README 更新安装指南
- `data_builder.py` 新增 `--limit` 参数，支持小批量测试（默认 20 条）
- `run_data_build.sh` 支持参数控制：默认 20 条测试，`all` 全量生成
- README 新增模型下载说明和数据构建测试示例

---

## [v0.1.0] - 2026-05-23 — 项目初始化与架构搭建

### Added（新增）
- 完整项目目录结构：`configs/`, `src/`, `scripts/`, `notebooks/`, `outputs/`, `report/`
- YAML 配置管理系统（支持继承与命令行覆盖）
  - `configs/base.yaml` — 公共基础配置
  - `configs/sft_baseline.yaml` — Baseline SFT 配置
  - `configs/sft_cot.yaml` — CoT SFT 配置
  - `configs/dpo.yaml` — DPO 配置
  - `configs/grpo.yaml` — GRPO 配置
  - `configs/infer.yaml` — 统一推理配置
- 核心代码模块 `src/`
  - `src/utils/` — 配置加载、随机种子、评测指标、wandb 日志
  - `src/data/` — Dataset 类、答案提取器、CoT 数据构建（DeepSeek API）、数据增强、预处理
  - `src/models/` — 统一模型加载（base/LoRA/merge）、GRPO 奖励函数
  - `src/training/` — SFT Trainer、DPO Trainer、GRPO Trainer
  - `src/inference/` — 统一推理接口、CoT 提示工程（zero-shot/few-shot）、批量推理
- Shell 一键运行脚本 `scripts/`
  - 各方案独立脚本 + 全流程自动化脚本
- 项目辅助文件：`requirements.txt`, `.gitignore`, `README.md`
- 原始数据迁移至 `data/raw/`（train.json: 11999条, test.json: 8000条）
- 本 CHANGELOG 文档

### 技术决策
- 训练框架选择 **trl**（HuggingFace），DPO/GRPO 原生支持
- 日志从 swanlab 切换至 **wandb**
- 答案提取采用多级正则回退策略
- CoT 格式统一为 `<think>...</think><answer>...</answer>`

### 参考项目
- Baseline: [AI-FDU/Math_Solver](https://github.com/AI-FDU/Math_Solver)
- CoT: [QwenLM/Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)
- DPO: [huggingface/trl](https://github.com/huggingface/trl)
- GRPO: [huggingface/open-r1](https://github.com/huggingface/open-r1)

---

## [Unreleased] — 待完成事项

### TODO
- [ ] 环境验证：在 AutoDL 上跑通 baseline
- [ ] 方案1：CoT 提示工程实验（zero-shot / few-shot 对比）
- [ ] 方案2：调用 DeepSeek V4 API 生成 CoT 数据
- [ ] 方案2：CoT SFT 训练
- [ ] 方案3：DPO 偏好数据构建 + 训练
- [ ] 方案4：GRPO 奖励函数调优 + 训练
- [ ] 全部方案推理提交
- [ ] 消融实验（LoRA 秩、学习率、CoT 长度等）
- [ ] 错误分析（按题目类型分类）
- [ ] 实验报告撰写

---

## 版本记录模板

<!-- 每次迭代复制此模板填写 -->
<!--
## [vX.Y.Z] - YYYY-MM-DD — 简短标题

### Added（新增）
- 新增了什么功能/模块

### Changed（变更）
- 修改了什么行为/配置

### Fixed（修复）
- 修复了什么 bug

### Experiment（实验记录）
- 方案: xxx
- 超参: lr=xxx, epochs=xxx, ...
- 结果: 正确率 xx.xx%
- wandb run: [链接]
- 分析: ...
-->
