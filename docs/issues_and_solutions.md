# 问题与解决方案记录

> 项目：CCF BDCI 小学数学应用题自动解题
> 更新日期：2026-05-28

---

## 1. 文件截断问题（data_builder.py）

**现象：** data_builder.py 多次出现文件末尾被截断，语法错误。

**原因：** 文件包含大量中文字符，通过编辑工具写入时在多字节 UTF-8 边界处截断。

**解决：** 使用 bash heredoc 或 Python 脚本直接写入。每次写入后用 ast.parse 验证语法。

---

## 2. Baseline 推理找不到 checkpoint

**现象：** run_baseline.sh 推理报错 adapter_config.json not found at best。

**原因：** Trainer 保存为 checkpoint-{step}，不会自动创建 best 目录。

**解决：** batch_infer.py 新增 _resolve_checkpoint_path()，自动查找最新 checkpoint-* 目录。

---

## 3. AutoDL 模型下载超时

**现象：** 模型加载卡住，无法下载 Qwen2.5-0.5B-Instruct。

**原因：** AutoDL 网络代理拦截了下载请求。

**解决：** 配置中改为绝对路径 /root/autodl-tmp/steven/Math/QwenMath2026/model_cache/Qwen/Qwen2.5-0.5B-Instruct。

---

## 4. DeepSeek API 代理冲突

**现象：** 调用 API 超时，连接被拒绝。

**原因：** AutoDL 的 http_proxy 环境变量拦截了 api.deepseek.com 请求。

**解决：** 运行前 unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY。

---

## 5. DeepSeek API 401 + 模型废弃

**现象：** API 返回 401，deepseek-chat 模型即将废弃。

**原因：** 原代码用 requests 手动拼 HTTP，认证格式不对。

**解决：** 重写为 OpenAI SDK：from openai import OpenAI，base_url="https://api.deepseek.com"，模型改为 deepseek-v4-pro。

---

## 6. 正则无法匹配分数答案

**现象：** API 返回"答案：3/5"，提取结果为"3"，丢失"/5"。

**原因：** 正则 -?\d+\.?\d* 不支持分数。

**解决：** 扩展为 -?\d+\.?\d*(?:/\d+\.?\d*)?%?，同步修改 data_builder.py、answer_extractor.py、metrics.py。

---

## 7. 分数与小数等价判断缺失

**现象：** api_answer="3/5" 与 answer="0.6" 被判定为不匹配。

**原因：** 字符串直接比较。

**解决：** 新增 _normalize_answer() 统一转 float，_answers_match() 做容差比较（tol=1e-6）。

---

## 8. answer_match=False 无后续处理

**需求：** 第一轮 API 答案不匹配或输出为空，需要自动重试。

**解决：** 两轮处理机制。第一轮正常并行，第二轮对失败条目降低温度(0.1)重试。每条结果新增 status 字段（ok/api_failed/parse_error:xxx）。

---

## 9. 错误推理生成不自然

**需求：** 原"故意犯错"prompt 生成的错误太明显。

**解决：** 实现 SCDPO 风格错误推理生成：_split_cot_steps() 拆分正确推理为步骤，_generate_scdpo_wrong() 随机选中间步骤注入错误，保留前面正确步骤，拼接正确前半段+错误后半段。步骤太少时回退原始方式。

---

## 10. 百分数答案未完整提取

**需求：** 10% 应完整提取，不是只取 10。

**解决：** 正则末尾加 %?，_clean_answer() 百分数保留原样，_normalize_answer() 取 % 前数值。

---

## 11. 错误推理未过滤"算对了"的情况

**现象：** `--generate_wrong` 生成的错误推理中，部分 wrong_answer 与正确答案完全相同。

**原因：** 生成后没有调用 `_answers_match()` 校验，模型虽被要求犯错但仍可能给出正确答案。

**解决：** 错误推理生成后增加 `_answers_match` 过滤，匹配（即答对了）则丢弃并重试，最多 3 次。引入 `wrong_status` 字段追踪来源（simple_try1/2/3, hard_fallback, scdpo, failed）。

---

## 12. 小批量测试污染生产数据

**现象：** `--limit 10` 测试时输出写入 `train_cot_raw.json`，断点续传加载了旧数据导致测试不可靠。

**原因：** 测试和生产共用同一输出路径和断点续传逻辑。

**解决：** `limit > 0` 时自动切换输出到 `data/processed/test/test_{limit}.json`，关闭断点续传。保存频率动态调整：`save_every = min(100, max(10, len(todo) // 5))`。

---

## 13. 错误推理 prompt 太弱，成功率仅 10%

**现象：** 纯角色扮演 prompt（"你是小明，三年级学生"）+ temperature=0.8，20 条中只有 2 条生成了错误答案。

**原因：** 模型推理能力太强，角色扮演不足以让它犯错。

**解决：** 采用两层 prompt 策略：
1. `_PROMPT_WRONG`（主力）：明确要求"故意犯一个合理的计算错误"，约束"不要用'故意''错误地'等词"保持自然。
2. `_PROMPT_WRONG_HARD`（兜底）：3 次都算对时启用，告知正确答案要求给出不同答案。

成功率从 10% 提升到 60%+。

---

## 14. DeepSeek Thinking 模式导致温度参数失效

**现象：** 设置 `temperature=0.8` 或 `1.3`，但输出多样性无明显变化。

**原因：** DeepSeek V4 Flash 默认开启 thinking 模式，该模式下 `temperature` 和 `top_p` 被静默忽略。

**解决：** 保留 thinking 模式（对推理质量有益），接受温度不可控的现实。如需禁用可加 `extra_body={"thinking": {"type": "disabled"}}`。

**参考：** https://api-docs.deepseek.com/zh-cn/

---

## 15. Thinking 模式要求内容块格式

**现象：** API 返回 400：`expected struct ChatCompletionRequestContentBlock`。

**原因：** Thinking 模式开启时，DeepSeek 要求 `messages[].content` 为内容块格式 `[{"type": "text", "text": "..."}]`，不能是纯字符串。

**解决：** 将 user message 改为内容块格式：
```python
{"role": "user", "content": [{"type": "text", "text": question}]}
```
同时将 `max_tokens` 从 1024 提升到 8192 以适配 thinking 模式更长的输出。

---

## 16. Question 字段类型不一致（str vs list）

**现象：** `_sanitize_question()` 报错 `AttributeError: 'list' object has no attribute 'replace'`。

**原因：** 原始数据中部分 question 因含引号被 JSON 解析为 list（如 `['"某小学...', '一班...?"']`），而非预期的 str。

**解决：** `_sanitize_question()` 增加类型判断，list 先 `"".join()` 拼回字符串，再清理首尾残留引号和转义字符。

---

## 17. Git index.lock 阻塞操作

**现象：** `git add` 报错 `fatal: Unable to create '.git/index.lock': File exists`。

**原因：** 之前的 git 操作异常退出遗留锁文件。

**解决：** 手动删除：`del .git\index.lock`（Windows）或 `rm .git/index.lock`（Linux）。

---

## 18. DeepSeek API 上下文缓存机制与优化

**发现：** DeepSeek V4 API 内置自动前缀缓存，缓存命中价格仅为未命中的 1/50（$0.0028 vs $0.14/M tokens）。

**问题：** 初始缓存命中率仅 0.2%，因为正确推理和错误推理交替发送，system_prompt 不断切换导致缓存失效。

**解决：** 按 prompt 类型分阶段批处理（方案 B）：
- Phase 1: 全部正确推理（同一 `_PROMPT_CORRECT`）
- Phase 2: 全部 simple 错误推理（同一 `_PROMPT_WRONG`）
- Phase 3: 全部 hard_fallback（`_PROMPT_WRONG_HARD`）

**效果：** 缓存命中率从 0.2% 提升至 **52%**，API 成本大幅降低。

---

## 19. 新增独立失败数据修复脚本 data_repair.py

**需求：** 全量生成后需要修复 api_failed 和 wrong_status=failed 的条目，不想重跑全量。

**设计：**
- 幂等：每次从 `train_cot_raw.json` 扫描失败条目，修复后原地写回，可反复运行
- Type A（api_failed/parse_error）：Phase 1 重跑正确推理 → Phase 2 simple 错误推理
- Type B（wrong_status=failed）：直接跳到 Phase 3 hard_fallback（上一轮 simple 已失败过，不再浪费）
- 支持 `--limit N` 小批量测试，输出到 `data/processed/test/repair_test_N.json`
- 复用 `data_builder.py` 的工具函数，不重复造轮子

**用法：**
```bash
python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY"            # 全量修复
python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY" --limit 5  # 测试
```

---

## 20. 错误推理 prompt 优化：剔除错误原因描述

**现象：** 模型在错误推理中会写"误将""不对""等等"之类的词，暴露了故意犯错的意图，不够自然。

**解决：** 在 `_PROMPT_WRONG_HARD` 中增加明确约束：
```
不要用"故意""但是""错误地""误将""不对""等等"之类的词。不准写错误原因。
```

**效果：** 错误推理更加自然，读起来像真实的学生计算错误，没有元认知痕迹。

---

## 21. LaTeX cot 检测与降级

**现象：** 部分正确推理（cot）中包含 LaTeX 格式（如 `\frac`, `\times`, `\pi`），Qwen2.5-0.5B 无法正确学习这种格式。

**解决：** 在 `data_repair.py` 预处理阶段 0a 中：
- 使用 `re.compile(r'\\[a-zA-Z]')` 检测 LaTeX 命令
- 含 LaTeX 的 cot 降级为 wrong_cot（作为 DPO 负样本）
- 清空 cot 和 api_answer，标记为需要重跑正确推理
- 更新 `_PROMPT_CORRECT` 明确禁止 LaTeX 格式

---

## 22. answer_match=false 强制修正与删除

**现象：** 部分条目 answer_match=false，但同时有 api_answer 和 wrong_answer。

**解决：** 在 `data_repair.py` 预处理阶段 0b 中：
- 如果 `api_answer == wrong_answer`（两次算出同一答案）→ 强制将 `answer` 修正为该共识答案
- 如果 `api_answer != wrong_answer`（数据不可靠）→ 从 JSON 数组中彻底移除该记录

---

## 23. Type B 路由优化：区分 fresh 与 failed

**现象：** 重新筛选后，type_b 错误地将从未尝试过 simple 错误推理的"新鲜"条目直接送入 Phase 3 (hard_fallback)，跳过了 3 轮 simple 尝试。

**原因：** 从"按字段值筛选"改为"按实际数据状态筛选"后，type_b 捕获了所有"已匹配但无错误推理"的条目。

**解决：** 将 type_b 拆分为：
- `type_b_fresh`：无 wrong_status 或 wrong_status 非 failed → 先走 Phase 2 (simple)
- `type_b_failed`：wrong_status=failed → 直接走 Phase 3 (hard)

同时修复小批量测试模式的 limit 在预处理重新筛选后丢失的问题。

---

## 24. Type C 回收：错误正采样作为 DPO 负样本

**需求：** 正采样结果 answer_match=false（API 算错了），这些错误推理可以直接作为 DPO 负样本使用，无需浪费。

**设计：**
- 质量门槛：cot 非空、api_answer 非空、cot 长度 ≥ 10
- 达标的旧 cot/api_answer 保存为 wrong_cot/wrong_answer，wrong_status="recycled"
- 然后重跑正确推理（合并进 Phase 1）
- 已回收负样本的条目跳过 Phase 2（不再重复生成错误推理）

**额外检测 — ground_truth_suspect：**
- 如果重跑后 API 再次给出与回收的 wrong_answer 相同的答案（模型两次算出同一结果，但与标注不同），标记 `wrong_status="ground_truth_suspect"`，清除回收的负样本
- 这类条目大概率是标注错误，需要人工审查

---

## 25. 按实际数据状态筛选，不依赖特定字段值

**现象：** 部分 JSON 条目缺少 `status`、`wrong_status` 等字段（如早期生成的数据），导致基于字段值的筛选（`status == "api_failed"`）遗漏这些条目。

**解决：** 所有分类逻辑改为检查实际数据状态：
```python
has_cot = bool(d.get("cot", "").strip())
has_api_answer = bool(d.get("api_answer", "").strip())
matched = d.get("answer_match", False)
has_wrong = bool(d.get("wrong_cot", "").strip())
```
不再依赖 `status`、`wrong_status` 的具体值来决定是否需要修复。

---

## 26. _PROMPT_CORRECT 优化：提升 CoT 质量与可学习性

**需求：** 为 Qwen2.5-0.5B 生成更易学习的细粒度推理链。

**优化项：**
- LaTeX 禁令从单一示例（frac）扩展为通用规则（`\frac`、`\times`、`\sqrt`、`\pi` 等所有反斜杠命令）
- 明确运算符号规范：乘号写×，除号写÷，分数写 a/b
- 答案形式规则细化：百分率 → 25% 形式；分率 → 3/5 形式；能整除给整数，不能整除保留小数或分数
- 新增推理粒度约束："每步只做一个运算，不要跳步，不要合并多步计算"
- 用【格式要求】【计算规则】【推理要求】分区，结构更清晰

---

## 27. 缺少 wrong_cot/wrong_answer 字段的条目处理

**现象：** 部分 JSON 条目完全不含 `wrong_cot`、`wrong_answer` 字段（非空字符串，而是字段本身缺失）。

**原因：** 早期 data_builder.py 生成时未写入这些字段，或中途中断。

**解决：** `d.get("wrong_cot", "")` 对缺失字段返回空字符串，`has_wrong` 判定为 False。这些条目会被分类为：
- `type_b_fresh`（如果正确推理已匹配）→ Phase 2 (simple) → Phase 3 (hard)
- `type_c`（如果正确推理不匹配）→ 回收 + 重跑

无需特殊处理，通用逻辑已覆盖。

---

## 28. DPO 数据转换 ArrowInvalid：list 与 str 混用

**现象：** 运行 `bash scripts/run_dpo.sh` 时，DPO 数据加载阶段报错：
```text
pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values
```

**原因：** `Dataset.from_dict()` 会通过 PyArrow 推断列类型，同一列或嵌套字段中不能同时出现 list 和 str。远端 `train_dpo.json` 中部分字段可能来自 OpenAI/DeepSeek 内容块格式，例如 `question=[{"type":"text","text":"..."}]`，而其他样本是普通字符串。`dpo_trainer.py` 又把 `question` 放入 `prompt[1]["content"]`，导致 `prompt.content` 中混入 list 和 str。

**解决：** 新增 `_to_text()` 文本归一化函数，在 DPO 加载和 DPO 数据生成阶段统一把 `question`、`instruction`、`chosen`、`rejected` 转为字符串。

**修改位置：**
- `src/training/dpo_trainer.py`：`_load_dpo_dataset()` 中调用 `_to_text()`，避免 Arrow 类型推断失败
- `src/data/preprocessor.py`：`prepare_dpo_data()` 输出前将 `question` 归一化，防止后续生成的新数据再次混入 list

**排查命令：**
```bash
python - <<'PY'
import json, collections
p = "data/processed/train_dpo.json"
data = json.load(open(p, encoding="utf-8"))
for f in ["question", "instruction", "chosen", "rejected"]:
    c = collections.Counter(type(x.get(f)).__name__ for x in data)
    print(f, c)
PY
```

---

## 29. TRL DPOConfig 参数版本不兼容

**现象：** DPO 数据加载成功后，构造 `DPOConfig` 时报错：
```text
TypeError: DPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'
```

**原因：** 不同版本 TRL 的 `DPOConfig` API 不一致。当前环境中的 `DPOConfig` 不接受 `max_prompt_length`，而代码固定传入该参数，导致初始化失败。

**解决：** 新增 `_filter_supported_kwargs()`，通过 `inspect.signature()` 检查当前安装版本实际支持的参数，只传入支持的 kwargs。`DPOTrainer` 同样做兼容处理：新版使用 `processing_class=tokenizer`，旧版使用 `tokenizer=tokenizer`。

**修改位置：** `src/training/dpo_trainer.py`

**版本检查命令：**
```bash
python - <<'PY'
import trl, inspect
from trl import DPOConfig, DPOTrainer

print("trl version:", trl.__version__)
print("DPOConfig:", inspect.signature(DPOConfig))
print("DPOTrainer:", inspect.signature(DPOTrainer.__init__))
PY
```

---

## 30. TRL Tokenizing 阶段 prompt/list 与 chosen/str 拼接错误

**现象：** `DPOTrainer` 初始化进入 tokenizing 阶段后报错：
```text
TypeError: can only concatenate list (not "str") to list
```

**原因：** 当前 TRL 内部 tokenizing 逻辑会执行：
```python
example["prompt"] + example["chosen"]
```
当 `prompt` 是 chat message list，而 `chosen` 是普通字符串时，实际变成 `list + str`，无法拼接。

**解决：** 在 `_load_dpo_dataset()` 中提前使用 tokenizer 的 chat template 将 system/user prompt 渲染为字符串：
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```
这样传给 TRL 的 `prompt`、`chosen`、`rejected` 三列都是字符串，内部 `prompt + chosen` 可以正常执行。

**修改位置：** `src/training/dpo_trainer.py`，`_load_dpo_dataset(data_path, tokenizer)` 接收 tokenizer 并渲染 prompt 字符串。

**验证：** 修改后至少通过语法检查：
```bash
python -m py_compile src/training/dpo_trainer.py src/data/preprocessor.py
```

---

## 31. GRPO 数据转换 ArrowInvalid：与 DPO 相同的字段类型混用

**现象：** 运行 GRPO 训练时，加载原始训练集并构造 HuggingFace Dataset 报错：
```text
pyarrow.lib.ArrowInvalid: cannot mix list and non-list, non-null values
```

**原因：** `grpo_trainer.py` 原先直接把 `item["question"]` 放入 chat message：
```python
{"role": "user", "content": item["question"]}
```
当原始数据中部分 `question` 是内容块 list、部分是普通字符串时，`Dataset.from_dict()` 在推断 `prompt.content` 类型时失败。该问题与 DPO 的 ArrowInvalid 本质相同。

**解决：** 抽取 DPO/GRPO 共用 RL 训练工具模块 `src/training/rl_utils.py`：
- `to_text()`：统一将 str/list/dict/None 转为字符串
- `render_chat_prompt()`：用 tokenizer 的 chat template 将 messages 渲染为字符串 prompt
- `filter_supported_kwargs()`：过滤当前 TRL 版本不支持的 Config/Trainer 参数
- `add_tokenizer_kwarg()`：兼容 `processing_class=tokenizer` 与旧版 `tokenizer=tokenizer`

**修改位置：**
- `src/training/rl_utils.py`：新增共用工具函数
- `src/training/dpo_trainer.py`：移除重复 `_to_text()` / `_filter_supported_kwargs()`，改用共用模块
- `src/training/grpo_trainer.py`：`_build_grpo_dataset(data_path, tokenizer)` 中将 `question` 归一化，并把 chat messages 渲染为字符串 prompt

**验证：**
```bash
python -m py_compile src/training/rl_utils.py src/training/dpo_trainer.py src/training/grpo_trainer.py
```

---

## 32. GRPO 五维奖励函数重新设计

**背景：** 原 GRPO 奖励函数仅有二值正确性 + 格式匹配（`combined_reward`），对 0.5B 小模型来说奖励信号过于稀疏，策略梯度方差大、训练不稳定。参考 DeepSeek-R1、GRPO-LEAD（EMNLP 2025）等工作，重新设计为五维独立奖励函数。

**设计原则：**
- 正确答案的最低总分 > 错误答案的最高总分（正确排序保证）
- 负惩罚打破奖励稀疏：错误/无法解析 -0.5，无逻辑词 -0.3，LaTeX -0.3
- 5 个独立函数，TRL GRPOTrainer 自动求和 + 分维度日志

**五维奖励体系（理论总分 -1.4 ~ +2.15）：**

| 维度 | 函数 | 范围 | 说明 |
|------|------|------|------|
| R1 正确性 | `correctness_reward_fn` | -0.5 ~ +1.0 | 精确匹配 +1.0，四舍五入等价 +0.3，错误/无法解析 -0.5 |
| R2 格式标签 | `format_reward_fn` | 0.0 ~ +0.2 | `<think></think><answer></answer>` 全有 +0.2 |
| R3 逻辑词 | `logic_word_reward_fn` | -0.3 ~ +0.95 | 基础组 ×0.05 + 结构组 ×0.10，按种类数计分，0 命中 -0.3 |
| R4 长度正则 | `length_reward_fn` | -0.3 ~ 0.0 | <30 字符 -0.3，>300 字符 -0.2，合理范围 0.0 |
| R5 无 LaTeX | `no_latex_reward_fn` | -0.3 ~ 0.0 | 含 LaTeX 命令 -0.3，否则 0.0 |

**关键设计点：**
- **四舍五入等价**：`_round_match()` 将预测和标准答案 round 到较少的小数位数比较；仅在题目不含"保留""精确到"关键词时启用
- **逻辑词分组**：基础组（即/所以/因此/得到/那么）高频但信息量低，结构组（首先/已知/题意/设/因为/由于/根据）低频但信息量高
- **长度阈值数据驱动**：基于 train_cot 正样本统计（min=21, p10=48, median=79, p90=145）设定

**配置变更（grpo.yaml）：**
- `num_generations`: 4 → 16（28GB 显存 + 0.5B 模型，保证 advantage 稳定）
- `temperature`: 0.7 → 0.8（增加探索多样性）
- `learning_rate`: 1.0e-5 → 5.0e-6（保守防退化）
- `max_grad_norm`: 新增 0.1（激进梯度裁剪）

**修改位置：**
- `src/models/reward.py`：完全重写，5 个独立奖励函数 + `build_reward_funcs()`
- `src/training/grpo_trainer.py`：`_make_reward_funcs()` 改用 `build_reward_funcs()`
- `configs/grpo.yaml`：更新训练超参数，移除旧 reward 配置节

**验证：**
```bash
python -c "import ast; ast.parse(open('src/models/reward.py').read()); print('OK')"
python -c "import ast; ast.parse(open('src/training/grpo_trainer.py').read()); print('OK')"
python -c "import yaml; yaml.safe_load(open('configs/grpo.yaml')); print('OK')"
```

---

## 待解决 / 后续计划

| 编号 | 事项 | 状态 |
|------|------|------|
| 1 | ~~DeepSeek API 小批量测试~~ | ✅ 已完成 |
| 2 | 全量 CoT 数据生成（12000条） | 进行中 |
| 3 | ~~换 deepseek-v4-flash 降成本~~ | ✅ 已切换 |
| 4 | 考虑 SiliconFlow 免费 API（Qwen3-8B） | 待决策 |
| 5 | CoT SFT 训练 | 待执行 |
| 6 | DPO 训练（偏好对数据） | 待执行 |
| 7 | GRPO 训练 | 待执行 |
| 8 | 全部方案推理+提交 | 待执行 |
| 9 | 课程报告撰写 | 待执行 |
| 10 | 失败数据修复（data_repair.py） | 进行中 |
