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
