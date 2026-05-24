# 问题与解决方案记录

> 项目：CCF BDCI 小学数学应用题自动解题
> 更新日期：2026-05-24

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

## 待解决 / 后续计划

| 编号 | 事项 | 状态 |
|------|------|------|
| 1 | DeepSeek API 小批量测试（10条） | 待测试 |
| 2 | 全量 CoT 数据生成（12000条） | 待执行 |
| 3 | 考虑换 deepseek-v4-flash 降成本（12倍） | 待决策 |
| 4 | 考虑 SiliconFlow 免费 API（Qwen3-8B） | 待决策 |
| 5 | CoT SFT 训练 | 待执行 |
| 6 | DPO 训练（SCDPO 数据） | 待执行 |
| 7 | GRPO 训练 | 待执行 |
| 8 | 全部方案推理+提交 | 待执行 |
| 9 | 课程报告撰写 | 待执行 |
