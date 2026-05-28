"""
CoT 数据构建模块
调用 DeepSeek V4 API（OpenAI SDK 兼容格式）为训练数据生成带推理步骤的答案
同时生成错误推理路径供 DPO 使用
"""
import json
import time
import os
import re
import logging
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("math_solver.data_builder")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"  # deepseek-v4-pro | deepseek-v4-flash

# 数字正则：整数、小数、分数(3/5)、百分数(10%)
_NUM = r'-?\d+\.?\d*(?:/\d+\.?\d*)?%?'

# ============================================================
# 基础工具函数
# ============================================================

def _get_client(api_key: str):
    """获取 OpenAI 兼容客户端"""
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _sanitize_question(q) -> str:
    """清理题目文本中可能导致 API 请求失败的特殊字符"""
    if not q:
        return ""
    if isinstance(q, list):
        q = "".join(q)
    q = q.replace('\\"', '"')       # 转义引号残留
    q = q.replace('\\n', '\n')      # 转义换行残留
    q = q.replace('\\t', ' ')       # 转义制表符
    q = q.strip().strip('"')        # 去除首尾残留引号
    return q


def _normalize_answer(s: str) -> float | None:
    """将分数/小数/百分数统一转 float"""
    if not s:
        return None
    s = s.strip()
    try:
        if s.endswith('%'):
            return float(s[:-1])
        if '/' in s:
            parts = s.split('/')
            if len(parts) == 2:
                n, d = float(parts[0]), float(parts[1])
                return n / d if d != 0 else None
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def _answers_match(a: str, b: str, tol: float = 1e-6) -> bool:
    """判断两答案是否等价（支持 3/5==0.6, 10%==10% 等）"""
    if str(a).strip() == str(b).strip():
        return True
    va, vb = _normalize_answer(str(a)), _normalize_answer(str(b))
    if va is not None and vb is not None:
        return abs(va - vb) < tol
    return False


def _parse_response(content: str) -> dict:
    """解析 API 响应，提取推理过程和答案"""
    if not content or not content.strip():
        return {"cot": "", "answer": "", "parse_error": "empty_response"}

    cot, answer = "", ""
    cot_m = re.search(r'推理过程[：:]\s*(.*?)(?=答案[：:])', content, re.DOTALL)
    ans_m = re.search(rf'答案[：:]\s*({_NUM})', content)

    if cot_m:
        cot = cot_m.group(1).strip()
    elif ans_m:
        cot = content[:ans_m.start()].strip()
    else:
        cot = content.strip()

    if ans_m:
        answer = ans_m.group(1).strip()
    else:
        nums = re.findall(_NUM, content)
        answer = nums[-1] if nums else ""

    result = {"cot": cot, "answer": answer}
    if not answer:
        result["parse_error"] = "no_answer_extracted"
    elif not cot:
        result["parse_error"] = "no_cot_extracted"
    return result


# ============================================================
# API 调用
# ============================================================

_PROMPT_CORRECT = (
    "你是一位小学数学老师，请一步一步解答以下数学题。\n"
    "【格式要求】\n"
    "- 全程使用纯文本，禁止任何LaTeX命令（如\\frac、\\times、\\sqrt、\\pi等反斜杠开头的符号）。"
    "乘号写×，除号写÷，分数写a/b。\n"
    "【计算规则】\n"
    "- 圆周率取3.14，必须代入算出数值。\n"
    "- 能整除的给整数，不能整除的保留小数或分数（与题目一致）。\n"
    "- 问百分之几或求百分率：答案写成25%的形式，不能写成分数%（如1/4%）。\n"
    "- 问几分之几或求分率：答案写成分数如3/5，不要化成小数。\n"
    "【推理要求】\n"
    "- 每步只做一个运算，写清楚算式和结果。\n"
    "- 不要跳步，不要合并多步计算。\n"
    "请严格按以下格式回答：\n"
    "推理过程：<逐步解题>\n"
    "答案：<纯数字结果，不带单位>"
)

_PROMPT_WRONG = (
    "你是一个数学学生，请尝试解答以下小学数学题，"
    "但请故意在某个步骤中犯一个合理的计算错误，导致最终答案错误。"
    "语气自然，不要用\"故意\"\"但是\"\"错误地\"\"误将\"\"不对\"\"等等\"\"不小心\"等词。"
    "请用以下格式回答：\n"
    "推理过程：<你的错误推理过程>\n"
    "答案：<错误的数字答案>"
)

# 兜底 prompt：当上面的 prompt 3 次都生成了正确答案时使用
# 直接告知正确答案，强制要求不同
_PROMPT_WRONG_HARD = (
    "你是一个数学不好的学生，请解答以下小学数学题，"
    "但你必须在某个计算步骤中犯一个自然的错误，使最终答案与正确答案不同。"
    "正确答案是 {correct_answer}，你的答案必须与之不同。"
    "不要用\"故意\"\"但是\"\"错误地\"\"误将\"\"不对\"\"等等\"之类的词。不准写错误原因。"
    "不要提及以上要求，语气自然。"
    "请用以下格式回答：\n"
    "推理过程：<你的错误解题过程>\n"
    "答案：<错误的数字答案>"
)


def _call_api(
    question: str,
    client,
    model: str,
    generate_wrong: bool = False,
    temperature: float | None = None,
    max_retries: int = 3,
    system_prompt: str | None = None,
) -> Optional[dict]:
    """调用 API 生成推理"""
    if system_prompt is None:
        system_prompt = _PROMPT_WRONG if generate_wrong else _PROMPT_CORRECT
    if temperature is None:
        temperature = 1.3 if generate_wrong else 0.3

    question = _sanitize_question(question)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": question}]},
    ]

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=8192,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            # 记录缓存命中情况
            usage = resp.usage
            cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
            cache_miss = getattr(usage, 'prompt_cache_miss_tokens', 0) or 0
            if cache_hit > 0:
                logger.debug(f"缓存命中: {cache_hit} tokens, 未命中: {cache_miss} tokens")
            return _parse_response(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    return None


# ============================================================
# 主流程
# ============================================================

def build_cot_dataset(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    generate_wrong: bool = False,
    wrong_method: str = "simple",
    resume: bool = True,
    limit: int = 0,
) -> None:
    """
    批量生成 CoT 推理数据

    Args:
        input_path: 原始训练数据路径
        output_path: 输出路径（同时作为断点续传检查点）
        api_key: API 密钥
        model: 模型名称
        max_workers: 并行线程数
        generate_wrong: 是否同时生成错误推理（DPO 用）
        wrong_method: 错误推理生成方式 ("simple" | "scdpo")
        resume: 是否启用断点续传
        limit: 限制条数（0=全部）
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]
        # 小批量测试：输出到独立目录，关闭断点续传
        test_dir = Path(output_path).parent / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(test_dir / f"test_{limit}.json")
        resume = False
        logger.info(f"小批量测试模式: {limit} 条, 输出到 {output_path}")

    # 断点续传
    results = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results = {item["id"]: item for item in existing}
        logger.info(f"断点续传：已有 {len(results)} 条结果")

    todo = [item for item in data if item["id"] not in results]
    logger.info(f"待处理: {len(todo)} 条，已完成: {len(results)} 条")

    client = _get_client(api_key)
    INST = "请一步一步思考，然后给出数字答案。"

    # 按需加载 SCDPO
    scdpo_gen = None
    if generate_wrong and wrong_method == "scdpo":
        from src.data.scdpo import generate_scdpo_wrong
        scdpo_gen = generate_scdpo_wrong
        logger.info("错误推理方式: SCDPO（步骤注入）")
    elif generate_wrong:
        logger.info("错误推理方式: simple（优化 prompt）")

    def process_item(item, is_retry=False):
        temp = 0.1 if is_retry else None
        result = _call_api(item["question"], client, model, temperature=temp)

        # API 失败
        if result is None:
            return {"id": item["id"], "question": item["question"],
                    "instruction": INST, "answer": item["answer"],
                    "cot": "", "api_answer": "", "answer_match": False,
                    "status": "api_failed"}

        # 解析失败
        if result.get("parse_error") == "empty_response" or not result["answer"]:
            return {"id": item["id"], "question": item["question"],
                    "instruction": INST, "answer": item["answer"],
                    "cot": result.get("cot", ""), "api_answer": "",
                    "answer_match": False,
                    "status": "parse_error:" + result.get("parse_error", "empty")}

        out = {
            "id": item["id"], "question": item["question"],
            "instruction": INST, "answer": item["answer"],
            "cot": result["cot"], "api_answer": result["answer"],
            "answer_match": _answers_match(result["answer"], item["answer"]),
            "status": "ok",
        }

        # 生成错误推理（仅对正确匹配的条目）
        if generate_wrong and out["answer_match"]:
            wrong = None
            wrong_status = "none"

            if scdpo_gen is not None:
                wrong = scdpo_gen(
                    question=item["question"],
                    correct_cot=result["cot"],
                    correct_answer=str(item["answer"]),
                    client=client, model=model,
                    parse_fn=_parse_response, match_fn=_answers_match,
                )
                if (wrong and wrong.get("answer")
                        and not _answers_match(wrong["answer"], str(item["answer"]))):
                    wrong_status = "scdpo"

            # SCDPO 失败或使用 simple 方式，最多尝试 3 次
            if wrong_status == "none":
                for _try in range(3):
                    wrong = _call_api(item["question"], client, model,
                                      generate_wrong=True)
                    if (wrong and wrong.get("answer")
                            and not _answers_match(wrong["answer"], str(item["answer"]))):
                        wrong_status = f"simple_try{_try + 1}"
                        break

            # 3 次都算对了 → 兜底：告知正确答案，强制不同（温度 0.7）
            if wrong_status == "none":
                hard_prompt = _PROMPT_WRONG_HARD.format(
                    correct_answer=item["answer"])
                wrong = _call_api(item["question"], client, model,
                                  generate_wrong=True,
                                  system_prompt=hard_prompt,
                                  temperature=0.7)
                if (wrong and wrong.get("answer")
                        and not _answers_match(wrong["answer"], str(item["answer"]))):
                    wrong_status = "hard_fallback"

            if wrong_status != "none":
                out["wrong_cot"] = wrong["cot"]
                out["wrong_answer"] = wrong["answer"]
                out["wrong_status"] = wrong_status
                if "error_step" in wrong:
                    out["wrong_error_step"] = wrong["error_step"]
            else:
                out["wrong_status"] = "failed"

        return out

    # ========== 第一轮 ==========
    completed, failed_items = 0, []
    save_every = min(100, max(10, len(todo) // 5))  # 小批量时更频繁保存
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in todo}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results[result["id"]] = result
                completed += 1
                if not result.get("answer_match") or result.get("status") != "ok":
                    failed_items.append(futures[future])
                if completed % save_every == 0:
                    _save(results, output_path)
                    logger.info(f"[第1轮] 进度: {completed}/{len(todo)}")

    _save(results, output_path)
    mc = sum(1 for r in results.values() if r.get("answer_match"))
    logger.info(f"[第1轮] 完成: {len(results)}, 匹配: {mc}, 需重试: {len(failed_items)}")

    # ========== 第二轮：重试 ==========
    if failed_items:
        logger.info(f"[第2轮] 重试 {len(failed_items)} 条（低温度）...")
        fixed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, it, True): it for it in failed_items}
            for future in as_completed(futures):
                r = future.result()
                if r and r.get("answer_match") and r.get("status") == "ok":
                    results[r["id"]] = r
                    fixed += 1
        _save(results, output_path)
        logger.info(f"[第2轮] 修复: {fixed}/{len(failed_items)}")

    # 统计
    total = len(results)
    mc = sum(1 for r in results.values() if r.get("answer_match"))
    ec = sum(1 for r in results.values() if r.get("status") != "ok")
    logger.info(f"完成: {total} 条, 匹配率: {mc}/{total} ({mc/total*100:.1f}%)")
    if generate_wrong:
        # 按 wrong_status 分类统计
        ws_counts: dict[str, int] = {}
        for r in results.values():
            ws = r.get("wrong_status", "")
            if ws:
                ws_counts[ws] = ws_counts.get(ws, 0) + 1
        logger.info(f"错误推理统计:")
        for ws, cnt in sorted(ws_counts.items()):
            logger.info(f"  {ws}: {cnt} 条")
    if ec:
        logger.info(f"异常: {ec} 条")


def _save(results: dict, path: str) -> None:
    """保存结果（按 id 排序）"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = sorted(results.values(), key=lambda x: int(x["id"]))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build CoT dataset")
    parser.add_argument("--input", default="data/raw/train.json")
    parser.add_argument("--output", default="data/processed/train_cot.json")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generate_wrong", action="store_true",
                        help="同时生成错误推理（DPO 用）")
    parser.add_argument("--wrong_method", default="simple",
                        choices=["simple", "scdpo"],
                        help="错误推理方式: simple(优化prompt) | scdpo(步骤注入)")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_cot_dataset(
        input_path=args.input, output_path=args.output,
        api_key=args.api_key, model=args.model,
        max_workers=args.workers,
        generate_wrong=args.generate_wrong,
        wrong_method=args.wrong_method,
        limit=args.limit,
    )
