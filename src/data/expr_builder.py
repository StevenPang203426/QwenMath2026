"""
表达式数据构建模块
调用 DeepSeek V4 API 为训练数据生成可计算的中缀数学表达式
同时构建 DPO 偏好对（正确表达式 vs 错误表达式）

复用 data_builder.py 的 API 调用模式和缓存策略
"""
import json
import time
import os
import re
import ast
import operator
import logging
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.answer_normalizer import (
    answers_match as _normalized_answers_match,
    answers_match_by_question,
    normalize_answer_for_question,
    safe_eval_expression,
)

logger = logging.getLogger("math_solver.expr_builder")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
MAX_REPAIR_ATTEMPTS = 3


# ============================================================
# 安全 eval（AST 白名单）
# ============================================================

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expr_str: str) -> float:
    """
    安全计算数学表达式

    仅允许数字常量和基本运算符，不允许函数调用或变量访问。
    """
    return safe_eval_expression(expr_str)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Non-numeric constant: {node.value}")
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported binary op: {op_type}")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise ValueError("Division by zero")
        return _SAFE_OPS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand)
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported unary op: {op_type}")
        return _SAFE_OPS[op_type](operand)
    else:
        raise ValueError(f"Unsupported node: {type(node).__name__}")


# ============================================================
# 工具函数
# ============================================================

def _get_client(api_key: str):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _sanitize_question(q) -> str:
    if not q:
        return ""
    if isinstance(q, list):
        q = q[0].get("content", "") if q and isinstance(q[0], dict) else "".join(str(x) for x in q)
    q = str(q).replace('\\"', '"').replace('\\n', '\n').replace('\\t', ' ')
    return q.strip().strip('"')


def _record_sort_key(item: dict) -> tuple[int, object]:
    item_id = str(item.get("id", ""))
    return (0, int(item_id)) if item_id.isdigit() else (1, item_id)


def _load_json_records(path: str) -> list[dict]:
    """Load either pretty JSON arrays or legacy JSONL records."""
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        records = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"跳过无法解析的 JSON 行: {line[:80]}")
        return records


def _save_json_records(path: str, records: list[dict]) -> None:
    """Write records as readable, indented JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    records = sorted(records, key=_record_sort_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _normalize_answer(s: str) -> Optional[float]:
    if not s:
        return None
    s = str(s).strip()
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


def _answers_match(a, b, tol: float = 1e-4) -> bool:
    return _normalized_answers_match(a, b, tol=tol)


def _extract_expr(content: str) -> str:
    """从 API 响应中提取 <expr>...</expr> 标签内的表达式"""
    if not content:
        return ""
    m = re.search(r'<expr>\s*(.*?)\s*</expr>', content, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 兜底：如果没有标签，尝试整行作为表达式
    content = content.strip()
    # 去除可能的解释文字
    lines = content.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and re.match(r'^[\d\s\+\-\*/\(\)\.\^%]+$', line):
            return line
    return content.split('\n')[-1].strip() if lines else ""


# ============================================================
# Prompt 定义
# ============================================================

_PROMPT_EXPR_CORRECT = (
    "你是一位小学数学老师。请为以下数学题写出一个Python可直接计算的中缀数学表达式。\n"
    "【格式要求】\n"
    "- 只输出一个数学表达式，不要输出推理过程\n"
    "- 使用纯数字和运算符：+ - * / ** ( )\n"
    "- 圆周率用3.14代替\n"
    "- 不要使用任何变量名、中文、单位、LaTeX\n"
    "- 分数直接写除法，如三分之二写成 2/3\n"
    "- 百分数写成小数运算，如25%写成 25/100 或 0.25\n"
    "- 表达式必须能被Python的eval()直接计算\n"
    "- 表达式中不包含空格\n"
    "请严格按以下格式回答：\n"
    "<expr>你的表达式</expr>"
)

_PROMPT_EXPR_WRONG = (
    "你是一个粗心的数学学生。请为以下数学题写出一个Python可计算的数学表达式，"
    "但请故意在某个数字或运算符上犯一个错误。\n"
    "语气自然，不要用\"故意\"\"错误地\"等词。不要提及错误原因。\n"
    "表达式中不包含空格。你的答案表达式一定是错的，\n"
    "请严格按以下格式回答：\n"
    "<expr>你的表达式</expr>"
)


# ============================================================
# API 调用
# ============================================================

def _call_expr_api(
    question: str,
    client,
    model: str,
    generate_wrong: bool = False,
    temperature: float = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """调用 API 生成表达式"""
    system_prompt = _PROMPT_EXPR_WRONG if generate_wrong else _PROMPT_EXPR_CORRECT
    if temperature is None:
        temperature = 1.0 if generate_wrong else 0.3

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
                max_tokens=512,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            content = resp.choices[0].message.content
            expr = _extract_expr(content)

            # 记录缓存命中
            usage = resp.usage
            cache_hit = getattr(usage, 'prompt_cache_hit_tokens', 0) or 0
            if cache_hit > 0:
                logger.debug(f"缓存命中: {cache_hit} tokens")

            return {"expression": expr, "raw_response": content}
        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)
    return None


# ============================================================
# 验证函数
# ============================================================

def verify_expression(expr_str: str, gold_answer: str, question: str = "") -> dict:
    """
    验证表达式是否正确

    Returns:
        {"valid": bool, "eval_result": str|None, "error": str|None}
    """
    if not expr_str:
        return {"valid": False, "eval_result": None, "error": "empty_expression"}

    # 预处理：替换 ^ 为 **，× 为 *，÷ 为 /
    expr_clean = expr_str.replace('^', '**').replace('×', '*').replace('÷', '/')

    try:
        result = safe_eval(expr_clean)
    except Exception as e:
        return {"valid": False, "eval_result": None, "error": f"eval_failed: {e}"}

    result_str = str(int(result)) if result == int(result) else str(result)
    formatted_result = normalize_answer_for_question(result_str, question) if question else result_str

    if answers_match_by_question(result_str, gold_answer, question) if question else _answers_match(result_str, gold_answer):
        return {"valid": True, "eval_result": formatted_result, "error": None}
    else:
        return {
            "valid": False,
            "eval_result": formatted_result,
            "error": f"mismatch: eval={formatted_result}, gold={gold_answer}",
        }


def _is_compliant_expr_record(item: dict, generate_wrong: bool) -> bool:
    """Return whether an existing expression record is safe to reuse."""
    if item.get("status") != "ok":
        return False
    if not item.get("expression"):
        return False
    if generate_wrong:
        return item.get("valid") is False and item.get("eval_result") is not None
    return item.get("valid") is True


def _repair_reason(item: dict | None, generate_wrong: bool) -> str:
    if item is None:
        return "missing"
    if not item.get("expression"):
        return "empty_expression"
    if item.get("status") == "api_failed":
        return "api_failed"
    if generate_wrong:
        if item.get("status") == "accidentally_correct" or item.get("valid") is True:
            return "wrong_accidentally_correct"
        if item.get("status") == "unparseable" or item.get("eval_result") is None:
            return "wrong_unparseable"
        return "wrong_invalid"
    if item.get("valid") is not True:
        return "correct_invalid"
    return "invalid_status"


# ============================================================
# 主流程
# ============================================================

def build_expr_dataset(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    generate_wrong: bool = False,
    resume: bool = True,
    limit: int = 0,
) -> None:
    """
    批量生成表达式数据

    Args:
        input_path: 原始训练数据 JSON 路径
        output_path: 输出路径
        api_key: DeepSeek API key
        model: 模型名
        max_workers: 并发数
        generate_wrong: 是否生成错误表达式（DPO rejected）
        resume: 断点续传
        limit: 限制条数（0=不限制）
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit > 0:
        data = data[:limit]

    logger.info(f"数据总量: {len(data)}, 模式: {'错误表达式' if generate_wrong else '正确表达式'}")

    # 断点续传：只跳过合规数据，不合规数据默认进入修复队列
    existing = {}
    reusable = {}
    repair_reasons = {}
    if resume:
        existing = {item["id"]: item for item in _load_json_records(output_path) if "id" in item}
        if existing:
            reusable = {
                item_id: item
                for item_id, item in existing.items()
                if _is_compliant_expr_record(item, generate_wrong)
            }
            repair_reasons = {
                item_id: _repair_reason(item, generate_wrong)
                for item_id, item in existing.items()
                if item_id not in reusable
            }
            logger.info(
                f"已有结果: {len(existing)} 条，合规跳过: {len(reusable)} 条，"
                f"需重算: {len(repair_reasons)} 条"
            )

    # 按 prompt 类型排序（提高缓存命中率）
    # 正样本和负样本分开调用，prompt 前缀固定 → 缓存命中率高
    client = _get_client(api_key)
    results = dict(existing)

    processed = 0
    success = 0
    skipped = len(reusable)

    def process_item(item):
        item_id = item["id"]
        if item_id in reusable:
            return None

        question = item["question"]
        gold_answer = str(item["answer"])
        initial_reason = repair_reasons.get(item_id, "missing")
        last_output = None

        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            result = _call_expr_api(
                question=question,
                client=client,
                model=model,
                generate_wrong=generate_wrong,
            )

            if result is None:
                last_output = {
                    "id": item_id,
                    "question": _sanitize_question(question),
                    "answer": gold_answer,
                    "expression": "",
                    "eval_result": None,
                    "valid": False,
                    "error": "api_failed",
                    "status": "api_failed",
                }
            else:
                expr = result["expression"]
                verification = verify_expression(expr, gold_answer, question)

                last_output = {
                    "id": item_id,
                    "question": _sanitize_question(question),
                    "answer": gold_answer,
                    "expression": expr,
                    "eval_result": verification["eval_result"],
                    "valid": verification["valid"],
                    "error": verification["error"],
                    "status": "ok" if verification["valid"] else "eval_mismatch",
                }

                # 对于错误表达式模式：valid=False 且可解析才是合规负样本
                if generate_wrong:
                    if verification["error"] and "eval_failed" in verification["error"]:
                        last_output["status"] = "unparseable"
                    elif verification["valid"]:
                        last_output["status"] = "accidentally_correct"
                    else:
                        last_output["status"] = "ok"

            last_output["attempts"] = attempt
            last_output["repair_reason"] = initial_reason
            if _is_compliant_expr_record(last_output, generate_wrong):
                return last_output
            last_output["repair_reason"] = _repair_reason(last_output, generate_wrong)

        return last_output

    # 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in data}

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is None:
                    continue
                processed += 1
                if result.get("status") == "ok":
                    success += 1
                results[result["id"]] = result
                _save_json_records(output_path, list(results.values()))

                if processed % 100 == 0:
                    total = processed + skipped
                    logger.info(
                        f"进度: {total}/{len(data)}, "
                        f"成功: {success}/{processed} ({success/processed*100:.1f}%)"
                    )
            except Exception as e:
                logger.error(f"处理失败: {e}")

    _save_json_records(output_path, list(results.values()))
    logger.info(
        f"完成! 新处理: {processed}, 成功: {success}, "
        f"成功率: {success/processed*100:.1f}%" if processed > 0 else "无新数据"
    )


def convert_to_sft_format(
    expr_jsonl_path: str,
    output_path: str,
) -> None:
    """
    将表达式 JSONL 转换为 SFT 训练格式

    只保留 valid=True 的正确表达式
    """
    items = []
    for item in _load_json_records(expr_jsonl_path):
        if item.get("valid") and item.get("status") == "ok":
            items.append({
                "id": item["id"],
                "question": item["question"],
                "expression": item["expression"],
                "answer": item["answer"],
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    logger.info(f"SFT 数据: {len(items)} 条 → {output_path}")


def convert_to_dpo_format(
    correct_jsonl_path: str,
    wrong_jsonl_path: str,
    output_path: str,
) -> None:
    """
    将正确和错误表达式合并为 DPO 偏好对

    配对规则：同 id 的正确表达式做 chosen，错误表达式做 rejected
    """
    correct = {}
    for item in _load_json_records(correct_jsonl_path):
        if item.get("valid") and item.get("status") == "ok":
            correct[item["id"]] = item

    pairs = []
    for item in _load_json_records(wrong_jsonl_path):
        item_id = item["id"]
        if item_id in correct and item.get("status") == "ok":
            pairs.append({
                "id": item_id,
                "question": correct[item_id]["question"],
                "chosen": f"<expr>{correct[item_id]['expression']}</expr><answer>{correct[item_id]['answer']}</answer>",
                "rejected": f"<expr>{item['expression']}</expr><answer>{item.get('eval_result', '')}</answer>",
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    logger.info(f"DPO 偏好对: {len(pairs)} 条 → {output_path}")


def export_failed_to_candidates(
    correct_path: str,
    wrong_path: str,
    output_path: str,
    raw_data_path: str = "data/raw/train.json",
) -> None:
    """
    导出 3 次重算仍失败的记录到 quality_candidates.json

    供 quality_auditor 统一审计。收集以下信号：
    - correct 模式下 valid != True 的记录
    - wrong 模式下多次生成仍 accidentally_correct 或 unparseable 的记录
    - 两个模式的交集（同一题在两个模式都失败 → 高概率是题目本身有问题）
    """
    with open(raw_data_path, "r", encoding="utf-8") as f:
        raw_data = {item["id"]: item for item in json.load(f)}

    correct_failed = {}
    for item in _load_json_records(correct_path):
        if not _is_compliant_expr_record(item, generate_wrong=False):
            correct_failed[item["id"]] = item

    wrong_failed = {}
    for item in _load_json_records(wrong_path):
        if not _is_compliant_expr_record(item, generate_wrong=True):
            wrong_failed[item["id"]] = item

    # 合并：取并集
    all_failed_ids = set(correct_failed.keys()) | set(wrong_failed.keys())
    both_failed_ids = set(correct_failed.keys()) & set(wrong_failed.keys())

    candidates = []
    for item_id in sorted(all_failed_ids, key=lambda x: int(x) if str(x).isdigit() else 0):
        raw = raw_data.get(item_id, {})
        signals = []
        if item_id in correct_failed:
            reason = correct_failed[item_id].get("repair_reason", "unknown")
            signals.append(f"expr_correct_failed:{reason}")
        if item_id in wrong_failed:
            reason = wrong_failed[item_id].get("repair_reason", "unknown")
            signals.append(f"expr_wrong_failed:{reason}")
        if item_id in both_failed_ids:
            signals.append("both_modes_failed")

        candidates.append({
            "id": item_id,
            "question": raw.get("question", ""),
            "answer": str(raw.get("answer", "")),
            "signals": signals,
            "risk_score": 0.9 if item_id in both_failed_ids else 0.7,
        })

    _save_json_records(output_path, candidates)
    logger.info(
        f"导出审计候选: {len(candidates)} 条 → {output_path} "
        f"(correct 失败: {len(correct_failed)}, wrong 失败: {len(wrong_failed)}, "
        f"两模式都失败: {len(both_failed_ids)})"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if len(sys.argv) < 2:
        print("用法:")
        print("  生成正确表达式:  python -m src.data.expr_builder correct <api_key> [--limit N]")
        print("  生成错误表达式:  python -m src.data.expr_builder wrong <api_key> [--limit N]")
        print("  转换 SFT 数据:   python -m src.data.expr_builder convert_sft")
        print("  转换 DPO 数据:   python -m src.data.expr_builder convert_dpo")
        print("  导出审计候选:    python -m src.data.expr_builder export_failed")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "correct":
        api_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DEEPSEEK_API_KEY", "")
        limit_val = 0
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            limit_val = int(sys.argv[idx + 1])
        build_expr_dataset(
            input_path="data/raw/train.json",
            output_path="data/processed/expr_correct.jsonl",
            api_key=api_key,
            generate_wrong=False,
            limit=limit_val,
        )
    elif mode == "wrong":
        api_key = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DEEPSEEK_API_KEY", "")
        limit_val = 0
        if "--limit" in sys.argv:
            idx = sys.argv.index("--limit")
            limit_val = int(sys.argv[idx + 1])
        build_expr_dataset(
            input_path="data/raw/train.json",
            output_path="data/processed/expr_wrong.jsonl",
            api_key=api_key,
            generate_wrong=True,
            limit=limit_val,
        )
    elif mode == "convert_sft":
        convert_to_sft_format(
            "data/processed/expr_correct.jsonl",
            "data/processed/train_expr.json",
        )
    elif mode == "convert_dpo":
        convert_to_dpo_format(
            "data/processed/expr_correct.jsonl",
            "data/processed/expr_wrong.jsonl",
            "data/processed/train_expr_dpo.json",
        )
    elif mode == "export_failed":
        export_failed_to_candidates(
            correct_path="data/processed/expr_correct.jsonl",
            wrong_path="data/processed/expr_wrong.jsonl",
            output_path="data/processed/intermediate/quality/quality_candidates.json",
        )
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)
