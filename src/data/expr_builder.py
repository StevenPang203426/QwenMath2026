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

logger = logging.getLogger("math_solver.expr_builder")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


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
    if not expr_str or not expr_str.strip():
        raise ValueError("Empty expression")
    expr_str = expr_str.strip()
    tree = ast.parse(expr_str, mode='eval')
    return _eval_node(tree.body)


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
    if str(a).strip() == str(b).strip():
        return True
    va, vb = _normalize_answer(str(a)), _normalize_answer(str(b))
    if va is not None and vb is not None:
        return abs(va - vb) < tol
    return False


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
    "请严格按以下格式回答：\n"
    "<expr>你的表达式</expr>"
)

_PROMPT_EXPR_WRONG = (
    "你是一个数学学生。请为以下数学题写出一个Python可计算的数学表达式，"
    "但请故意在某个数字或运算符上犯一个合理的错误。\n"
    "语气自然，不要用\"故意\"\"错误地\"等词。\n"
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

def verify_expression(expr_str: str, gold_answer: str) -> dict:
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

    if _answers_match(result_str, gold_answer):
        return {"valid": True, "eval_result": result_str, "error": None}
    else:
        return {"valid": False, "eval_result": result_str, "error": f"mismatch: eval={result_str}, gold={gold_answer}"}


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

    # 断点续传：加载已有结果
    existing = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    existing[item["id"]] = item
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"已有结果: {len(existing)} 条，将跳过")

    # 按 prompt 类型排序（提高缓存命中率）
    # 正样本和负样本分开调用，prompt 前缀固定 → 缓存命中率高
    client = _get_client(api_key)
    out_f = open(output_path, "a", encoding="utf-8")

    processed = 0
    success = 0
    skipped = len(existing)

    def process_item(item):
        item_id = item["id"]
        if item_id in existing:
            return None

        question = item["question"]
        gold_answer = str(item["answer"])

        result = _call_expr_api(
            question=question,
            client=client,
            model=model,
            generate_wrong=generate_wrong,
        )

        if result is None:
            return {"id": item_id, "status": "api_failed"}

        expr = result["expression"]
        verification = verify_expression(expr, gold_answer)

        output = {
            "id": item_id,
            "question": _sanitize_question(question),
            "answer": gold_answer,
            "expression": expr,
            "eval_result": verification["eval_result"],
            "valid": verification["valid"],
            "error": verification["error"],
            "status": "ok" if verification["valid"] else "eval_mismatch",
        }

        # 对于错误表达式模式：valid=False 反而是我们想要的
        if generate_wrong:
            # 错误表达式需要：可解析但结果不对
            if verification["error"] and "eval_failed" in verification["error"]:
                output["status"] = "unparseable"
            elif verification["valid"]:
                output["status"] = "accidentally_correct"
            else:
                output["status"] = "ok"

        return output

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
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

                if processed % 100 == 0:
                    total = processed + skipped
                    logger.info(
                        f"进度: {total}/{len(data)}, "
                        f"成功: {success}/{processed} ({success/processed*100:.1f}%)"
                    )
            except Exception as e:
                logger.error(f"处理失败: {e}")

    out_f.close()
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
    with open(expr_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
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
    with open(correct_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("valid") and item.get("status") == "ok":
                correct[item["id"]] = item

    pairs = []
    with open(wrong_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
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


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    if len(sys.argv) < 2:
        print("用法:")
        print("  生成正确表达式: python -m src.data.expr_builder correct <api_key> [--limit N]")
        print("  生成错误表达式: python -m src.data.expr_builder wrong <api_key> [--limit N]")
        print("  转换 SFT 数据:  python -m src.data.expr_builder convert_sft")
        print("  转换 DPO 数据:  python -m src.data.expr_builder convert_dpo")
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
    else:
        print(f"未知模式: {mode}")
        sys.exit(1)
