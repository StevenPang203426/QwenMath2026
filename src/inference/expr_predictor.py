"""
表达式推理模块

模型输出表达式 → safe_eval 计算 → postprocess 格式化 → fallback 策略
"""
import re
import logging
from typing import Optional

from src.data.expr_builder import safe_eval
from src.inference.answer_postprocessor import postprocess_answer

logger = logging.getLogger("math_solver.expr_predictor")


def extract_expr_from_output(text: str) -> str:
    """从模型输出中提取 <expr>...</expr> 标签内容"""
    if not text:
        return ""
    m = re.search(r'<expr>\s*(.*?)\s*</expr>', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # 兜底：如果没有标签，尝试提取看起来像表达式的行
    for line in reversed(text.strip().split('\n')):
        line = line.strip()
        if line and re.match(r'^[\d\s\+\-\*/\(\)\.\^%]+$', line):
            return line
    return ""


def eval_expression(expr_str: str) -> Optional[float]:
    """
    安全计算表达式

    预处理：替换 ^ → **，× → *，÷ → /

    Returns:
        计算结果，失败返回 None
    """
    if not expr_str:
        return None
    expr_clean = expr_str.replace('^', '**').replace('×', '*').replace('÷', '/')
    try:
        return safe_eval(expr_clean)
    except Exception as e:
        logger.debug(f"eval 失败: {expr_str!r} -> {e}")
        return None


def format_eval_result(result: float) -> str:
    """将 eval 结果转为字符串"""
    if result == int(result):
        return str(int(result))
    return str(result)


def expr_predict_single(
    raw_output: str,
    question: str,
    fallback_answer: Optional[str] = None,
) -> dict:
    """
    表达式推理单条

    流程：
    1. 从模型输出提取 <expr> 标签内的表达式
    2. safe_eval 计算
    3. postprocess_answer 格式化（根据题目上下文）
    4. eval 失败则 fallback 到 CoT 答案

    Args:
        raw_output: 模型原始输出文本
        question: 原始题目文本
        fallback_answer: CoT 模型的答案（eval 失败时使用）

    Returns:
        {"answer": str, "expression": str, "eval_success": bool, "source": str}
    """
    expr = extract_expr_from_output(raw_output)
    result = eval_expression(expr)

    if result is not None:
        # eval 成功
        raw_answer = format_eval_result(result)
        final_answer = postprocess_answer(raw_answer, question)
        return {
            "answer": final_answer,
            "expression": expr,
            "eval_success": True,
            "source": "expr_eval",
        }
    else:
        # eval 失败，fallback
        if fallback_answer:
            final_answer = postprocess_answer(str(fallback_answer), question)
            return {
                "answer": final_answer,
                "expression": expr,
                "eval_success": False,
                "source": "fallback_cot",
            }
        else:
            # 无 fallback，尝试从 <answer> 标签提取
            m = re.search(r'<answer>\s*(.*?)\s*</answer>', raw_output, re.DOTALL)
            ans = m.group(1).strip() if m else "0"
            final_answer = postprocess_answer(ans, question)
            return {
                "answer": final_answer,
                "expression": expr,
                "eval_success": False,
                "source": "answer_tag_fallback",
            }
