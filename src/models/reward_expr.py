"""
表达式方案 GRPO 六维奖励函数

维度说明（6 个独立函数，由 TRL GRPOTrainer 自动求和）：
  1. eval 正确性    (-0.6 ~ +1.0)  safe_eval 结果与 gold answer 匹配
  2. 可解析性        (-0.3 ~ +0.3)  表达式能被 safe_eval 成功计算
  3. 格式标签        ( 0.0 ~ +0.2)  <expr></expr><answer></answer> 全有
  4. 无非法字符      (-0.3 ~  0.0)  含中文/字母/LaTeX → -0.3
  5. answer 一致性  (-0.2 ~ +0.2)  answer 标签与表达式题意归一结果一致
  6. 输出洁净        (-0.2 ~  0.0)  标签外解释或超长输出 → -0.2

理论总分范围：-1.6 ~ +1.7
"""
import re
from typing import Optional

from src.data.expr_builder import safe_eval
from src.utils.answer_normalizer import (
    answers_match_by_question,
    format_auto,
    normalize_answer_for_question,
)
from src.utils.expression_policy import validate_expression

def _extract_text(completion) -> str:
    """从 GRPOTrainer 的 completion 格式中提取文本"""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _extract_expr_tag(text: str) -> str:
    """提取 <expr>...</expr> 标签内容"""
    m = re.search(r'<expr>\s*(.*?)\s*</expr>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _extract_answer_tag(text: str) -> str:
    """提取 <answer>...</answer> 标签内容"""
    m = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _try_eval(expr_str: str) -> Optional[float]:
    """尝试 safe_eval，失败返回 None"""
    if not expr_str:
        return None
    policy = validate_expression(expr_str)
    if not policy.ok:
        return None
    try:
        expr_clean = policy.normalized.replace('^', '**').replace('×', '*').replace('÷', '/')
        return safe_eval(expr_clean)
    except Exception:
        return None


def _normalized_eval_answer(expr_str: str, question: str) -> str | None:
    result = _try_eval(expr_str)
    if result is None:
        return None
    return normalize_answer_for_question(format_auto(result), question)


def _question_for_index(prompt, i: int) -> str:
    if prompt is None:
        return ""
    source = prompt[i] if isinstance(prompt, list) and i < len(prompt) else prompt
    if isinstance(source, list):
        for msg in source:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", ""))
    return str(source)


# ============================================================
# 奖励函数 1: eval 正确性（-0.6 ~ +1.0）
# ============================================================

def expr_correctness_fn(completions, answer=None, prompt=None, **kwargs):
    """
    eval(表达式) 与 gold answer 匹配

    - eval 正确:  +1.0
    - eval 错误:  -0.6
    - 无法解析:  -0.6
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        gold = answer[i] if answer else None

        if gold is None:
            rewards.append(0.0)
            continue

        expr = _extract_expr_tag(text)
        result = _try_eval(expr)

        if result is None:
            rewards.append(-0.6)
            continue

        result_str = format_auto(result)
        question = _question_for_index(prompt, i)

        if answers_match_by_question(result_str, gold, question):
            rewards.append(1.0)
        else:
            rewards.append(-0.6)

    return rewards


# ============================================================
# 奖励函数 2: 可解析性（-0.3 ~ +0.3）
# ============================================================

def expr_parseable_fn(completions, **kwargs):
    """
    表达式能被 safe_eval 成功计算

    - 可解析:  +0.3
    - 不可解析: -0.3
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        expr = _extract_expr_tag(text)
        result = _try_eval(expr)
        rewards.append(0.3 if result is not None else -0.3)
    return rewards


# ============================================================
# 奖励函数 3: 格式标签（0.0 ~ +0.2）
# ============================================================

def expr_format_fn(completions, **kwargs):
    """
    <expr></expr><answer></answer> 四标签全有

    - 全有:  +0.2
    - 缺失:  0.0
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        has_all = (
            "<expr>" in text
            and "</expr>" in text
            and "<answer>" in text
            and "</answer>" in text
        )
        rewards.append(0.2 if has_all else 0.0)
    return rewards


# ============================================================
# 奖励函数 4: 无非法字符（-0.3 ~ 0.0）
# ============================================================

def expr_clean_fn(completions, **kwargs):
    """
    表达式中含中文/字母/LaTeX → 惩罚

    - 无非法字符:  0.0
    - 有非法字符: -0.3
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        expr = _extract_expr_tag(text)
        policy = validate_expression(expr) if expr else None
        if policy is not None and not policy.ok:
            rewards.append(-0.3)
        else:
            rewards.append(0.0)
    return rewards


def expr_answer_consistency_fn(completions, prompt=None, **kwargs):
    """
    <answer> 标签必须与表达式 eval 后题意归一结果一致。

    - 一致:  +0.2
    - 不一致或缺失: -0.2
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        expr = _extract_expr_tag(text)
        answer = _extract_answer_tag(text)
        question = _question_for_index(prompt, i)
        normalized = _normalized_eval_answer(expr, question)
        if normalized is not None and answer and answers_match_by_question(normalized, answer, question):
            rewards.append(0.2)
        else:
            rewards.append(-0.2)
    return rewards


def expr_output_cleanliness_fn(completions, **kwargs):
    """
    防止模型在标签外输出解释或生成过长文本。

    - 干净: 0.0
    - 标签外有正文或超过 128 字符: -0.2
    """
    rewards = []
    pattern = re.compile(r"^\s*<expr>\s*.*?\s*</expr>\s*<answer>\s*.*?\s*</answer>\s*$", re.DOTALL)
    for comp in completions:
        text = _extract_text(comp)
        if len(text) > 128 or not pattern.match(text or ""):
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


# ============================================================
# 构建奖励函数列表
# ============================================================

def build_expr_reward_funcs():
    """
    返回 6 个独立奖励函数列表

    TRL GRPOTrainer 会自动求和，并在日志中分别记录每个维度的均值。
    """
    return [
        expr_correctness_fn,   # R1: eval 正确性  -0.6 ~ +1.0
        expr_parseable_fn,     # R2: 可解析性     -0.3 ~ +0.3
        expr_format_fn,        # R3: 格式标签      0.0 ~ +0.2
        expr_clean_fn,         # R4: 无非法字符   -0.3 ~  0.0
        expr_answer_consistency_fn,  # R5: answer 一致性 -0.2 ~ +0.2
        expr_output_cleanliness_fn,  # R6: 输出洁净      -0.2 ~  0.0
    ]
