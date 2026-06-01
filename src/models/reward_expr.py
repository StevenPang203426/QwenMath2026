"""
表达式方案 GRPO 四维奖励函数

维度说明（4 个独立函数，由 TRL GRPOTrainer 自动求和）：
  1. eval 正确性   (-0.5 ~ +1.0)  safe_eval 结果与 gold answer 匹配
  2. 可解析性       ( 0.0 ~ +0.3)  表达式能被 safe_eval 成功计算
  3. 格式标签       ( 0.0 ~ +0.2)  <expr></expr><answer></answer> 全有
  4. 无非法字符     (-0.3 ~  0.0)  含中文/字母/LaTeX → -0.3

理论总分范围：-1.1 ~ +1.5
"""
import re
from typing import Optional

from src.data.expr_builder import safe_eval, _sanitize_question
from src.utils.metrics import normalize_number

# 非法字符模式：中文、英文字母（运算符 e 除外）、LaTeX 命令
_ILLEGAL_CHARS = re.compile(r'[一-鿿]|\\[a-zA-Z]|[a-df-zA-DF-Z]')


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
    try:
        expr_clean = expr_str.replace('^', '**').replace('×', '*').replace('÷', '/')
        return safe_eval(expr_clean)
    except Exception:
        return None


# ============================================================
# 奖励函数 1: eval 正确性（-0.5 ~ +1.0）
# ============================================================

def expr_correctness_fn(completions, answer=None, **kwargs):
    """
    eval(表达式) 与 gold answer 匹配

    - eval 正确:  +1.0
    - eval 错误:  -0.5
    - 无法解析:  -0.5
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
            rewards.append(-0.5)
            continue

        result_str = str(int(result)) if result == int(result) else str(result)
        gold_norm = normalize_number(str(gold))
        pred_norm = normalize_number(result_str)

        if pred_norm is not None and gold_norm is not None and pred_norm == gold_norm:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)

    return rewards


# ============================================================
# 奖励函数 2: 可解析性（0.0 ~ +0.3）
# ============================================================

def expr_parseable_fn(completions, **kwargs):
    """
    表达式能被 safe_eval 成功计算

    - 可解析:  +0.3
    - 不可解析: 0.0
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        expr = _extract_expr_tag(text)
        result = _try_eval(expr)
        rewards.append(0.3 if result is not None else 0.0)
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
        if expr and _ILLEGAL_CHARS.search(expr):
            rewards.append(-0.3)
        else:
            rewards.append(0.0)
    return rewards


# ============================================================
# 构建奖励函数列表
# ============================================================

def build_expr_reward_funcs():
    """
    返回 4 个独立奖励函数列表

    TRL GRPOTrainer 会自动求和，并在日志中分别记录每个维度的均值。
    """
    return [
        expr_correctness_fn,   # R1: eval 正确性  -0.5 ~ +1.0
        expr_parseable_fn,     # R2: 可解析性      0.0 ~ +0.3
        expr_format_fn,        # R3: 格式标签      0.0 ~ +0.2
        expr_clean_fn,         # R4: 无非法字符   -0.3 ~  0.0
    ]
