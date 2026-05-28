"""
GRPO 五维奖励函数模块

维度说明（5 个独立函数，由 TRL GRPOTrainer 自动求和）：
  1. 正确性奖励  (-0.5 ~ +1.0)  精确匹配/精度偏差/错误/无法解析
  2. 格式标签奖励 ( 0.0 ~ +0.2)  <think></think><answer></answer> 全有 +0.2
  3. 逻辑词奖励  (-0.3 ~ +0.95) 基础组 ×0.05 + 结构组 ×0.1，0 命中 -0.3
  4. 长度正则奖励 (-0.3 ~  0.0)  过短 -0.3，过长 -0.2，合理 0.0
  5. 无 LaTeX 奖励(-0.3 ~  0.0)  含 LaTeX -0.3，否则 0.0

理论总分范围：-1.4 ~ +2.15
"""
import math
import re
from typing import Optional

from src.data.answer_extractor import extract_answer
from src.utils.metrics import normalize_number

# ============================================================
# 工具函数
# ============================================================

_LATEX_PATTERN = re.compile(r'\\[a-zA-Z]')

# 精度要求关键词
_PRECISION_PATTERN = re.compile(r'保留|精确到')

# 逻辑词分组
_LOGIC_BASIC = {"即", "所以", "因此", "得到", "那么"}          # 高频推导词
_LOGIC_STRUCT = {"首先", "已知", "题意", "设", "因为", "由于", "根据"}  # 低频结构词

# 长度阈值（基于 train_cot_raw 统计：正样本 p10=48, median=79, p90=145）
_LEN_TOO_SHORT = 30    # 低于正样本 min(21) 附近
_LEN_TOO_LONG = 300    # 高于正样本 p90(145) + 充足余量


def _extract_text(completion) -> str:
    """从 GRPOTrainer 的 completion 格式中提取文本"""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _round_match(pred_str: str, gold_str: str) -> bool:
    """
    四舍五入等价判断：
    将 pred 和 gold 分别 round 到较少的小数位数，
    如果相等则视为精度偏差（非计算错误）。
    例: pred='3.333', gold='3.3' → round(3.333, 1) == 3.3 → True
    例: pred='72', gold='73' → 都是整数，72 != 73 → False
    """
    try:
        pred_val = float(pred_str)
        gold_val = float(gold_str)
    except (ValueError, TypeError):
        return False

    # 计算各自小数位数
    def _decimal_places(s: str) -> int:
        s = s.strip()
        if '.' not in s:
            return 0
        return len(s.split('.')[1])

    pred_dp = _decimal_places(pred_str)
    gold_dp = _decimal_places(gold_str)
    min_dp = min(pred_dp, gold_dp)

    return round(pred_val, min_dp) == round(gold_val, min_dp)


# ============================================================
# 奖励函数 1: 正确性（-0.5 ~ +1.0）
# ============================================================

def correctness_reward_fn(completions, answer=None, prompt=None, **kwargs):
    """
    分级正确性奖励 + 负惩罚

    - 精确匹配:       +1.0
    - 精度偏差（无精度要求词 + 四舍五入等价）: +0.3
    - 错误:           -0.5
    - 无法解析:       -0.5
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        gold = answer[i] if answer else None

        if gold is None:
            rewards.append(0.0)
            continue

        predicted = extract_answer(text)
        pred_norm = normalize_number(predicted)
        gold_norm = normalize_number(str(gold))

        # 无法解析
        if pred_norm is None or gold_norm is None:
            rewards.append(-0.5)
            continue

        # 精确匹配
        if pred_norm == gold_norm:
            rewards.append(1.0)
            continue

        # 精度偏差判断（仅在题目无精度要求时）
        question = ""
        if prompt is not None:
            if isinstance(prompt, list):
                # chat format: 取 user message
                for msg in prompt:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        question = msg.get("content", "")
                        break
            elif isinstance(prompt, str):
                question = prompt

        has_precision_req = bool(_PRECISION_PATTERN.search(question))

        if not has_precision_req and _round_match(pred_norm, gold_norm):
            rewards.append(0.3)
        else:
            rewards.append(-0.5)

    return rewards


# ============================================================
# 奖励函数 2: 格式标签（0.0 ~ +0.2）
# ============================================================

def format_reward_fn(completions, **kwargs):
    """
    二值格式奖励：四标签全有 +0.2，否则 0.0
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        has_all = (
            "<think>" in text
            and "</think>" in text
            and "<answer>" in text
            and "</answer>" in text
        )
        rewards.append(0.2 if has_all else 0.0)
    return rewards


# ============================================================
# 奖励函数 3: 逻辑词覆盖（-0.3 ~ +0.95）
# ============================================================

def logic_word_reward_fn(completions, **kwargs):
    """
    按逻辑词种类数计分，防止刷词

    基础组（即/所以/因此/得到/那么）:   每命中一种 +0.05，共 5 种
    结构组（首先/已知/题意/设/因为/由于/根据）: 每命中一种 +0.10，共 7 种
    0 个命中: -0.3
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)

        basic_hits = sum(1 for w in _LOGIC_BASIC if w in text)
        struct_hits = sum(1 for w in _LOGIC_STRUCT if w in text)
        total_hits = basic_hits + struct_hits

        if total_hits == 0:
            rewards.append(-0.3)
        else:
            score = basic_hits * 0.05 + struct_hits * 0.1
            rewards.append(score)
    return rewards


# ============================================================
# 奖励函数 4: 长度正则（-0.3 ~ 0.0）
# ============================================================

def length_reward_fn(completions, **kwargs):
    """
    长度正则化奖励，纯防守型（只惩罚极端，不奖励合理）

    阈值基于 train_cot_raw 统计：
      正样本: min=21, p10=48, median=79, p90=145, max=582
      负样本: min=34, p10=78, median=108, p90=169

    < 30 字符:   -0.3（几乎无推理）
    30~300 字符:   0.0（合理范围）
    > 300 字符:  -0.2（接近截断风险）
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        char_len = len(text)

        if char_len < _LEN_TOO_SHORT:
            rewards.append(-0.3)
        elif char_len > _LEN_TOO_LONG:
            rewards.append(-0.2)
        else:
            rewards.append(0.0)
    return rewards


# ============================================================
# 奖励函数 5: 无 LaTeX（-0.3 ~ 0.0）
# ============================================================

def no_latex_reward_fn(completions, **kwargs):
    """
    LaTeX 惩罚，复用 data_repair.py 的检测标准

    含 LaTeX 命令（\\frac, \\times 等）: -0.3
    无 LaTeX: 0.0
    """
    rewards = []
    for comp in completions:
        text = _extract_text(comp)
        has_latex = bool(_LATEX_PATTERN.search(text)) if text else False
        rewards.append(-0.3 if has_latex else 0.0)
    return rewards


# ============================================================
# 构建奖励函数列表（供 GRPOTrainer 使用）
# ============================================================

def build_reward_funcs():
    """
    返回 5 个独立奖励函数列表

    TRL GRPOTrainer 会自动求和，并在日志中分别记录每个维度的均值。
    """
    return [
        correctness_reward_fn,   # R1: 正确性   -0.5 ~ +1.0
        format_reward_fn,        # R2: 格式标签  0.0 ~ +0.2
        logic_word_reward_fn,    # R3: 逻辑词   -0.3 ~ +0.95
        length_reward_fn,        # R4: 长度正则  -0.3 ~  0.0
        no_latex_reward_fn,      # R5: 无LaTeX  -0.3 ~  0.0
    ]


# ============================================================
# 向后兼容：保留旧接口
# ============================================================

def combined_reward(
    response: str,
    gold_answer: str,
    format_weight: float = 0.2,
    correctness_weight: float = 1.0,
    format_pattern: str = r'<think>.*?</think>\s*<answer>.*?</answer>',
) -> float:
    """旧版组合奖励（向后兼容，不推荐新代码使用）"""
    fmt = 1.0 if re.search(format_pattern, response, re.DOTALL) else 0.0
    predicted = extract_answer(response)
    pred_norm = normalize_number(predicted)
    gold_norm = normalize_number(str(gold_answer))
    cor = 1.0 if (pred_norm is not None and gold_norm is not None
                   and pred_norm == gold_norm) else 0.0
    return format_weight * fmt + correctness_weight * cor


def build_reward_fn(config):
    """旧版单一奖励函数构建（向后兼容）"""
    reward_cfg = config.reward if hasattr(config, 'reward') else None
    fmt_weight = getattr(reward_cfg, 'format_weight', 0.2) if reward_cfg else 0.2
    cor_weight = getattr(reward_cfg, 'correctness_weight', 1.0) if reward_cfg else 1.0
    fmt_pattern = (
        getattr(reward_cfg, 'format_pattern', r'<think>.*?</think>\s*<answer>.*?</answer>')
        if reward_cfg else r'<think>.*?</think>\s*<answer>.*?</answer>'
    )

    def reward_fn(response: str, gold_answer: str) -> float:
        return combined_reward(response, gold_answer, fmt_weight, cor_weight, fmt_pattern)

    return reward_fn
