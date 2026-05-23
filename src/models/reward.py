"""
GRPO 奖励函数模块
定义格式奖励和正确性奖励
"""
import re
from typing import Optional


def format_reward(
    response: str,
    pattern: str = r'<think>.*?</think>\s*<answer>.*?</answer>',
) -> float:
    """
    格式奖励：检查输出是否符合指定格式

    Args:
        response: 模型输出
        pattern: 期望的格式正则

    Returns:
        奖励值 (0.0 或 1.0)
    """
    if re.search(pattern, response, re.DOTALL):
        return 1.0
    return 0.0


def correctness_reward(
    response: str,
    gold_answer: str,
) -> float:
    """
    正确性奖励：检查提取的答案是否与标注一致

    Args:
        response: 模型输出
        gold_answer: 标注答案

    Returns:
        奖励值 (0.0 或 1.0)
    """
    from src.data.answer_extractor import extract_answer
    from src.utils.metrics import normalize_number

    predicted = extract_answer(response)
    pred_norm = normalize_number(predicted)
    gold_norm = normalize_number(str(gold_answer))

    if pred_norm is not None and gold_norm is not None and pred_norm == gold_norm:
        return 1.0
    return 0.0


def combined_reward(
    response: str,
    gold_answer: str,
    format_weight: float = 0.2,
    correctness_weight: float = 1.0,
    format_pattern: str = r'<think>.*?</think>\s*<answer>.*?</answer>',
) -> float:
    """
    组合奖励函数

    Args:
        response: 模型输出
        gold_answer: 标注答案
        format_weight: 格式奖励权重
        correctness_weight: 正确性奖励权重
        format_pattern: 格式正则

    Returns:
        加权总奖励
    """
    fmt = format_reward(response, format_pattern)
    cor = correctness_reward(response, gold_answer)

    return format_weight * fmt + correctness_weight * cor


def build_reward_fn(config):
    """
    根据配置构建奖励函数（供 GRPOTrainer 使用）

    Args:
        config: 包含 reward 配置的 Config 对象

    Returns:
        奖励函数
    """
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
