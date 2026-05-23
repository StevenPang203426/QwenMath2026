"""
方案 1：CoT 提示工程模块
不微调模型，通过优化 prompt 提升推理能力
"""
import logging
from typing import Optional

logger = logging.getLogger("math_solver.cot_prompting")


# ============================================================
# Zero-shot CoT 提示模板
# ============================================================

ZERO_SHOT_TEMPLATES = {
    "cn_step_by_step": (
        "这是一道小学数学应用题。请你一步一步仔细思考，"
        "列出详细的解题过程，最后给出一个纯数字答案。\n"
        "请用以下格式回答：\n"
        "<think>你的推理过程</think>\n"
        "<answer>纯数字答案</answer>"
    ),
    "cn_simple": (
        "请认真分析这道数学题，一步一步思考，最后给出数字答案。\n"
        "答案格式：<answer>数字</answer>"
    ),
    "en_lets_think": (
        "Let's think step by step to solve this math problem. "
        "Show your reasoning, then give the final numerical answer.\n"
        "Format: <think>reasoning</think><answer>number</answer>"
    ),
}

# ============================================================
# Few-shot CoT 示例
# ============================================================

FEW_SHOT_EXAMPLES = [
    {
        "question": "商店有4框苹果，每框55千克，已经卖出135千克，还剩多少千克苹果？",
        "reasoning": (
            "首先计算总共有多少千克苹果：4框 × 55千克/框 = 220千克。"
            "然后减去卖出的：220千克 - 135千克 = 85千克。"
        ),
        "answer": "85",
    },
    {
        "question": "玩具厂生产了960个电子玩具，每3个装一盒，每5盒装一箱，一共装了多少箱？",
        "reasoning": (
            "先算一共装了多少盒：960 ÷ 3 = 320盒。"
            "再算装了多少箱：320 ÷ 5 = 64箱。"
        ),
        "answer": "64",
    },
    {
        "question": "小明有36块糖，小红的糖是小明的2倍，小红有多少块糖？",
        "reasoning": "小红的糖 = 小明的糖 × 2 = 36 × 2 = 72块。",
        "answer": "72",
    },
]


def build_zero_shot_prompt(
    question: str,
    template_key: str = "cn_step_by_step",
) -> list[dict]:
    """
    构建 Zero-shot CoT 提示

    Args:
        question: 数学题目
        template_key: 模板名称

    Returns:
        messages 列表（OpenAI 格式）
    """
    template = ZERO_SHOT_TEMPLATES.get(template_key, ZERO_SHOT_TEMPLATES["cn_step_by_step"])

    return [
        {"role": "system", "content": template},
        {"role": "user", "content": question},
    ]


def build_few_shot_prompt(
    question: str,
    examples: list[dict] | None = None,
    num_examples: int = 3,
) -> list[dict]:
    """
    构建 Few-shot CoT 提示

    Args:
        question: 数学题目
        examples: 示例列表（可选，默认使用内置示例）
        num_examples: 使用的示例数量

    Returns:
        messages 列表
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES

    examples = examples[:num_examples]

    system_msg = (
        "你是一个数学老师，请一步一步解答数学题。"
        "请用以下格式回答：\n"
        "<think>推理过程</think>\n"
        "<answer>纯数字答案</answer>"
    )

    messages = [{"role": "system", "content": system_msg}]

    # 添加 few-shot 示例
    for ex in examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({
            "role": "assistant",
            "content": f"<think>{ex['reasoning']}</think><answer>{ex['answer']}</answer>",
        })

    # 添加实际问题
    messages.append({"role": "user", "content": question})

    return messages


def get_cot_strategy(strategy: str):
    """
    获取 CoT 策略对应的 prompt 构建函数

    Args:
        strategy: "zero_shot" 或 "few_shot"

    Returns:
        prompt 构建函数
    """
    strategies = {
        "zero_shot": build_zero_shot_prompt,
        "few_shot": build_few_shot_prompt,
    }
    if strategy not in strategies:
        raise ValueError(f"未知 CoT 策略: {strategy}，可选: {list(strategies.keys())}")
    return strategies[strategy]
