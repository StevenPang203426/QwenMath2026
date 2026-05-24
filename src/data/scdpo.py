"""
SCDPO (Step-Controlled DPO) 错误推理生成模块

将正确推理拆分为步骤，随机选择注入点，保留正确前半段+错误后半段
适用于需要更精细控制错误位置的场景

注意：相比简单 prompt 方式，SCDPO 会多消耗 input token（需发送正确前缀）
大多数场景下简单 prompt 已足够自然，建议仅在需要时启用
"""
import re
import random
import time
import logging
from typing import Optional

logger = logging.getLogger("math_solver.scdpo")


def split_cot_steps(cot: str) -> list[str]:
    """
    将 CoT 推理过程拆分为多个步骤

    拆分策略（按优先级）：
    1. 按显式标记拆分
    2. 按换行符拆分
    3. 按句号拆分
    """
    if not cot or not cot.strip():
        return []

    step_pattern = r'(?:第[一二三四五六七八九十\d]+步|步骤\s*\d+|\d+[.、)）])'
    parts = re.split(step_pattern, cot)
    steps = [p.strip() for p in parts if p.strip()]
    if len(steps) >= 3:
        return steps

    lines = [line.strip() for line in cot.split('\n') if line.strip()]
    if len(lines) >= 3:
        return lines

    sentences = [s.strip() for s in re.split(r'[。.；;]', cot) if s.strip()]
    if len(sentences) >= 2:
        return sentences

    return [cot.strip()]


def generate_scdpo_wrong(
    question: str,
    correct_cot: str,
    correct_answer: str,
    client,
    model: str,
    parse_fn,
    match_fn,
) -> Optional[dict]:
    """
    SCDPO 风格错误推理生成

    Args:
        question: 题目
        correct_cot: 正确推理过程
        correct_answer: 正确答案
        client: OpenAI 兼容客户端
        model: 模型名称
        parse_fn: 响应解析函数
        match_fn: 答案匹配函数

    Returns:
        {"cot": str, "answer": str, "error_step": int} 或 None
    """
    steps = split_cot_steps(correct_cot)

    if len(steps) < 2:
        return None

    error_idx = random.randint(1, len(steps) - 1)
    correct_prefix = "\n".join(steps[:error_idx])

    system_prompt = (
        "你是一个数学学生。下面给出了一道小学数学题和前几步正确的推理过程。"
        "请从标注位置继续解题，但在后续某个计算中犯一个合理的错误"
        "（如计算失误、单位混淆、条件遗漏等），导致最终答案与正确答案不同。\n"
        f"正确答案是 {correct_answer}，你的最终答案必须与之不同。\n"
        "错误要自然，不要太明显。请用以下格式回答：\n"
        "继续推理：<含错误的后续推理>\n"
        "答案：<错误的数字答案>"
    )

    user_content = (
        f"题目：{question}\n\n"
        f"已有的正确推理：\n{correct_prefix}\n\n"
        "请从这里继续，并在后续步骤中引入一个计算错误。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=768,
                stream=False,
            )
            resp_content = response.choices[0].message.content
            if not resp_content:
                continue

            parsed = parse_fn(resp_content)
            if not parsed.get("answer"):
                continue

            if match_fn(parsed["answer"], correct_answer):
                continue

            wrong_suffix = parsed.get("cot", "")
            cont_match = re.search(r'继续推理[：:]\s*(.*)', resp_content, re.DOTALL)
            if cont_match:
                wrong_suffix = cont_match.group(1).strip()
                ans_pos = re.search(r'答案[：:]', wrong_suffix)
                if ans_pos:
                    wrong_suffix = wrong_suffix[:ans_pos.start()].strip()

            return {
                "cot": correct_prefix + "\n" + wrong_suffix,
                "answer": parsed["answer"],
                "error_step": error_idx,
            }

        except Exception as e:
            logger.warning(f"SCDPO failed (attempt {attempt+1}/3): {e}")
            time.sleep(2 ** attempt)

    return None
