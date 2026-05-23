"""
数据增强模块
通过修改题目中的数字来生成新的训练数据
"""
import re
import json
import random
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger("math_solver.data_augmentor")


def _extract_numbers(text: str) -> list[tuple[str, int, int]]:
    """
    从文本中提取所有数字及其位置

    Returns:
        [(数字字符串, 起始位置, 结束位置), ...]
    """
    pattern = r'\d+\.?\d*'
    return [(m.group(), m.start(), m.end()) for m in re.finditer(pattern, text)]


def _perturb_number(num_str: str, ratio: float = 0.3) -> str:
    """
    对数字做随机扰动

    Args:
        num_str: 原始数字字符串
        ratio: 扰动幅度比例

    Returns:
        扰动后的数字字符串
    """
    try:
        val = float(num_str)
    except ValueError:
        return num_str

    if val == 0:
        return str(random.randint(1, 10))

    # 整数还是小数
    is_int = '.' not in num_str

    # 随机扰动
    delta = val * ratio
    new_val = val + random.uniform(-delta, delta)

    if is_int:
        new_val = max(1, round(new_val))
        return str(int(new_val))
    else:
        new_val = round(new_val, len(num_str.split('.')[-1]))
        return str(new_val)


def augment_by_number_change(
    input_path: str,
    output_path: str,
    num_augments_per_sample: int = 1,
    ratio: float = 0.3,
    seed: int = 42,
) -> None:
    """
    通过修改题目中的数字来生成增强数据

    注意：增强后的数据需要用 DeepSeek API 重新验证答案正确性

    Args:
        input_path: 输入数据路径
        output_path: 输出路径
        num_augments_per_sample: 每个样本生成的增强数量
        ratio: 数字扰动幅度
        seed: 随机种子
    """
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    augmented = []
    aug_id = len(data)

    for item in data:
        question = item["question"]
        numbers = _extract_numbers(question)

        if len(numbers) < 2:
            # 数字太少，跳过
            continue

        for _ in range(num_augments_per_sample):
            new_question = question
            offset = 0

            for num_str, start, end in numbers:
                new_num = _perturb_number(num_str, ratio)
                new_question = (
                    new_question[:start + offset]
                    + new_num
                    + new_question[end + offset:]
                )
                offset += len(new_num) - len(num_str)

            augmented.append({
                "id": str(aug_id),
                "question": new_question,
                "answer": "",  # 需要 API 重新计算
                "instruction": item.get("instruction", ""),
                "source_id": item["id"],
                "augmented": True,
            })
            aug_id += 1

    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)

    logger.info(f"数据增强完成: 原始 {len(data)} 条 → 增强 {len(augmented)} 条")
    logger.info(f"注意: 增强数据的 answer 字段为空，需要用 API 重新计算")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据增强")
    parser.add_argument("--input", default="data/raw/train.json")
    parser.add_argument("--output", default="data/processed/train_augmented.json")
    parser.add_argument("--num_augments", type=int, default=1)
    parser.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    augment_by_number_change(args.input, args.output, args.num_augments, args.ratio)
