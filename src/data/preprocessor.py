"""
数据预处理模块
训练/验证集划分、格式转换
"""
import json
import random
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("math_solver.preprocessor")


def _to_text(value: Any) -> str:
    """Normalize raw dataset fields that may be strings or content lists."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_to_text(part) for part in value)
    if isinstance(value, dict):
        if "text" in value:
            return _to_text(value["text"])
        if "content" in value:
            return _to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def split_train_val(
    input_path: str,
    train_output: str,
    val_output: str,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[int, int]:
    """
    将训练数据划分为训练集和验证集

    Args:
        input_path: 原始训练数据路径
        train_output: 训练集输出路径
        val_output: 验证集输出路径
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (训练集大小, 验证集大小)
    """
    random.seed(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)
    split_idx = int(len(data) * (1 - val_ratio))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(val_output).parent.mkdir(parents=True, exist_ok=True)

    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_output, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    logger.info(f"数据划分完成: 训练 {len(train_data)} 条, 验证 {len(val_data)} 条")
    return len(train_data), len(val_data)


def prepare_sft_data(
    cot_data_path: str,
    output_path: str,
    filter_matched_only: bool = True,
) -> int:
    """
    将 CoT 数据转换为 SFT 训练格式

    只保留 API 生成答案与标注答案一致的样本

    Args:
        cot_data_path: CoT 数据路径（data_builder 输出）
        output_path: SFT 格式数据输出路径
        filter_matched_only: 是否只保留答案匹配的样本

    Returns:
        输出样本数量
    """
    with open(cot_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sft_data = []
    for item in data:
        if filter_matched_only and not item.get("answer_match", True):
            continue

        sft_data.append({
            "id": item["id"],
            "question": item["question"],
            "instruction": "请一步一步思考，然后给出数字答案。",
            "answer": item["answer"],
            "cot": item.get("cot", ""),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    logger.info(f"SFT 数据准备完成: {len(sft_data)}/{len(data)} 条 "
                f"(过滤了 {len(data) - len(sft_data)} 条不匹配样本)")
    return len(sft_data)


def prepare_dpo_data(
    cot_data_path: str,
    output_path: str,
) -> int:
    """
    将 CoT 数据转换为 DPO 偏好对格式

    chosen = 正确推理 + 正确答案
    rejected = 错误推理 + 错误答案（来自 data_builder 的 wrong_cot）

    Args:
        cot_data_path: 含 wrong_cot 的 CoT 数据路径
        output_path: DPO 格式数据输出路径

    Returns:
        输出样本数量
    """
    with open(cot_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dpo_data = []
    for item in data:
        # 需要同时有正确和错误推理
        if not item.get("cot") or not item.get("wrong_cot"):
            continue
        if not item.get("answer_match", True):
            continue

        chosen = f"<think>{item['cot']}</think><answer>{item['answer']}</answer>"
        rejected = (
            f"<think>{item['wrong_cot']}</think>"
            f"<answer>{item.get('wrong_answer', '0')}</answer>"
        )

        dpo_data.append({
            "id": item["id"],
            "question": _to_text(item["question"]),
            "instruction": _to_text("请一步一步思考，然后给出数字答案。"),
            "chosen": chosen,
            "rejected": rejected,
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    logger.info(f"DPO 数据准备完成: {len(dpo_data)} 条偏好对")
    return len(dpo_data)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("--action", choices=["split", "sft", "dpo"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val_output", default="data/splits/val_split.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.action == "split":
        split_train_val(args.input, args.output, args.val_output)
    elif args.action == "sft":
        prepare_sft_data(args.input, args.output)
    elif args.action == "dpo":
        prepare_dpo_data(args.input, args.output)
