"""
Dataset 类定义
适配 SFT 和 DPO 两种训练模式
"""
import json
import torch
from typing import Optional
from torch.utils.data import Dataset


class MathDataset(Dataset):
    """
    数学应用题 SFT 数据集

    支持两种模式：
    1. 直接回答模式（baseline）：question → 数字答案
    2. CoT 模式：question → <think>推理</think><answer>答案</answer>
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        use_cot: bool = False,
    ):
        """
        Args:
            data_path: JSON 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
            use_cot: 是否使用 CoT 格式（数据中需含 cot 字段）
        """
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cot = use_cot

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        instruction = item.get("instruction", "请直接输出数字答案，不带单位。")
        question = item["question"]

        # 构建 prompt
        prompt = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # 构建 response
        if self.use_cot and "cot" in item:
            response = f"<think>{item['cot']}</think><answer>{item['answer']}</answer>"
        else:
            response = str(item["answer"])

        # 分词
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)
        response_ids = self.tokenizer(response, add_special_tokens=False)

        input_ids = (
            prompt_ids["input_ids"]
            + response_ids["input_ids"]
            + [self.tokenizer.pad_token_id]
        )
        attention_mask = (
            prompt_ids["attention_mask"]
            + response_ids["attention_mask"]
            + [1]
        )
        # 只对 response 部分计算损失
        labels = (
            [-100] * len(prompt_ids["input_ids"])
            + response_ids["input_ids"]
            + [self.tokenizer.pad_token_id]
        )

        # 截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MathDPODataset(Dataset):
    """
    DPO 偏好数据集

    数据格式：
    {
        "question": "...",
        "instruction": "...",
        "chosen": "正确推理+答案",
        "rejected": "错误推理或答案"
    }
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 768):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        instruction = item.get("instruction", "请一步一步思考，然后给出数字答案。")
        question = item["question"]

        prompt = (
            f"<|im_start|>system\n{instruction}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return {
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
