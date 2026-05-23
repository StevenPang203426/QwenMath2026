"""
统一推理接口
支持各种模型配置的预测
"""
import torch
import logging
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.answer_extractor import extract_answer

logger = logging.getLogger("math_solver.predictor")


class MathPredictor:
    """
    数学应用题预测器

    统一处理各方案的推理逻辑
    """

    def __init__(
        self,
        model,
        tokenizer,
        use_cot: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ):
        """
        Args:
            model: 语言模型
            tokenizer: 分词器
            use_cot: 是否使用 CoT 推理
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否采样
        """
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.device = next(model.parameters()).device

    def predict_single(
        self,
        question: str,
        instruction: Optional[str] = None,
    ) -> dict:
        """
        预测单道题

        Args:
            question: 题目文本
            instruction: 指令（可选）

        Returns:
            {"raw_output": 原始输出, "answer": 提取的答案}
        """
        if instruction is None:
            if self.use_cot:
                instruction = (
                    "请一步一步思考，然后给出数字答案。"
                    "用<think></think>标签包裹推理过程，"
                    "用<answer></answer>标签包裹最终数字答案。"
                )
            else:
                instruction = "这是小学数学1-6年级的校内题目，无需进行分析，请直接输出数字答案，不带单位。"

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
            )

        # 提取生成的部分
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        raw_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # 提取答案
        if self.use_cot:
            answer = extract_answer(raw_output)
        else:
            answer = raw_output.strip()

        return {
            "raw_output": raw_output,
            "answer": answer,
        }

    def predict_batch(
        self,
        questions: list[dict],
    ) -> list[dict]:
        """
        批量预测

        Args:
            questions: 题目列表，每项含 "question" 和可选 "instruction"

        Returns:
            预测结果列表
        """
        results = []
        for item in questions:
            result = self.predict_single(
                question=item["question"],
                instruction=item.get("instruction"),
            )
            result["id"] = item.get("id", "")
            results.append(result)
        return results
