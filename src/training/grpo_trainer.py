"""
GRPO 训练模块
使用 trl.GRPOTrainer 实现组相对策略优化
参考: DeepSeek-R1, huggingface/open-r1
"""
import json
import re
import logging
from datasets import Dataset
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

from src.models.model_loader import load_model_and_tokenizer, apply_lora
from src.models.reward import build_reward_funcs
from src.data.answer_extractor import extract_answer
from src.training.rl_utils import (
    add_tokenizer_kwarg,
    filter_supported_kwargs,
    render_chat_prompt,
    to_text,
)
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import normalize_number
from src.utils.logger import setup_wandb, finish_wandb

logger = logging.getLogger("math_solver.grpo_trainer")


def _build_grpo_dataset(data_path: str, tokenizer) -> Dataset:
    """
    构建 GRPO 数据集

    每条数据只需要 prompt 和 gold_answer

    Args:
        data_path: 训练数据路径
        tokenizer: 用于渲染 chat template 的 tokenizer

    Returns:
        HuggingFace Dataset
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    answers = []

    for item in data:
        instruction = "请一步一步思考，然后给出数字答案。用<think></think>标签包裹推理过程，用<answer></answer>标签包裹最终数字答案。"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": to_text(item["question"])},
        ]
        prompt = render_chat_prompt(tokenizer, messages)
        prompts.append(prompt)
        answers.append(to_text(item["answer"]))

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
    })


def _make_reward_funcs(config):
    """
    构建 GRPO 的五维奖励函数列表

    返回 5 个独立奖励函数，TRL GRPOTrainer 自动求和并分维度记录日志：
      1. 正确性   (-0.5 ~ +1.0)
      2. 格式标签  ( 0.0 ~ +0.2)
      3. 逻辑词   (-0.3 ~ +0.95)
      4. 长度正则  (-0.3 ~  0.0)
      5. 无LaTeX  (-0.3 ~  0.0)

    Args:
        config: 配置对象（保留参数以兼容调用方式）

    Returns:
        奖励函数列表
    """
    return build_reward_funcs()


def train_grpo(config_path: str) -> None:
    """
    执行 GRPO 训练

    Args:
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    set_seed(config.training.seed)

    # 初始化 wandb
    setup_wandb(
        project=config.logging.project,
        run_name=getattr(config.logging, "run_name", "grpo"),
        config=config.to_dict(),
        tags=getattr(config.logging, "tags", ["grpo"]),
    )

    # 加载模型（从 SFT checkpoint 开始）
    sft_checkpoint = getattr(config.model, "sft_checkpoint", None)

    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=config.model.torch_dtype,
    )

    if sft_checkpoint:
        logger.info(f"从 SFT checkpoint 加载: {sft_checkpoint}")
        model = PeftModel.from_pretrained(model, model_id=sft_checkpoint)
        model = model.merge_and_unload()

    # 应用 LoRA
    model = apply_lora(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
    )

    # 加载数据
    train_dataset = _build_grpo_dataset(config.data.train_path, tokenizer)
    logger.info(f"GRPO 训练集大小: {len(train_dataset)}")

    # 构建奖励函数
    reward_funcs = _make_reward_funcs(config)

    # GRPO 配置
    grpo_cfg = config.grpo
    grpo_kwargs = {
        "output_dir": config.training.output_dir,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "num_train_epochs": config.training.num_train_epochs,
        "learning_rate": config.training.learning_rate,
        "warmup_ratio": getattr(config.training, "warmup_ratio", 0.1),
        "lr_scheduler_type": getattr(config.training, "lr_scheduler_type", "cosine"),
        "logging_steps": config.training.logging_steps,
        "save_steps": getattr(config.training, "save_steps", 200),
        "save_total_limit": getattr(config.training, "save_total_limit", 3),
        "bf16": config.training.bf16,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        # GRPO 特有参数
        "num_generations": grpo_cfg.num_generations,
        "max_completion_length": getattr(grpo_cfg, "max_completion_length", 512),
        "max_prompt_length": getattr(grpo_cfg, "max_prompt_length", 256),
        "temperature": grpo_cfg.temperature,
        "top_p": getattr(grpo_cfg, "top_p", 1.0),
        "report_to": "wandb",
    }
    grpo_config = GRPOConfig(**filter_supported_kwargs(GRPOConfig, grpo_kwargs))

    # GRPO Trainer
    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": train_dataset,
        "reward_funcs": reward_funcs,
    }
    add_tokenizer_kwarg(GRPOTrainer, trainer_kwargs, tokenizer)
    trainer = GRPOTrainer(**filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs))

    # 开始训练
    logger.info("开始 GRPO 训练...")
    trainer.train()

    # 保存
    best_dir = f"{config.training.output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"GRPO 模型已保存至: {best_dir}")

    finish_wandb()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.grpo_trainer --config configs/grpo.yaml")
        sys.exit(1)

    train_grpo(sys.argv[2])
