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
from src.data.answer_extractor import extract_answer
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.metrics import normalize_number
from src.utils.logger import setup_wandb, finish_wandb

logger = logging.getLogger("math_solver.grpo_trainer")


def _build_grpo_dataset(data_path: str) -> Dataset:
    """
    构建 GRPO 数据集

    每条数据只需要 prompt 和 gold_answer

    Args:
        data_path: 训练数据路径

    Returns:
        HuggingFace Dataset
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    answers = []

    for item in data:
        instruction = "请一步一步思考，然后给出数字答案。用<think></think>标签包裹推理过程，用<answer></answer>标签包裹最终数字答案。"
        prompt = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": item["question"]},
        ]
        prompts.append(prompt)
        answers.append(str(item["answer"]))

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
    })


def _make_reward_funcs(config):
    """
    构建 GRPO 的奖励函数列表

    Args:
        config: 配置对象

    Returns:
        奖励函数列表
    """
    reward_cfg = config.reward if hasattr(config, 'reward') else None
    fmt_weight = getattr(reward_cfg, 'format_weight', 0.2) if reward_cfg else 0.2
    cor_weight = getattr(reward_cfg, 'correctness_weight', 1.0) if reward_cfg else 1.0

    def format_reward_fn(completions, **kwargs):
        """格式奖励：检查是否包含 <think> 和 <answer> 标签"""
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else completion
            has_think = bool(re.search(r'<think>.*?</think>', text, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', text, re.DOTALL))
            reward = fmt_weight * (1.0 if (has_think and has_answer) else 0.0)
            rewards.append(reward)
        return rewards

    def correctness_reward_fn(completions, answer=None, **kwargs):
        """正确性奖励：检查答案是否正确"""
        rewards = []
        for i, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else completion
            gold = answer[i] if answer else None

            if gold is None:
                rewards.append(0.0)
                continue

            predicted = extract_answer(text)
            pred_norm = normalize_number(predicted)
            gold_norm = normalize_number(str(gold))

            is_correct = (
                pred_norm is not None
                and gold_norm is not None
                and pred_norm == gold_norm
            )
            rewards.append(cor_weight if is_correct else 0.0)
        return rewards

    return [format_reward_fn, correctness_reward_fn]


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
    train_dataset = _build_grpo_dataset(config.data.train_path)
    logger.info(f"GRPO 训练集大小: {len(train_dataset)}")

    # 构建奖励函数
    reward_funcs = _make_reward_funcs(config)

    # GRPO 配置
    grpo_cfg = config.grpo
    grpo_config = GRPOConfig(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=getattr(config.training, "warmup_ratio", 0.1),
        lr_scheduler_type=getattr(config.training, "lr_scheduler_type", "cosine"),
        logging_steps=config.training.logging_steps,
        save_steps=getattr(config.training, "save_steps", 200),
        save_total_limit=getattr(config.training, "save_total_limit", 3),
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        # GRPO 特有参数
        num_generations=grpo_cfg.num_generations,
        max_completion_length=getattr(grpo_cfg, "max_completion_length", 512),
        max_prompt_length=getattr(grpo_cfg, "max_prompt_length", 256),
        temperature=grpo_cfg.temperature,
        report_to="wandb",
    )

    # GRPO Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
    )

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
