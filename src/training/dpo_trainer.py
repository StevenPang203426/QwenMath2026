"""
DPO 训练模块
使用 trl.DPOTrainer 在 SFT checkpoint 基础上进行偏好优化
"""
import json
import logging
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
from src.training.rl_utils import (
    add_tokenizer_kwarg,
    filter_supported_kwargs,
    render_chat_prompt,
    to_text,
)
from src.training.runtime import (
    apply_lora_from_config,
    finish_training_run,
    load_base_model,
    merge_adapter_if_present,
    save_best_checkpoint,
    start_training_run,
)
from src.utils.config import load_config

logger = logging.getLogger("math_solver.dpo_trainer")


def _load_dpo_dataset(data_path: str, tokenizer: AutoTokenizer) -> Dataset:
    """
    加载 DPO 偏好数据并转换为 HuggingFace Dataset

    Args:
        data_path: DPO 数据路径
        tokenizer: 用于渲染 chat template 的 tokenizer

    Returns:
        HuggingFace Dataset 对象
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    chosens = []
    rejecteds = []

    for item in data:
        instruction = to_text(item.get("instruction", "请一步一步思考，然后给出数字答案。"))
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": to_text(item["question"])},
        ]
        prompt = render_chat_prompt(tokenizer, messages)
        prompts.append(prompt)
        chosens.append(to_text(item["chosen"]))
        rejecteds.append(to_text(item["rejected"]))

    return Dataset.from_dict({
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    })


def train_dpo(config_path: str) -> None:
    """
    执行 DPO 训练

    Args:
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    start_training_run(config, default_run_name="dpo", default_tags=["dpo"])

    # 加载 SFT checkpoint 作为起点
    sft_checkpoint = getattr(config.model, "sft_checkpoint", None)
    if not sft_checkpoint:
        logger.info("无 SFT checkpoint，从基础模型开始 DPO")
    model, tokenizer = load_base_model(config)
    model = merge_adapter_if_present(model, sft_checkpoint, "SFT checkpoint")
    model = apply_lora_from_config(model, config)

    # 加载 ref model (DPO 需要)
    ref_model, _ = load_base_model(config)
    ref_model = merge_adapter_if_present(ref_model, sft_checkpoint, "reference SFT checkpoint")

    # 加载数据
    train_dataset = _load_dpo_dataset(config.data.train_path, tokenizer)
    logger.info(f"DPO 训练集大小: {len(train_dataset)}")

    # DPO 配置
    dpo_kwargs = {
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
        "beta": config.dpo.beta,
        "loss_type": getattr(config.dpo, "loss_type", "sigmoid"),
        "max_length": getattr(config.dpo, "max_length", 768),
        "max_prompt_length": getattr(config.dpo, "max_prompt_length", 256),
        "report_to": "wandb",
    }
    dpo_config = DPOConfig(**filter_supported_kwargs(DPOConfig, dpo_kwargs))

    # DPO Trainer
    trainer_kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": dpo_config,
        "train_dataset": train_dataset,
        "beta": config.dpo.beta,
        "loss_type": getattr(config.dpo, "loss_type", "sigmoid"),
        "max_length": getattr(config.dpo, "max_length", 768),
        "max_prompt_length": getattr(config.dpo, "max_prompt_length", 256),
    }
    add_tokenizer_kwarg(DPOTrainer, trainer_kwargs, tokenizer)
    trainer = DPOTrainer(**filter_supported_kwargs(DPOTrainer.__init__, trainer_kwargs))

    # 开始训练
    logger.info("开始 DPO 训练...")
    trainer.train()

    # 保存
    save_best_checkpoint(trainer, tokenizer, config.training.output_dir, "DPO 模型")

    finish_training_run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.dpo_trainer --config configs/cot/dpo.yaml")
        sys.exit(1)

    train_dpo(sys.argv[2])
