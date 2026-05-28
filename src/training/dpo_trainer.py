"""
DPO 训练模块
使用 trl.DPOTrainer 在 SFT checkpoint 基础上进行偏好优化
"""
import json
import logging
import inspect
from typing import Any
from datasets import Dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer
from peft import PeftModel

from src.models.model_loader import load_model_and_tokenizer, apply_lora
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_wandb, finish_wandb

logger = logging.getLogger("math_solver.dpo_trainer")


def _to_text(value: Any) -> str:
    """Normalize JSON text fields before Arrow infers column types."""
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


def _filter_supported_kwargs(cls_or_fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs supported by the installed TRL version."""
    signature = inspect.signature(cls_or_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


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
        instruction = _to_text(item.get("instruction", "请一步一步思考，然后给出数字答案。"))
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": _to_text(item["question"])},
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)
        chosens.append(_to_text(item["chosen"]))
        rejecteds.append(_to_text(item["rejected"]))

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
    set_seed(config.training.seed)

    # 初始化 wandb
    setup_wandb(
        project=config.logging.project,
        run_name=getattr(config.logging, "run_name", "dpo"),
        config=config.to_dict(),
        tags=getattr(config.logging, "tags", ["dpo"]),
    )

    # 加载 SFT checkpoint 作为起点
    sft_checkpoint = getattr(config.model, "sft_checkpoint", None)
    if sft_checkpoint:
        logger.info(f"从 SFT checkpoint 加载: {sft_checkpoint}")
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model.name,
            cache_dir=config.model.cache_dir,
            torch_dtype=config.model.torch_dtype,
        )
        model = PeftModel.from_pretrained(model, model_id=sft_checkpoint)
        model = model.merge_and_unload()
        # 重新应用 LoRA 用于 DPO 训练
        model = apply_lora(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
        )
    else:
        logger.info("无 SFT checkpoint，从基础模型开始 DPO")
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model.name,
            cache_dir=config.model.cache_dir,
            torch_dtype=config.model.torch_dtype,
        )
        model = apply_lora(
            model,
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            lora_dropout=config.lora.lora_dropout,
            target_modules=config.lora.target_modules,
        )

    # 加载 ref model (DPO 需要)
    ref_model, _ = load_model_and_tokenizer(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=config.model.torch_dtype,
    )
    if sft_checkpoint:
        ref_model = PeftModel.from_pretrained(ref_model, model_id=sft_checkpoint)
        ref_model = ref_model.merge_and_unload()

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
    dpo_config = DPOConfig(**_filter_supported_kwargs(DPOConfig, dpo_kwargs))

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
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**_filter_supported_kwargs(DPOTrainer.__init__, trainer_kwargs))

    # 开始训练
    logger.info("开始 DPO 训练...")
    trainer.train()

    # 保存
    best_dir = f"{config.training.output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"DPO 模型已保存至: {best_dir}")

    finish_wandb()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.dpo_trainer --config configs/dpo.yaml")
        sys.exit(1)

    train_dpo(sys.argv[2])
