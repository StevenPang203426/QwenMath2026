"""Shared runtime helpers for training entrypoints."""
import logging
from typing import Any

from peft import PeftModel

from src.models.model_loader import apply_lora, load_model_and_tokenizer
from src.utils.logger import finish_wandb, setup_wandb
from src.utils.seed import set_seed

logger = logging.getLogger("math_solver.training_runtime")


def start_training_run(config: Any, default_run_name: str | None = None, default_tags: list[str] | None = None) -> None:
    """Apply seed and initialize the configured experiment logger."""
    set_seed(config.training.seed)
    setup_wandb(
        project=config.logging.project,
        run_name=getattr(config.logging, "run_name", default_run_name),
        config=config.to_dict(),
        tags=getattr(config.logging, "tags", default_tags),
    )


def load_base_model(config: Any):
    """Load the configured base model and tokenizer."""
    return load_model_and_tokenizer(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=config.model.torch_dtype,
    )


def merge_adapter_if_present(model: Any, adapter_path: str | None, label: str) -> Any:
    """Merge an adapter into a base model when a checkpoint path is configured."""
    if not adapter_path:
        return model
    logger.info("从 %s 加载: %s", label, adapter_path)
    peft_model = PeftModel.from_pretrained(model, model_id=adapter_path)
    return peft_model.merge_and_unload()


def apply_lora_from_config(model: Any, config: Any) -> Any:
    """Attach the project LoRA adapter defined by the config."""
    return apply_lora(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
    )


def load_lora_base_model(config: Any):
    """Load the raw base model and attach a fresh trainable LoRA adapter."""
    model, tokenizer = load_base_model(config)
    return apply_lora_from_config(model, config), tokenizer


def save_best_checkpoint(trainer: Any, tokenizer: Any, output_dir: str, label: str) -> str:
    """Save trainer and tokenizer under the project-standard best directory."""
    best_dir = f"{output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info("%s已保存至: %s", label, best_dir)
    return best_dir


def finish_training_run() -> None:
    """Finish the configured experiment logger."""
    finish_wandb()
