"""
表达式方案 GRPO 训练模块
使用 trl.GRPOTrainer + 表达式专用 6 维奖励函数
"""
import json
import logging
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from src.models.reward_expr import build_expr_reward_funcs
from src.training.rl_utils import (
    add_tokenizer_kwarg,
    drop_conflicting_grpo_generation_args,
    filter_supported_kwargs,
    render_chat_prompt,
    to_text,
)
from src.utils.config import load_config
from src.training.runtime import (
    apply_lora_from_config,
    finish_training_run,
    load_base_model,
    merge_adapter_if_present,
    save_best_checkpoint,
    start_training_run,
)

logger = logging.getLogger("math_solver.grpo_expr_trainer")


# 表达式方案的 system instruction
_EXPR_INSTRUCTION = (
    "请为以下数学题写出一个Python可直接计算的中缀数学表达式。"
    "用<expr></expr>标签包裹表达式，用<answer></answer>标签包裹计算结果。"
)


def _build_expr_grpo_dataset(data_path: str, tokenizer) -> Dataset:
    """
    构建表达式 GRPO 数据集

    每条数据只需要 prompt 和 gold_answer
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = []
    answers = []

    for item in data:
        messages = [
            {"role": "system", "content": _EXPR_INSTRUCTION},
            {"role": "user", "content": to_text(item["question"])},
        ]
        prompt = render_chat_prompt(tokenizer, messages)
        prompts.append(prompt)
        answers.append(to_text(item["answer"]))

    return Dataset.from_dict({
        "prompt": prompts,
        "answer": answers,
    })


def train_grpo_expr(config_path: str) -> None:
    """执行表达式 GRPO 训练"""
    config = load_config(config_path)
    start_training_run(config, default_run_name="grpo-expr", default_tags=["grpo", "expression"])

    # 加载模型（从表达式 SFT checkpoint 开始）
    sft_checkpoint = getattr(config.model, "sft_checkpoint", None)

    model, tokenizer = load_base_model(config)
    model = merge_adapter_if_present(model, sft_checkpoint, "表达式 SFT checkpoint")
    model = apply_lora_from_config(model, config)

    # 加载数据
    train_dataset = _build_expr_grpo_dataset(config.data.train_path, tokenizer)
    logger.info(f"表达式 GRPO 训练集大小: {len(train_dataset)}")
    eval_dataset = None
    val_path = getattr(config.data, "val_path", "")
    if val_path:
        eval_dataset = _build_expr_grpo_dataset(val_path, tokenizer)
        logger.info(f"表达式 GRPO 验证集大小: {len(eval_dataset)}")

    # 构建表达式专用奖励函数
    reward_funcs = build_expr_reward_funcs()

    # GRPO 配置
    grpo_cfg = config.grpo
    grpo_kwargs = {
        "output_dir": config.training.output_dir,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "num_train_epochs": config.training.num_train_epochs,
        "max_steps": getattr(config.training, "max_steps", -1),
        "learning_rate": config.training.learning_rate,
        "warmup_ratio": getattr(config.training, "warmup_ratio", 0.1),
        "lr_scheduler_type": getattr(config.training, "lr_scheduler_type", "cosine"),
        "logging_steps": config.training.logging_steps,
        "save_strategy": getattr(config.training, "save_strategy", "steps"),
        "eval_strategy": getattr(config.training, "eval_strategy", "steps"),
        "eval_steps": getattr(config.training, "eval_steps", 100),
        "save_steps": getattr(config.training, "save_steps", 100),
        "save_total_limit": getattr(config.training, "save_total_limit", 3),
        "bf16": config.training.bf16,
        "gradient_checkpointing": getattr(config.training, "gradient_checkpointing", False),
        # GRPO 特有参数
        "num_generations": grpo_cfg.num_generations,
        "num_generations_eval": getattr(grpo_cfg, "num_generations_eval", None),
        "max_completion_length": getattr(grpo_cfg, "max_completion_length", 128),
        "max_prompt_length": getattr(grpo_cfg, "max_prompt_length", 256),
        "temperature": grpo_cfg.temperature,
        "top_p": getattr(grpo_cfg, "top_p", 1.0),
        "report_to": "wandb",
        # 加速参数
        "use_vllm": getattr(grpo_cfg, "use_vllm", False),
        "vllm_mode": getattr(grpo_cfg, "vllm_mode", "colocate"),
        "vllm_gpu_memory_utilization": getattr(grpo_cfg, "vllm_gpu_memory_utilization", 0.3),
        "vllm_max_model_length": getattr(grpo_cfg, "vllm_max_model_length", None),
        "generation_batch_size": getattr(grpo_cfg, "generation_batch_size", None),
        "steps_per_generation": getattr(grpo_cfg, "steps_per_generation", None),
        "dataloader_num_workers": getattr(config.training, "dataloader_num_workers", 0),
        "max_grad_norm": getattr(config.training, "max_grad_norm", 1.0),
    }
    grpo_kwargs = drop_conflicting_grpo_generation_args(grpo_kwargs, logger)
    grpo_config = GRPOConfig(**filter_supported_kwargs(GRPOConfig, grpo_kwargs))

    # GRPO Trainer
    trainer_kwargs = {
        "model": model,
        "args": grpo_config,
        "train_dataset": train_dataset,
        "reward_funcs": reward_funcs,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    add_tokenizer_kwarg(GRPOTrainer, trainer_kwargs, tokenizer)
    trainer = GRPOTrainer(**filter_supported_kwargs(GRPOTrainer.__init__, trainer_kwargs))

    # 开始训练
    logger.info("开始表达式 GRPO 训练...")
    trainer.train()

    # 保存
    save_best_checkpoint(trainer, tokenizer, config.training.output_dir, "表达式 GRPO 模型")

    finish_training_run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.grpo_expr_trainer --config configs/expr/grpo_expr.yaml")
        sys.exit(1)

    train_grpo_expr(sys.argv[2])
