"""
SFT 训练模块
支持 Baseline 直接回答 和 CoT 推理两种模式
"""
import json
import logging
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

from src.data.dataset import MathDataset
from src.models.model_loader import load_model_and_tokenizer, apply_lora
from src.utils.config import load_config, parse_args_with_config
from src.utils.seed import set_seed
from src.utils.logger import setup_wandb, finish_wandb
from src.utils.metrics import compute_accuracy
from src.data.answer_extractor import extract_answer

logger = logging.getLogger("math_solver.sft_trainer")


class EvalAccuracyCallback(TrainerCallback):
    """在验证时计算正确率的回调"""

    def __init__(self, val_data, model, tokenizer, use_cot=False):
        self.val_data = val_data
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot

    def on_evaluate(self, args, state, control, **kwargs):
        """评估时采样部分验证集计算正确率"""
        import torch
        sample_size = min(100, len(self.val_data))
        sample = self.val_data[:sample_size]

        predictions = []
        references = []

        self.model.eval()
        with torch.no_grad():
            for item in sample:
                instruction = item.get("instruction", "请直接输出数字答案。")
                messages = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": item["question"]},
                ]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256 if not self.use_cot else 512,
                    do_sample=False,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                if self.use_cot:
                    pred = extract_answer(response)
                else:
                    pred = response.strip()

                predictions.append(pred)
                references.append(str(item["answer"]))

        result = compute_accuracy(predictions, references)
        logger.info(f"验证集正确率 (sample={sample_size}): {result['accuracy']:.4f}")

        try:
            import wandb
            if wandb.run:
                wandb.log({"eval/accuracy": result["accuracy"]}, step=state.global_step)
        except (ImportError, Exception):
            pass


def train_sft(config_path: str) -> None:
    """
    执行 SFT 训练

    Args:
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    set_seed(config.training.seed)

    # 初始化 wandb
    setup_wandb(
        project=config.logging.project,
        run_name=getattr(config.logging, "run_name", None),
        config=config.to_dict(),
        tags=getattr(config.logging, "tags", None),
    )

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_name=config.model.name,
        cache_dir=config.model.cache_dir,
        torch_dtype=config.model.torch_dtype,
    )

    # 应用 LoRA
    model = apply_lora(
        model,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
    )

    # 判断是否使用 CoT
    use_cot = "cot" in config.data.train_path

    # 加载数据
    train_dataset = MathDataset(
        data_path=config.data.train_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        use_cot=use_cot,
    )
    logger.info(f"训练集大小: {len(train_dataset)}, 使用 CoT: {use_cot}")

    # 加载验证数据（如果有）
    val_data = None
    val_path = config.data.train_path.replace("train", "val").replace("raw/", "splits/")
    try:
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        logger.info(f"验证集大小: {len(val_data)}")
    except FileNotFoundError:
        logger.info("未找到验证集，跳过验证")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.training.output_dir,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        num_train_epochs=config.training.num_train_epochs,
        learning_rate=config.training.learning_rate,
        warmup_ratio=getattr(config.training, "warmup_ratio", 0.05),
        lr_scheduler_type=getattr(config.training, "lr_scheduler_type", "cosine"),
        logging_steps=config.training.logging_steps,
        save_strategy=getattr(config.training, "save_strategy", "steps"),
        save_steps=getattr(config.training, "save_steps", 500),
        save_total_limit=getattr(config.training, "save_total_limit", 3),
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        report_to="wandb",
        save_on_each_node=True,
    )

    # 回调
    callbacks = []
    if val_data:
        callbacks.append(EvalAccuracyCallback(val_data, model, tokenizer, use_cot))

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=callbacks,
    )

    # 开始训练
    logger.info("开始 SFT 训练...")
    trainer.train()

    # 保存最终模型
    best_dir = f"{config.training.output_dir}/best"
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"模型已保存至: {best_dir}")

    finish_wandb()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.sft_trainer --config configs/sft_baseline.yaml")
        sys.exit(1)

    train_sft(sys.argv[2])
