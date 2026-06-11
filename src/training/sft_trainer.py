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
from src.utils.config import load_config
from src.training.runtime import (
    finish_training_run,
    load_lora_base_model,
    save_best_checkpoint,
    start_training_run,
)
from src.utils.metrics import compute_accuracy
from src.data.answer_extractor import extract_answer
from src.inference.expr_predictor import expr_predict_single

logger = logging.getLogger("math_solver.sft_trainer")


class EvalAccuracyCallback(TrainerCallback):
    """在验证时计算正确率的回调"""

    def __init__(self, val_data, model, tokenizer, use_cot=False, target_format="auto"):
        self.val_data = val_data
        self.model = model
        self.tokenizer = tokenizer
        self.use_cot = use_cot
        self.target_format = target_format

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
                    max_new_tokens=128 if self.target_format == "expression" else (512 if self.use_cot else 256),
                    do_sample=False,
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                if self.target_format == "expression":
                    pred = expr_predict_single(response, item["question"])["answer"]
                elif self.use_cot:
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
    start_training_run(config)

    # 加载模型并应用 LoRA
    model, tokenizer = load_lora_base_model(config)

    # 判断训练目标格式
    target_format = getattr(config.data, "target_format", "auto")
    use_cot = target_format == "cot" or (target_format == "auto" and "cot" in config.data.train_path)

    # 加载数据
    train_dataset = MathDataset(
        data_path=config.data.train_path,
        tokenizer=tokenizer,
        max_length=config.data.max_length,
        use_cot=use_cot,
        target_format=target_format,
    )
    logger.info(f"训练集大小: {len(train_dataset)}, 目标格式: {target_format}, 使用 CoT: {use_cot}")

    # 加载验证数据（如果有）
    val_data = None
    val_dataset = None
    val_path = getattr(config.data, "val_path", "")
    if not val_path:
        val_path = config.data.train_path.replace("train", "val").replace("raw/", "splits/")
    try:
        with open(val_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        val_dataset = MathDataset(
            data_path=val_path,
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            use_cot=use_cot,
            target_format=target_format,
        )
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
        eval_strategy=getattr(config.training, "eval_strategy", "steps" if val_dataset else "no"),
        eval_steps=getattr(config.training, "eval_steps", 500),
        save_strategy=getattr(config.training, "save_strategy", "steps"),
        save_steps=getattr(config.training, "save_steps", 500),
        save_total_limit=getattr(config.training, "save_total_limit", 3),
        per_device_eval_batch_size=getattr(
            config.training,
            "per_device_eval_batch_size",
            config.training.per_device_train_batch_size,
        ),
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        report_to="wandb",
        save_on_each_node=True,
    )

    # 回调
    callbacks = []
    if val_data:
        callbacks.append(EvalAccuracyCallback(val_data, model, tokenizer, use_cot, target_format))

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=callbacks,
    )

    # 开始训练
    logger.info("开始 SFT 训练...")
    trainer.train()

    # 保存最终模型
    save_best_checkpoint(trainer, tokenizer, config.training.output_dir, "模型")

    finish_training_run()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3 or sys.argv[1] != "--config":
        print("用法: python -m src.training.sft_trainer --config configs/cot/sft_baseline.yaml")
        sys.exit(1)

    train_sft(sys.argv[2])
