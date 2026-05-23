"""
统一模型加载模块
支持 base model、LoRA adapter、merged model 三种模式
"""
import torch
import logging
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

logger = logging.getLogger("math_solver.model_loader")


def load_model_and_tokenizer(
    model_name: str,
    cache_dir: str = "./model_cache",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple:
    """
    加载基础模型和分词器

    Args:
        model_name: 模型名称或路径
        cache_dir: 模型缓存目录
        torch_dtype: 数据类型
        device_map: 设备映射策略

    Returns:
        (model, tokenizer)
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info(f"加载模型: {model_name}")

    # 优先从本地缓存加载，否则从 ModelScope 下载
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            use_fast=False,
            trust_remote_code=True,
        )
    except OSError:
        logger.info(f"本地未找到，尝试从 ModelScope 下载...")
        from modelscope import snapshot_download
        model_dir = snapshot_download(model_name, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=False,
            trust_remote_code=True,
        )
        model_name = model_dir

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # 确保 pad_token 设置
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"模型加载完成: {model.num_parameters()/1e6:.1f}M 参数")
    return model, tokenizer


def apply_lora(
    model,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
):
    """
    为模型添加 LoRA adapter

    Args:
        model: 基础模型
        r: LoRA 秩
        lora_alpha: LoRA alpha
        lora_dropout: dropout 比例
        target_modules: 目标模块列表

    Returns:
        添加 LoRA 后的模型
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_peft_model(
    base_model_name: str,
    adapter_path: str,
    cache_dir: str = "./model_cache",
    torch_dtype: str = "bfloat16",
    merge: bool = False,
) -> tuple:
    """
    加载带 LoRA adapter 的模型

    Args:
        base_model_name: 基础模型名称
        adapter_path: LoRA adapter 路径
        cache_dir: 缓存目录
        torch_dtype: 数据类型
        merge: 是否合并 LoRA 权重到基础模型

    Returns:
        (model, tokenizer)
    """
    model, tokenizer = load_model_and_tokenizer(
        base_model_name, cache_dir, torch_dtype
    )

    logger.info(f"加载 LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, model_id=adapter_path)

    if merge:
        logger.info("合并 LoRA 权重到基础模型...")
        model = model.merge_and_unload()

    return model, tokenizer
