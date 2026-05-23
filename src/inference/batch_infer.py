"""
批量推理模块
对测试集进行批量预测并生成 submit.csv
"""
import json
import csv
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from src.models.model_loader import load_model_and_tokenizer, load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.cot_prompting import build_zero_shot_prompt, build_few_shot_prompt
from src.data.answer_extractor import extract_answer, batch_extract
from src.utils.config import load_config
from src.utils.seed import set_seed

logger = logging.getLogger("math_solver.batch_infer")


def run_inference(config_path: str) -> str:
    """
    执行批量推理并生成 submit.csv

    Args:
        config_path: 推理配置文件路径（configs/infer.yaml）

    Returns:
        输出文件路径
    """
    config = load_config(config_path)
    set_seed(42)

    active_method = config.active_method
    method_cfg = getattr(config.methods, active_method)
    gen_cfg = config.generation

    logger.info(f"推理方案: {active_method}")

    # 加载模型
    adapter_path = getattr(method_cfg, "adapter_path", None)
    if adapter_path:
        model, tokenizer = load_peft_model(
            base_model_name=method_cfg.base_model,
            adapter_path=adapter_path,
            torch_dtype="bfloat16",
        )
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name=method_cfg.base_model,
            torch_dtype="bfloat16",
        )

    use_cot = getattr(method_cfg, "use_cot", False)

    # 创建预测器
    predictor = MathPredictor(
        model=model,
        tokenizer=tokenizer,
        use_cot=use_cot,
        max_new_tokens=getattr(gen_cfg, "max_new_tokens", 512),
        temperature=getattr(gen_cfg, "temperature", 0.1),
        do_sample=getattr(gen_cfg, "do_sample", False),
    )

    # 加载测试数据
    test_path = "data/raw/test.json"
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"测试集大小: {len(test_data)}")

    # 批量推理
    results = []
    for item in tqdm(test_data, desc=f"推理 [{active_method}]"):
        question = item["question"]

        # 方案1 特殊处理：CoT 提示工程（不微调）
        if active_method == "cot_prompt":
            strategy = getattr(method_cfg, "cot_strategy", "few_shot")
            if strategy == "few_shot":
                messages = build_few_shot_prompt(question)
            else:
                messages = build_zero_shot_prompt(question)

            # 直接用 messages 推理
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            import torch
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=False,
                )
            output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
            raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            answer = extract_answer(raw_output)

            results.append({
                "id": item["id"],
                "raw_output": raw_output,
                "answer": answer,
            })
        else:
            # 其他方案：使用统一预测器
            result = predictor.predict_single(
                question=question,
                instruction=item.get("instruction"),
            )
            result["id"] = item["id"]
            results.append(result)

    # 生成 submit.csv
    output_dir = getattr(config.output, "dir", "outputs/submissions")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = getattr(config.output, "filename_template", "submit_{method}.csv")
    filename = filename.replace("{method}", active_method)
    output_path = f"{output_dir}/{filename}"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "ret"])
        for r in results:
            # 清理答案：去除换行等
            answer = str(r["answer"]).replace("\n", " ").strip()
            writer.writerow([r["id"], answer])

    logger.info(f"提交文件已生成: {output_path}")

    # 统计答案提取情况
    raw_outputs = [r["raw_output"] for r in results]
    batch_extract(raw_outputs)

    return output_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/infer.yaml"
    run_inference(config_path)
