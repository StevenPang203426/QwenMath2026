"""
多路推理投票融合模块

策略：
  A. 同模型多温度采样（5次，temperature 0.1~0.7）
  B. 多 checkpoint 投票（SFT / DPO / GRPO）
  C. Majority vote 选择最终答案

规则：
  - 多数一致（≥3/N 相同）→ 直接采用
  - 全部不同 → 选 GRPO 答案（RL 训练后最强）
  - 无时间限制，可充分采样
  - 不使用外部 API（竞赛不允许）
"""
import json
import csv
import logging
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

from src.models.model_loader import load_model_and_tokenizer, load_peft_model
from src.inference.predictor import MathPredictor
from src.inference.answer_postprocessor import postprocess_answer
from src.inference.question_classifier import build_adaptive_prompt
from src.data.answer_extractor import extract_answer
from src.utils.config import load_config
from src.utils.metrics import normalize_number
from src.utils.seed import set_seed

logger = logging.getLogger("math_solver.ensemble_infer")


# ============================================================
# 投票逻辑
# ============================================================

def normalize_for_vote(answer: str) -> str:
    """
    将答案标准化用于投票比较

    例如 "3.0" 和 "3" 应被视为相同答案
    """
    normed = normalize_number(answer)
    if normed is not None:
        return normed
    return answer.strip()


def majority_vote(answers: List[Tuple[str, str, float]]) -> str:
    """
    加权多数投票

    Args:
        answers: [(来源名称, 答案字符串, 权重), ...]

    Returns:
        投票胜��的答��（原始格式）
    """
    if not answers:
        return "0"

    # 按标准化答案分组
    groups: Dict[str, Dict] = {}
    for source, answer, weight in answers:
        key = normalize_for_vote(answer)
        if key not in groups:
            groups[key] = {
                "votes": 0,
                "weight_sum": 0.0,
                "sources": [],
                "raw_answers": [],
            }
        groups[key]["votes"] += 1
        groups[key]["weight_sum"] += weight
        groups[key]["sources"].append(source)
        groups[key]["raw_answers"].append(answer)

    # 按 (票数, 权重和) 排序
    sorted_groups = sorted(
        groups.items(),
        key=lambda x: (x[1]["votes"], x[1]["weight_sum"]),
        reverse=True,
    )

    # 返回得票最多的答案（取该组中权重最高来源的原始格式）
    winner = sorted_groups[0][1]
    # 优先返回高权重来源的答案格式
    best_idx = 0
    best_weight = 0
    for i, source in enumerate(winner["sources"]):
        w = answers[[a[0] for a in answers].index(source)][2] if source in [a[0] for a in answers] else 1.0
        if w > best_weight:
            best_weight = w
            best_idx = i
    return winner["raw_answers"][best_idx]


# ============================================================
# 多温度采样
# ============================================================

def multi_temperature_sample(
    predictor: MathPredictor,
    question: str,
    instruction: str,
    temperatures: List[float] = None,
    n_samples: int = 5,
) -> List[Tuple[str, str]]:
    """
    同模型多温度采样

    Args:
        predictor: 推理器
        question: 题目文本
        instruction: system instruction
        temperatures: 温度列表
        n_samples: 采样次数

    Returns:
        [(来��标签, 答案), ...]
    """
    if temperatures is None:
        temperatures = [0.1, 0.3, 0.5, 0.6, 0.7][:n_samples]

    results = []
    for i, temp in enumerate(temperatures):
        # 设置温度
        old_temp = predictor.temperature
        old_do_sample = predictor.do_sample
        predictor.temperature = temp
        predictor.do_sample = (temp > 0.0)

        result = predictor.predict_single(
            question=question,
            instruction=instruction,
        )
        answer = result.get("answer", "")
        results.append((f"sample_t{temp}", answer))

        # 恢复
        predictor.temperature = old_temp
        predictor.do_sample = old_do_sample

    return results


# ============================================================
# 多 Checkpoint 推理
# ============================================================

def load_checkpoint_predictors(
    config,
    checkpoint_configs: List[Dict],
) -> List[Tuple[str, MathPredictor, float]]:
    """
    加载多个 checkpoint 的预测器

    Args:
        config: 全局配置
        checkpoint_configs: [{name, adapter_path, weight}, ...]

    Returns:
        [(名称, 预测器, 权重), ...]
    """
    predictors = []
    gen_cfg = config.generation

    for ckpt in checkpoint_configs:
        name = ckpt["name"]
        adapter_path = ckpt["adapter_path"]
        weight = ckpt.get("weight", 1.0)

        logger.info(f"加载 checkpoint: {name} ({adapter_path})")

        base_model = ckpt.get("base_model", config.model.name)

        if adapter_path and Path(adapter_path).exists():
            model, tokenizer = load_peft_model(
                base_model_name=base_model,
                adapter_path=adapter_path,
                torch_dtype="bfloat16",
            )
        else:
            logger.warning(f"Checkpoint 不存在: {adapter_path}，跳过")
            continue

        predictor = MathPredictor(
            model=model,
            tokenizer=tokenizer,
            use_cot=True,
            max_new_tokens=getattr(gen_cfg, "max_new_tokens", 512),
            temperature=0.1,
            do_sample=False,
        )
        predictors.append((name, predictor, weight))

    return predictors


# ============================================================
# 主入口
# ============================================================

def run_ensemble_inference(config_path: str) -> str:
    """
    执行集成推理

    Args:
        config_path: 配置文件路径

    Returns:
        输出文件路径
    """
    config = load_config(config_path)
    set_seed(42)

    # 集成配置
    ensemble_cfg = getattr(config, "ensemble", None)
    if ensemble_cfg is None:
        logger.error("配置文件中缺少 ensemble 节")
        return ""

    n_samples = getattr(ensemble_cfg, "n_samples", 5)
    temperatures = getattr(ensemble_cfg, "temperatures", [0.1, 0.3, 0.5, 0.6, 0.7])
    checkpoint_configs = getattr(ensemble_cfg, "checkpoints", [])

    # 权重配置
    weights = {
        "grpo": 1.5,
        "dpo": 1.2,
        "sft": 1.0,
        "sample": 0.8,  # 采样权重较低
    }

    base_instruction = (
        "请一步一步思考，然后给出数字答案。"
        "用<think></think>标签包裹推理过程，用<answer></answer>标签包裹最终数字答案。"
    )

    # 加载主模型（用于多温度采样）
    primary_ckpt = checkpoint_configs[0] if checkpoint_configs else None
    if primary_ckpt:
        primary_model, primary_tokenizer = load_peft_model(
            base_model_name=config.model.name,
            adapter_path=primary_ckpt["adapter_path"],
            torch_dtype="bfloat16",
        )
    else:
        primary_model, primary_tokenizer = load_model_and_tokenizer(
            model_name=config.model.name,
            torch_dtype="bfloat16",
        )

    gen_cfg = config.generation
    primary_predictor = MathPredictor(
        model=primary_model,
        tokenizer=primary_tokenizer,
        use_cot=True,
        max_new_tokens=getattr(gen_cfg, "max_new_tokens", 512),
        temperature=0.1,
        do_sample=False,
    )

    # 加载其他 checkpoint
    other_predictors = []
    if len(checkpoint_configs) > 1:
        other_predictors = load_checkpoint_predictors(config, checkpoint_configs[1:])

    # 加载测试数据
    test_path = getattr(config.data, "test_path", "data/raw/test.json")
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info(f"测试集大小: {len(test_data)}, 采样次数: {n_samples}, checkpoints: {len(checkpoint_configs)}")

    # 批量推理
    results = []
    for item in tqdm(test_data, desc="集成推理"):
        question = item["question"]
        if isinstance(question, list):
            question_text = question[0].get("content", "") if question else ""
        else:
            question_text = str(question)

        # 自��应 prompt
        adaptive_instruction = build_adaptive_prompt(question_text, base_instruction)

        # 收集所有答案
        all_answers = []  # [(source, answer, weight)]

        # 策略 A：多温度采样
        samples = multi_temperature_sample(
            primary_predictor, question_text, adaptive_instruction,
            temperatures=temperatures[:n_samples],
            n_samples=n_samples,
        )
        primary_weight = weights.get(primary_ckpt.get("name", "sft") if primary_ckpt else "sft", 1.0)
        for source, answer in samples:
            all_answers.append((source, answer, weights["sample"]))
        # 第一个采样（最低温度）用主���型权重
        if all_answers:
            all_answers[0] = (all_answers[0][0], all_answers[0][1], primary_weight)

        # 策略 B：多 checkpoint
        for name, predictor, weight in other_predictors:
            result = predictor.predict_single(
                question=question_text,
                instruction=adaptive_instruction,
            )
            answer = result.get("answer", "")
            all_answers.append((name, answer, weight))

        # 投票
        voted_answer = majority_vote(all_answers)

        # 后处理
        final_answer = postprocess_answer(voted_answer, question_text)

        results.append({
            "id": item["id"],
            "answer": final_answer,
            "n_candidates": len(all_answers),
        })

    # 生成 submit.csv
    output_dir = getattr(config.output, "dir", "outputs/submissions")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/submit_ensemble.csv"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            writer.writerow([r["id"], r["answer"]])

    logger.info(f"集成提交文件已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/inference/infer_ensemble.yaml"
    run_ensemble_inference(config_path)
