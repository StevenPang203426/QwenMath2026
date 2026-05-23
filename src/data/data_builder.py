"""
CoT 数据构建模块
调用 DeepSeek V4 API（OpenAI SDK 兼容格式）为训练数据生成带推理步骤的答案
同时生成错误推理路径供 DPO 使用
"""
import json
import time
import os
import re
import logging
from typing import Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("math_solver.data_builder")

# DeepSeek API 配置
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"  # deepseek-v4-pro | deepseek-v4-flash


def _get_client(api_key: str):
    """获取 OpenAI 兼容客户端（复用连接）"""
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _call_deepseek_api(
    question: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    generate_wrong: bool = False,
    max_retries: int = 3,
    client=None,
) -> Optional[dict]:
    """
    调用 DeepSeek API 生成 CoT 推理

    Args:
        question: 数学题目
        api_key: API 密钥
        model: 模型名称 (deepseek-v4-pro / deepseek-v4-flash)
        generate_wrong: 是否生成错误推理路径
        max_retries: 最大重试次数
        client: 复用的 OpenAI 客户端

    Returns:
        {"cot": "推理过程", "answer": "答案"} 或 None
    """
    if client is None:
        client = _get_client(api_key)

    if generate_wrong:
        system_prompt = (
            "你是一个数学学生，请尝试解答以下小学数学题，但请故意在某个步骤中犯一个合理的计算错误，"
            "导致最终答案错误。请用以下格式回答：\n"
            "推理过程：<你的错误推理过程>\n"
            "答案：<错误的数字答案>"
        )
    else:
        system_prompt = (
            "你是一个数学老师，请一步一步解答以下小学数学题。"
            "请用以下格式回答：\n"
            "推理过程：<详细的解题步骤>\n"
            "答案：<纯数字答案，不带单位>"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3 if not generate_wrong else 0.8,
                max_tokens=1024,
                stream=False,
            )
            content = response.choices[0].message.content
            return _parse_response(content)

        except Exception as e:
            logger.warning(f"API 调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(2 ** attempt)

    return None


def _parse_response(content: str) -> dict:
    """解析 API 返回的文本，提取推理过程和答案"""
    cot = ""
    answer = ""

    # 尝试匹配格式
    cot_match = re.search(r'推理过程[：:]\s*(.*?)(?=答案[：:])', content, re.DOTALL)
    answer_match = re.search(r'答案[：:]\s*(-?\d+\.?\d*)', content)

    if cot_match:
        cot = cot_match.group(1).strip()
    else:
        # 回退：取答案前的所有文本作为推理过程
        if answer_match:
            cot = content[:answer_match.start()].strip()
        else:
            cot = content.strip()

    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        # 回退：取最后一个数字
        nums = re.findall(r'-?\d+\.?\d*', content)
        answer = nums[-1] if nums else ""

    return {"cot": cot, "answer": answer}


def build_cot_dataset(
    input_path: str,
    output_path: str,
    api_key: str,
    model: str = "deepseek-chat",
    max_workers: int = 4,
    generate_wrong: bool = False,
    resume: bool = True,
    limit: int = 0,
) -> None:
    """
    为训练数据批量生成 CoT 推理步骤

    Args:
        input_path: 原始训练数据路径
        output_path: 输出路径
        api_key: DeepSeek API 密钥
        model: 模型名称
        max_workers: 并行线程数
        generate_wrong: 是否同时生成错误推理
        resume: 是否断点续传
        limit: 限制处理条数（0=全部），用于小批量测试
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 小批量测试：只取前 limit 条
    if limit > 0:
        data = data[:limit]
        logger.info(f"小批量测试模式: 只处理前 {limit} 条数据")

    # 断点续传：加载已有结果
    results = {}
    if resume and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        results = {item["id"]: item for item in existing}
        logger.info(f"断点续传：已有 {len(results)} 条结果")

    # 筛选未处理的数据
    todo = [item for item in data if item["id"] not in results]
    logger.info(f"待处理: {len(todo)} 条，已完成: {len(results)} 条")

    # 创建共享客户端（复用连接池）
    client = _get_client(api_key)

    def process_item(item):
        result = _call_deepseek_api(
            question=item["question"],
            api_key=api_key,
            model=model,
            generate_wrong=False,
            client=client,
        )
        if result is None:
            return None

        output_item = {
            "id": item["id"],
            "question": item["question"],
            "instruction": "请一步一步思考，然后给出数字答案。",
            "answer": item["answer"],
            "cot": result["cot"],
            "api_answer": result["answer"],
            "answer_match": str(result["answer"]) == str(item["answer"]),
        }

        # 可选：生成错误推理
        if generate_wrong:
            wrong_result = _call_deepseek_api(
                question=item["question"],
                api_key=api_key,
                model=model,
                generate_wrong=True,
                client=client,
            )
            if wrong_result:
                output_item["wrong_cot"] = wrong_result["cot"]
                output_item["wrong_answer"] = wrong_result["answer"]

        return output_item

    # 并行处理
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in todo}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results[result["id"]] = result
                completed += 1

                # 每 100 条保存一次
                if completed % 100 == 0:
                    _save_results(results, output_path)
                    logger.info(f"进度: {completed}/{len(todo)}")

    # 最终保存
    _save_results(results, output_path)
    logger.info(f"CoT 数据生成完成: 共 {len(results)} 条")

    # 统计答案匹配率
    match_count = sum(1 for r in results.values() if r.get("answer_match"))
    logger.info(f"答案匹配率: {match_count}/{len(results)} "
                f"({match_count/len(results)*100:.1f}%)")


def _save_results(results: dict, output_path: str) -> None:
    """保存结果到文件"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output = sorted(results.values(), key=lambda x: int(x["id"]))
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build CoT dataset")
    parser.add_argument("--input", default="data/raw/train.json")
    parser.add_argument("--output", default="data/processed/train_cot.json")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--generate_wrong", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    build_cot_dataset(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.workers,
        generate_wrong=args.generate_wrong,
        limit=args.limit,
    )
