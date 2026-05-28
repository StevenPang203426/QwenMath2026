"""
失败数据修复脚本（幂等）

扫描 train_cot_raw.json，修复两类失败：
  Type A: status=api_failed/parse_error → 重跑正确推理 + 错误推理
  Type B: wrong_status=failed → 直接 hard_fallback（跳过 simple）

按 prompt 类型分阶段执行以最大化 DeepSeek 上下文缓存命中：
  Phase 1: 正确推理（_PROMPT_CORRECT，仅 Type A）
  Phase 2: 错误推理 simple（_PROMPT_WRONG，Phase 1 成功项，最多 3 轮）
  Phase 3: 错误推理 hard（_PROMPT_WRONG_HARD，Type B + Phase 2 剩余）

用法:
  python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY"              # 全量修复
  python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY" --limit 10   # 小批量测试
"""
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.data_builder import (
    _get_client, _call_api, _parse_response, _answers_match,
    _sanitize_question, _save,
    _PROMPT_WRONG, _PROMPT_WRONG_HARD, DEFAULT_MODEL,
)

logger = logging.getLogger("math_solver.data_repair")


def repair_dataset(
    input_path: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    limit: int = 0,
) -> None:
    """
    修复失败数据

    Args:
        input_path: 数据文件路径（同时作为输出路径，原地更新）
        api_key: DeepSeek API 密钥
        model: 模型名称
        max_workers: 并行线程数
        limit: 限制处理条数（0=全部），用于小批量测试
    """
    # ========== 1. 读取数据 ==========
    with open(input_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)

    data_by_id = {item["id"]: item for item in all_data}

    # ========== 2. 筛选失败条目 ==========
    type_a = [item for item in all_data
              if item.get("status") == "api_failed"
              or str(item.get("status", "")).startswith("parse_error")]

    type_b = [item for item in all_data
              if item.get("status") == "ok"
              and item.get("answer_match")
              and item.get("wrong_status") == "failed"]

    # 小批量测试模式
    test_mode = limit > 0
    output_path = input_path
    if test_mode:
        type_a = type_a[:limit]
        type_b = type_b[:limit]
        test_dir = Path(input_path).parent / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(test_dir / f"repair_test_{limit}.json")
        logger.info(f"小批量测试模式: Type A {len(type_a)} 条, Type B {len(type_b)} 条")
        logger.info(f"测试输出: {output_path}")

    logger.info(f"失败统计: Type A (api/parse 失败): {len(type_a)}, "
                f"Type B (错误推理失败): {len(type_b)}")

    if not type_a and not type_b:
        logger.info("无需修复，所有数据正常")
        return

    client = _get_client(api_key)
    INST = "请一步一步思考，然后给出数字答案。"

    # 保存间隔
    save_every = min(50, max(5, (len(type_a) + len(type_b)) // 5))

    # ========================================================
    # Phase 1: 正确推理（Type A → _PROMPT_CORRECT）
    # ========================================================
    need_wrong_simple = []  # Phase 1 成功且答案匹配的，进入 Phase 2
    fixed_a = 0

    if type_a:
        logger.info(f"[Phase 1/3] 正确推理: {len(type_a)} 条")

        def phase1_worker(item):
            result = _call_api(item["question"], client, model)
            return item["id"], result

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(phase1_worker, it): it for it in type_a}
            for fut in as_completed(futures):
                item_id, result = fut.result()
                completed += 1

                if result and result.get("answer"):
                    matched = _answers_match(
                        result["answer"], data_by_id[item_id]["answer"])
                    data_by_id[item_id].update({
                        "cot": result["cot"],
                        "api_answer": result["answer"],
                        "answer_match": matched,
                        "status": "ok",
                    })
                    fixed_a += 1
                    if matched:
                        need_wrong_simple.append(data_by_id[item_id])
                else:
                    logger.debug(f"  id={item_id} 仍然失败")

                if completed % save_every == 0:
                    _save(data_by_id, output_path)
                    logger.info(f"  进度: {completed}/{len(type_a)}")

        _save(data_by_id, output_path)
        logger.info(f"[Phase 1/3] 完成: 修复 {fixed_a}/{len(type_a)}, "
                    f"答案匹配 {len(need_wrong_simple)} 条")
    else:
        logger.info("[Phase 1/3] 跳过（无 Type A 失败）")

    # ========================================================
    # Phase 2: 错误推理 simple（Phase 1 成功项 → _PROMPT_WRONG）
    #          最多 3 轮，每轮用相同 prompt 最大化缓存命中
    # ========================================================
    need_hard = list(type_b)  # Type B 全部直接进 Phase 3
    simple_success = 0

    if need_wrong_simple:
        logger.info(f"[Phase 2/3] 错误推理(simple): {len(need_wrong_simple)} 条, 最多 3 轮")
        remaining = list(need_wrong_simple)

        for round_num in range(1, 4):
            if not remaining:
                break
            logger.info(f"  Round {round_num}/3: {len(remaining)} 条")

            next_remaining = []

            def make_phase2_worker(rn):
                """闭包捕获 round_num"""
                def phase2_worker(item):
                    wrong = _call_api(
                        item["question"], client, model, generate_wrong=True)
                    if (wrong and wrong.get("answer")
                            and not _answers_match(
                                wrong["answer"], str(item["answer"]))):
                        return item["id"], wrong, f"simple_try{rn}"
                    return item["id"], None, None
                return phase2_worker

            worker = make_phase2_worker(round_num)

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(worker, it): it for it in remaining}
                for fut in as_completed(futures):
                    item_id, wrong, status = fut.result()
                    if wrong:
                        data_by_id[item_id]["wrong_cot"] = wrong["cot"]
                        data_by_id[item_id]["wrong_answer"] = wrong["answer"]
                        data_by_id[item_id]["wrong_status"] = status
                        simple_success += 1
                    else:
                        next_remaining.append(data_by_id[item_id])

            remaining = next_remaining
            _save(data_by_id, output_path)
            logger.info(f"  Round {round_num} 完成, 剩余: {len(remaining)} 条")

        # 3 轮都没出错的 → Phase 3
        need_hard.extend(remaining)
        logger.info(f"[Phase 2/3] 完成: simple 成功 {simple_success}/{len(need_wrong_simple)}, "
                    f"转 hard: {len(remaining)} 条")
    else:
        logger.info("[Phase 2/3] 跳过（无需 simple 错误推理）")

    # ========================================================
    # Phase 3: 错误推理 hard（Type B + Phase 2 失败 → _PROMPT_WRONG_HARD）
    # ========================================================
    hard_success = 0

    if need_hard:
        logger.info(f"[Phase 3/3] 错误推理(hard): {len(need_hard)} 条")

        def phase3_worker(item):
            hard_prompt = _PROMPT_WRONG_HARD.format(
                correct_answer=item["answer"])
            wrong = _call_api(
                item["question"], client, model,
                generate_wrong=True,
                system_prompt=hard_prompt,
                temperature=0.7,
            )
            if (wrong and wrong.get("answer")
                    and not _answers_match(
                        wrong["answer"], str(item["answer"]))):
                return item["id"], wrong
            return item["id"], None

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(phase3_worker, it): it for it in need_hard}
            for fut in as_completed(futures):
                item_id, wrong = fut.result()
                completed += 1

                if wrong:
                    data_by_id[item_id]["wrong_cot"] = wrong["cot"]
                    data_by_id[item_id]["wrong_answer"] = wrong["answer"]
                    data_by_id[item_id]["wrong_status"] = "hard_fallback"
                    hard_success += 1
                else:
                    data_by_id[item_id]["wrong_status"] = "failed"

                if completed % save_every == 0:
                    _save(data_by_id, output_path)
                    logger.info(f"  进度: {completed}/{len(need_hard)}")

        _save(data_by_id, output_path)
        logger.info(f"[Phase 3/3] 完成: hard 成功 {hard_success}/{len(need_hard)}")
    else:
        logger.info("[Phase 3/3] 跳过（无需 hard 错误推理）")

    # ========== 最终统计 ==========
    total_failures = len(type_a) + len(type_b)
    total_fixed = fixed_a + simple_success + hard_success
    logger.info("=" * 40)
    logger.info(f"修复总结:")
    logger.info(f"  Type A (api 失败): {fixed_a}/{len(type_a)} 修复")
    logger.info(f"  错误推理 simple:   {simple_success} 条成功")
    logger.info(f"  错误推理 hard:     {hard_success}/{len(need_hard)} 条成功")
    logger.info(f"  总计: {total_fixed}/{total_failures} 修复")
    if test_mode:
        logger.info(f"  测试输出: {output_path}")
    else:
        logger.info(f"  已写回: {output_path}")

    # 剩余失败统计
    still_failed_a = sum(1 for it in type_a
                         if data_by_id[it["id"]].get("status") != "ok")
    still_failed_b = sum(1 for it in (type_b + need_wrong_simple)
                         if data_by_id[it["id"]].get("wrong_status") == "failed")
    if still_failed_a or still_failed_b:
        logger.info(f"  仍失败: api={still_failed_a}, wrong={still_failed_b}")
        logger.info(f"  可再次运行本脚本继续修复")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Repair failed data entries")
    parser.add_argument("--input",
                        default="data/processed/train_cot_raw.json",
                        help="数据文件路径")
    parser.add_argument("--api_key", required=True, help="DeepSeek API 密钥")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="模型名称")
    parser.add_argument("--workers", type=int, default=4, help="并行线程数")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制处理条数（0=全部），小批量测试用")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    repair_dataset(
        input_path=args.input,
        api_key=args.api_key,
        model=args.model,
        max_workers=args.workers,
        limit=args.limit,
    )
