"""
失败数据修复脚本（幂等）

扫描 train_cot_raw.json，修复三类失败：
  Type A: status=api_failed/parse_error → 重跑正确推理 + 错误推理
  Type B: wrong_status=failed → 直接 hard_fallback（跳过 simple）
  Type C: answer_match=false（正采样算错）→ 回收旧结果为负样本 + 重跑正确推理

按 prompt 类型分阶段执行以最大化 DeepSeek 上下文缓存命中：
  Phase 1: 正确推理（_PROMPT_CORRECT，Type A + Type C）
           Type C 在重跑前先回收旧 cot/api_answer 为负样本
  Phase 2: 错误推理 simple（_PROMPT_WRONG，Phase 1 成功项，最多 3 轮）
  Phase 3: 错误推理 hard（_PROMPT_WRONG_HARD，Type B + Phase 2 剩余）

用法:
  python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY"              # 全量修复
  python -m src.data.data_repair --api_key "$DEEPSEEK_API_KEY" --limit 10   # 小批量测试
"""
import json
import logging
import os
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.data_builder import (
    _get_client, _call_api, _parse_response, _answers_match,
    _sanitize_question, _save,
    _PROMPT_WRONG, _PROMPT_WRONG_HARD, DEFAULT_MODEL,
)

logger = logging.getLogger("math_solver.data_repair")

_LATEX_PATTERN = re.compile(r'\\[a-zA-Z]')


def _has_latex(text: str) -> bool:
    """检测文本是否包含 LaTeX 命令（如 \\frac, \\times, \\pi 等）"""
    return bool(_LATEX_PATTERN.search(text)) if text else False


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

    # ========== 2. 筛选失败条目（按实际数据状态判断） ==========
    type_a = []  # 正确推理缺失：需要重跑
    type_b = []  # 正确推理有了，但错误推理缺失
    type_c = []  # 正采样算错：回收为负样本 + 重跑

    for item in all_data:
        has_cot = bool(item.get("cot", "").strip())
        has_api_answer = bool(item.get("api_answer", "").strip())
        matched = item.get("answer_match", False)
        has_wrong = bool(item.get("wrong_cot", "").strip())

        if not has_cot or not has_api_answer:
            # 正确推理缺失（含 api_failed、parse_error、无 status 等情况）
            type_a.append(item)
        elif matched and not has_wrong:
            # 正确推理有了且匹配，但缺少错误推理
            type_b.append(item)
        elif not matched and not has_wrong:
            # 正采样算错，且尚无负样本可用
            type_c.append(item)

    # 小批量测试模式
    test_mode = limit > 0
    output_path = input_path
    if test_mode:
        type_a = type_a[:limit]
        type_b = type_b[:limit]
        type_c = type_c[:limit]
        test_dir = Path(input_path).parent / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(test_dir / f"repair_test_{limit}.json")
        logger.info(f"小批量测试模式: Type A {len(type_a)} 条, Type B {len(type_b)} 条")
        logger.info(f"测试输出: {output_path}")

    # ========================================================
    # 预处理 0a: LaTeX cot 降级为 wrong_cot
    # ========================================================
    latex_items = []  # cot 含 LaTeX，需要重跑正确推理
    latex_demoted = 0
    for item in all_data:
        cot = item.get("cot", "")
        if _has_latex(cot):
            wrong_cot = item.get("wrong_cot", "")
            # 如果 wrong_cot 为空或也有 LaTeX，则将 cot 降级为 wrong_cot
            if not wrong_cot.strip() or _has_latex(wrong_cot):
                data_by_id[item["id"]]["wrong_cot"] = cot
                data_by_id[item["id"]]["wrong_answer"] = item.get("api_answer", "")
                data_by_id[item["id"]]["wrong_status"] = "latex_demoted"
            # cot 清空，标记为需要重跑
            data_by_id[item["id"]]["cot"] = ""
            data_by_id[item["id"]]["api_answer"] = ""
            data_by_id[item["id"]]["answer_match"] = False
            latex_items.append(item)
            latex_demoted += 1

    if latex_demoted:
        logger.info(f"[预处理 LaTeX] {latex_demoted} 条 cot 含 LaTeX，已降级为 wrong_cot")

    # ========================================================
    # 预处理 0b: answer_match=false 强制修正 / 删除
    # ========================================================
    force_fixed = 0
    deleted_ids = set()
    for item in all_data:
        iid = item["id"]
        d = data_by_id[iid]
        # 只处理有 api_answer 和 wrong_answer 但不匹配的（排除空数据）
        if (not d.get("answer_match")
                and d.get("api_answer", "").strip()
                and d.get("wrong_answer", "").strip()):
            if _answers_match(d["api_answer"], d["wrong_answer"]):
                # 两次算出同一答案 → 强制修正 ground truth
                old_answer = d["answer"]
                d["answer"] = d["api_answer"]
                d["answer_match"] = True
                force_fixed += 1
                logger.debug(f"  id={iid} 强制修正: {old_answer} → {d['answer']}")
            else:
                # api_answer != wrong_answer，数据不可靠 → 删除
                deleted_ids.add(iid)

    if force_fixed:
        logger.info(f"[预处理 强制修正] {force_fixed} 条 answer 已修正为 API 共识答案")
    if deleted_ids:
        # 从数据中彻底移除
        for did in deleted_ids:
            del data_by_id[did]
        all_data = [item for item in all_data if item["id"] not in deleted_ids]
        logger.info(f"[预处理 删除] {len(deleted_ids)} 条数据已删除（api_answer != wrong_answer）")

    # ========================================================
    # 重新筛选（预处理可能改变了数据状态）
    # ========================================================
    type_a = []
    type_b_fresh = []   # 从未尝试过 simple 错误推理 → Phase 2 先试
    type_b_failed = []  # simple 已失败过 → 直接 Phase 3
    type_c = []
    for item in all_data:
        d = data_by_id[item["id"]]
        has_cot = bool(d.get("cot", "").strip())
        has_api_answer = bool(d.get("api_answer", "").strip())
        matched = d.get("answer_match", False)
        has_wrong = bool(d.get("wrong_cot", "").strip())

        if not has_cot or not has_api_answer:
            type_a.append(d)
        elif matched and not has_wrong:
            ws = d.get("wrong_status", "")
            if ws == "failed":
                type_b_failed.append(d)
            else:
                type_b_fresh.append(d)
        elif not matched and not has_wrong:
            type_c.append(d)

    # 小批量测试模式：重新应用 limit（预处理可能改变了分类）
    if test_mode:
        type_a = type_a[:limit]
        type_b_fresh = type_b_fresh[:limit]
        type_b_failed = type_b_failed[:limit]
        type_c = type_c[:limit]

    logger.info(f"失败统计: Type A (正确推理缺失): {len(type_a)}, "
                f"Type B-fresh (需 simple): {len(type_b_fresh)}, "
                f"Type B-failed (直接 hard): {len(type_b_failed)}, "
                f"Type C (正采样算错): {len(type_c)}")

    if not type_a and not type_b_fresh and not type_b_failed and not type_c:
        logger.info("无需修复，所有数据正常")
        _save(data_by_id, output_path)
        return

    client = _get_client(api_key)
    INST = "请一步一步思考，然后给出数字答案。"

    # 保存间隔
    total_todo = len(type_a) + len(type_b_fresh) + len(type_b_failed) + len(type_c)
    save_every = min(50, max(5, total_todo // 5))

    # ========================================================
    # Phase 1: 正确推理（Type A + Type C → _PROMPT_CORRECT）
    #   Type C 在重跑前先回收旧结果为负样本
    # ========================================================
    need_wrong_simple = []  # Phase 1 成功且答案匹配的，进入 Phase 2
    fixed_a = 0
    recycled_c = 0
    suspect_count = 0

    # Type C 回收：将旧的错误正采样保存为负样本
    type_c_ids = set()
    for item in type_c:
        old_cot = item.get("cot", "")
        old_answer = item.get("api_answer", "")
        # 质量门槛：cot 非空、api_answer 非空、cot 长度 >= 10
        if old_cot and old_answer and len(old_cot) >= 10:
            data_by_id[item["id"]]["wrong_cot"] = old_cot
            data_by_id[item["id"]]["wrong_answer"] = old_answer
            data_by_id[item["id"]]["wrong_status"] = "recycled"
            recycled_c += 1
            type_c_ids.add(item["id"])
        else:
            logger.debug(f"  id={item['id']} 旧数据质量不达标，跳过回收")

    if recycled_c:
        logger.info(f"[Type C 回收] {recycled_c}/{len(type_c)} 条旧错误结果已保存为负样本")

    # 合并 Type A + Type C 统一重跑正确推理
    phase1_items = list(type_a) + list(type_c)

    if phase1_items:
        logger.info(f"[Phase 1/3] 正确推理: {len(phase1_items)} 条 "
                    f"(Type A: {len(type_a)}, Type C: {len(type_c)})")

        def phase1_worker(item):
            result = _call_api(item["question"], client, model)
            return item["id"], result

        completed = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(phase1_worker, it): it for it in phase1_items}
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
                        # Type C 已回收负样本的，不再进 Phase 2
                        if item_id in type_c_ids:
                            pass  # wrong_cot/wrong_answer 已在回收时设置
                    elif item_id in type_c_ids:
                        # Type C 重跑仍不匹配：检查回收的负样本是否和新结果一致
                        old_wrong = data_by_id[item_id].get("wrong_answer", "")
                        if _answers_match(result["answer"], old_wrong):
                            # 模型两次算出同一答案 → 大概率标注有误，清除回收的负样本
                            data_by_id[item_id]["wrong_cot"] = ""
                            data_by_id[item_id]["wrong_answer"] = ""
                            data_by_id[item_id]["wrong_status"] = "ground_truth_suspect"
                            suspect_count += 1
                            logger.info(
                                f"  id={item_id} 疑似标注错误: "
                                f"API={result['answer']}, 标注={data_by_id[item_id]['answer']}")
                else:
                    logger.debug(f"  id={item_id} 仍然失败")

                if completed % save_every == 0:
                    _save(data_by_id, output_path)
                    logger.info(f"  进度: {completed}/{len(phase1_items)}")

        _save(data_by_id, output_path)
        logger.info(f"[Phase 1/3] 完成: 修复 {fixed_a}/{len(phase1_items)}, "
                    f"答案匹配 {len(need_wrong_simple)} 条")
    else:
        logger.info("[Phase 1/3] 跳过（无 Type A/C 失败）")

    # ========================================================
    # Phase 2: 错误推理 simple（Phase 1 成功项 → _PROMPT_WRONG）
    #          最多 3 轮，每轮用相同 prompt 最大化缓存命中
    # ========================================================
    need_hard = list(type_b_failed)  # 之前 simple 已失败的，直接进 Phase 3
    simple_success = 0

    # Type C 已回收负样本的不需要再跑错误推理，从 need_wrong_simple 中排除
    need_wrong_simple = [it for it in need_wrong_simple
                         if it["id"] not in type_c_ids]
    # Type B-fresh（包括缺少 wrong_cot/wrong_answer 字段的条目）先走 simple
    need_wrong_simple.extend(type_b_fresh)

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
    total_failures = len(type_a) + len(type_b_fresh) + len(type_b_failed) + len(type_c)
    total_fixed = fixed_a + simple_success + hard_success
    logger.info("=" * 40)
    logger.info(f"修复总结:")
    logger.info(f"  Type A+C 正确推理: {fixed_a}/{len(phase1_items)} 修复")
    logger.info(f"  Type C 负样本回收: {recycled_c}/{len(type_c)} 条")
    logger.info(f"  疑似标注错误:      {suspect_count} 条 (wrong_status=ground_truth_suspect)")
    logger.info(f"  错误推理 simple:   {simple_success} 条成功")
    logger.info(f"  错误推理 hard:     {hard_success}/{len(need_hard)} 条成功")
    logger.info(f"  总计: {total_fixed}/{total_failures} 修复")
    if test_mode:
        logger.info(f"  测试输出: {output_path}")
    else:
        logger.info(f"  已写回: {output_path}")

    # 剩余失败统计
    still_failed_a = sum(1 for it in type_a
                         if data_by_id.get(it["id"], {}).get("status") != "ok")
    still_failed_b = sum(1 for it in (type_b_fresh + type_b_failed + need_wrong_simple)
                         if data_by_id.get(it["id"], {}).get("wrong_status") == "failed")
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
