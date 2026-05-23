"""
评测指标模块
"""
import re
import json
from typing import Optional


def _values_equal(a: Optional[str], b: Optional[str], tol: float = 1e-6) -> bool:
    """数值容差比较，处理浮点精度问题（如 0.6 vs 3/5）"""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < tol
    except (ValueError, TypeError):
        return False


def normalize_number(text: str) -> Optional[str]:
    """
    标准化数字字符串，处理各种格式

    Args:
        text: 待标准化的文本

    Returns:
        标准化后的数字字符串，无法解析时返回 None
    """
    if not text:
        return None

    text = text.strip()

    # 处理分数（如 3/4）
    frac_match = re.match(r'^(-?\d+)/(\d+)$', text)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            result = num / den
            # 如果是整数结果，返回整数形式
            if result == int(result):
                return str(int(result))
            return str(round(result, 6))

    # 处理百分数（如 25%）
    pct_match = re.match(r'^(-?\d+\.?\d*)%$', text)
    if pct_match:
        return str(float(pct_match.group(1)))

    # 处理普通数字
    num_match = re.match(r'^(-?\d+\.?\d*)$', text)
    if num_match:
        val = float(num_match.group(1))
        if val == int(val):
            return str(int(val))
        return str(val)

    return None


def compute_accuracy(
    predictions: list[str],
    references: list[str],
    return_details: bool = False
) -> dict:
    """
    计算正确率

    Args:
        predictions: 模型预测列表
        references: 标准答案列表
        return_details: 是否返回逐条对比详情

    Returns:
        包含正确率和统计信息的字典
    """
    assert len(predictions) == len(references), \
        f"预测数量 ({len(predictions)}) 与标准答案数量 ({len(references)}) 不匹配"

    correct = 0
    total = len(predictions)
    details = []

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_norm = normalize_number(str(pred))
        ref_norm = normalize_number(str(ref))

        is_correct = _values_equal(pred_norm, ref_norm)

        if is_correct:
            correct += 1

        if return_details:
            details.append({
                "index": i,
                "prediction": pred,
                "reference": ref,
                "pred_normalized": pred_norm,
                "ref_normalized": ref_norm,
                "correct": is_correct,
            })

    result = {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
    }

    if return_details:
        result["details"] = details

    return result


def evaluate_from_files(
    prediction_csv: str,
    reference_json: str
) -> dict:
    """
    从文件计算正确率

    Args:
        prediction_csv: 预测结果 CSV 文件路径（id,ret 格式）
        reference_json: 标准答案 JSON 文件路径

    Returns:
        评测结果字典
    """
    # 加载预测结果
    preds = {}
    with open(prediction_csv, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("id"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                preds[str(parts[0])] = parts[1]

    # 加载标准答案
    with open(reference_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    predictions = []
    references = []
    for item in ref_data:
        item_id = str(item["id"])
        if item_id in preds:
            predictions.append(preds[item_id])
            references.append(item["answer"])

    return compute_accuracy(predictions, references, return_details=True)
