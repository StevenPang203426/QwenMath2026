"""
答案提取模块
从模型输出中提取最终数字答案，支持多级回退策略
"""
import re
from typing import Optional


# 数字模式：支持整数、小数、分数(3/5)、百分数(10%)
_NUM = r'-?\d+\.?\d*(?:/\d+\.?\d*)?%?'

# 默认提取规则（按优先级排列）
DEFAULT_PATTERNS = [
    # Level 1: <answer> 标签
    (r'<answer>\s*(.*?)\s*</answer>', "标签匹配"),
    # Level 2: 中文关键词（"答案是/为/：" 后的数字、分数或百分数）
    (rf'答案[是为：:]\s*({_NUM})', "关键词匹配"),
    # Level 3: "= 数字" 格式
    (rf'[=＝]\s*({_NUM})\s*(?:[。.\s]|$)', "等号匹配"),
    # Level 4: 最后一个数字、分数或百分数
    (rf'({_NUM})', "末尾数字"),
]


def extract_answer(
    text: str,
    patterns: list[tuple[str, str]] | None = None,
    return_level: bool = False,
) -> str | tuple[str, str]:
    """
    从模型输出中提取数字答案

    使用多级回退策略：
    1. 匹配 <answer>数字</answer> 标签
    2. 匹配"答案是/为/：" 后的数字
    3. 匹配 "= 数字" 格式
    4. 匹配文本中最后一个数字
    5. 兜底：去除非数字字符后返回

    Args:
        text: 模型输出文本
        patterns: 自定义提取模式列表，每项为 (正则表达式, 描述)
        return_level: 是否同时返回命中的提取级别

    Returns:
        提取的数字字符串；若 return_level=True，返回 (数字, 级别描述)
    """
    if not text:
        return ("", "空输出") if return_level else ""

    text = text.strip()

    if patterns is None:
        patterns = DEFAULT_PATTERNS

    # 逐级尝试提取
    for pattern, level_name in patterns[:-1]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # 清理答案（去除单位等）
            answer = _clean_answer(answer)
            if answer:
                return (answer, level_name) if return_level else answer

    # 最后一级：匹配所有数字，取最后一个
    last_pattern, last_name = patterns[-1]
    all_matches = re.findall(last_pattern, text)
    if all_matches:
        answer = _clean_answer(all_matches[-1])
        if answer:
            return (answer, last_name) if return_level else answer

    # 兜底：去除所有非数字字符（保留小数点、负号、斜杠）
    fallback = re.sub(r'[^\d.\-/]', '', text)
    return (fallback, "兜底清理") if return_level else fallback


def _clean_answer(text: str) -> str:
    """
    清理提取的答案字符串

    去除单位、多余空格、格式化数字
    保留百分数和分数的原始形式
    """
    if not text:
        return ""

    # 去除常见单位
    units = [
        "千克", "公斤", "克", "吨", "米", "厘米", "毫米", "千米", "公里",
        "平方米", "平方厘米", "立方米", "元", "角", "分", "块",
        "个", "只", "条", "本", "台", "辆", "棵", "支", "张", "把",
        "小时", "分钟", "秒", "天", "年", "月", "周",
        "人", "名", "位", "双", "对", "箱", "盒", "包", "瓶", "页",
        "km", "m", "cm", "mm", "kg", "g",
    ]
    for unit in units:
        text = text.replace(unit, "")

    text = text.strip()

    # 百分数保留原样（如 10%、25.5%）
    if text.endswith('%'):
        pct_match = re.match(r'^-?\d+\.?\d*%$', text)
        if pct_match:
            return text

    # 分数格式保留原样（如 3/5、-1/2）
    if '/' in text:
        frac_match = re.match(r'^-?\d+\.?\d*/\d+\.?\d*$', text)
        if frac_match:
            return text

    # 标准化数字格式
    try:
        val = float(text)
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, OverflowError):
        return text


def batch_extract(
    texts: list[str],
    patterns: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    批量提取答案并统计各级别命中率

    Args:
        texts: 模型输出文本列表

    Returns:
        提取结果列表，每项包含 answer 和 level
    """
    results = []
    level_counts: dict[str, int] = {}

    for text in texts:
        answer, level = extract_answer(text, patterns, return_level=True)
        results.append({"answer": answer, "level": level})
        level_counts[level] = level_counts.get(level, 0) + 1

    # 打印统计
    total = len(texts)
    print(f"\n答案提取统计 (共 {total} 条):")
    for level, count in sorted(level_counts.items(), key=lambda x: -x[1]):
        print(f"  {level}: {count} ({count/total*100:.1f}%)")

    return results
