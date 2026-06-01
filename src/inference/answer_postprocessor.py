"""
答案后处理规则引擎

根据题目上下文（问题文本）对模型输出的答案进行格式化：
- 百分数/分数/整数类型检测
- ceil/floor 取整判断
- 保留小数位数
- 格式标准化（去单位、尾零处理等）

优先级：格式（百分数/分数/保留位数）> 取整（ceil/floor）> 整数 > 自动
"""
import math
import re
from fractions import Fraction
from typing import Optional, Tuple


# ============================================================
# 答案类型枚举
# ============================================================

ANSWER_TYPES = [
    "percentage",      # 百分数
    "fraction",        # 分数
    "round_n",         # 保留 n 位小数
    "ceil_integer",    # 向上取整
    "floor_integer",   # 向下取整
    "integer",         # 整数（量词推断）
    "auto",            # 自动判断
]


# ============================================================
# 题目类型检测规则
# ============================================================

# 百分数关键词
_PCT_KEYWORDS = re.compile(
    r'百分之几|百分率|百分比|百分数'
)
# "率" 类关键词（排除 "率的" "效率高" 等非答案场景）
_RATE_KEYWORDS = re.compile(
    r'(?:合格|成活|发芽|出勤|命中|正确|错误|出错|达标|及格|'
    r'利润|增长|下降|节约|浪费|损失|淘汰|录取|中奖|'
    r'浓度|纯度|盐度|含盐|含糖|酒精)率'
)
# 分数关键词
_FRAC_KEYWORDS = re.compile(
    r'几分之几|分率|占[了]?[总全]|是[总全].*的几分之|'
    r'比例|比率|比重|占比'
)
# 向上取整
_CEIL_KEYWORDS = re.compile(r'至少|最少|起码')
_CEIL_CONTEXT = re.compile(
    r'租|船|拉完|运完|需要|需|要|限乘|乘坐|准乘|限载|限坐'
)
# 向下取整
_FLOOR_KEYWORDS = re.compile(r'至多|最多|顶多')
_FLOOR_CONTEXT = re.compile(r'能|可以|可|装|分|做|买|坐|乘')
# 量词（暗示整数）
_INTEGER_UNIT = re.compile(
    r'多少[辆人个只条棵支张把本台次箱盒包瓶页桶圈束件副头幅排枝根袋套双对名位]|'
    r'几[辆人个只条棵支张把本台次箱盒包瓶页桶圈束件副头幅排枝根袋套双对名位]'
)
# 保留小数位数
_ROUND_PATTERNS = [
    (re.compile(r'保留[到]?整数|精确到[个]?位|精确到整'), 0),
    (re.compile(r'保留[到]?[一1][位]小数|精确到0\.1|精确到十分位|保留[到]?小数点后[一1]位'), 1),
    (re.compile(r'保留[到]?[两二2][位]小数|精确到0\.01|精确到百分位|保留[到]?小数点后[两二2]位'), 2),
    (re.compile(r'保留[到]?[三3][位]小数|精确到0\.001|精确到千分位|保留[到]?小数点后[三3]位'), 3),
]


def detect_answer_type(question: str) -> Tuple[str, Optional[int]]:
    """
    从题目文本推断答案类型

    Returns:
        (类型名称, 附加参数)
        附加参数：round_n 时为小数位数 n，其他为 None
    """
    if not question:
        return ("auto", None)

    # 取最后一个分句用于判断（更精确）
    last_clause = _get_last_clause(question)

    # 优先级 1：保留位数（最精确的格式要求）
    for pattern, n in _ROUND_PATTERNS:
        if pattern.search(question):
            return ("round_n", n)

    # 优先级 2：百分数
    if _PCT_KEYWORDS.search(last_clause) or _RATE_KEYWORDS.search(last_clause):
        # 排除 "率的" "利率高" 等非答案场景
        if not re.search(r'率的|率高|率低', last_clause):
            return ("percentage", None)

    # 优先级 3：分数
    if _FRAC_KEYWORDS.search(last_clause):
        return ("fraction", None)

    # 优先级 4：ceil/floor
    if _CEIL_KEYWORDS.search(last_clause):
        if _CEIL_CONTEXT.search(last_clause) or _INTEGER_UNIT.search(last_clause):
            return ("ceil_integer", None)
    if _FLOOR_KEYWORDS.search(last_clause):
        if _FLOOR_CONTEXT.search(last_clause) or _INTEGER_UNIT.search(last_clause):
            return ("floor_integer", None)

    # 优先级 5：整数（量词推断）
    if _INTEGER_UNIT.search(last_clause):
        return ("integer", None)

    return ("auto", None)


def _get_last_clause(text: str) -> str:
    """获取最后一个分句（按标点断句）"""
    # 按常见分隔标点断句
    parts = re.split(r'[，。？！；,;?!]', text)
    # 取最后非空分句
    for part in reversed(parts):
        if part.strip():
            return part.strip()
    return text


# ============================================================
# 数值解析
# ============================================================

def parse_numeric(answer: str) -> Optional[float]:
    """
    将答案字符串解析为浮点数

    支持：整数、小数、分数 (a/b)、百分数 (x%)
    """
    if not answer:
        return None

    answer = answer.strip()

    # 已经是百分数
    pct_match = re.match(r'^(-?\d+\.?\d*)%$', answer)
    if pct_match:
        # 返回百分数的数值部分（不除以100）
        return float(pct_match.group(1))

    # 分数
    frac_match = re.match(r'^(-?\d+)/(\d+)$', answer)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        if den != 0:
            return num / den
        return None

    # 普通数字
    try:
        return float(answer)
    except (ValueError, TypeError):
        return None


def _is_already_formatted(answer: str, answer_type: str) -> bool:
    """检查答案是否已经是目标格式"""
    if answer_type == "percentage" and answer.endswith('%'):
        return True
    if answer_type == "fraction" and '/' in answer and '%' not in answer:
        return True
    return False


# ============================================================
# 格式化函数
# ============================================================

def format_percentage(value: float, question: str) -> str:
    """
    格式化为百分数

    策略：如果值 > 1，认为已经是百分比数值，直接加 %
          如果值 <= 1 且 > 0，乘以 100 再加 %
    """
    # 如果输入已经是百分数格式的数值部分（如 25 → 25%）
    # 根据用户决策：25 视为 25%
    if value > 1:
        pct_value = value
    elif 0 < value <= 1:
        # 可能是小数形式的比率，如 0.25 → 25%
        pct_value = value * 100
    elif value == 0:
        pct_value = 0
    else:
        # 负数，保持原样
        pct_value = value

    # 确定保留位数
    decimal_places = _detect_pct_precision(question)

    if decimal_places is not None:
        formatted = f"{pct_value:.{decimal_places}f}"
    else:
        # 默认：能整就整，不能整保留一位
        if pct_value == int(pct_value):
            formatted = str(int(pct_value))
        else:
            formatted = f"{pct_value:.1f}"
            # 去掉末尾的 .0
            if formatted.endswith('.0'):
                formatted = formatted[:-2]

    return f"{formatted}%"


def _detect_pct_precision(question: str) -> Optional[int]:
    """检测百分数精度要求"""
    if re.search(r'保留整数|保留到1%|精确到1%|保留到百分之一|精确到百分之一', question):
        return 0
    if re.search(r'保留[一1]位|精确到0\.1%', question):
        return 1
    if re.search(r'保留[两二2]位|精确到0\.01%', question):
        return 2
    return None


def format_fraction(value: float) -> str:
    """
    格式化为分数

    使用 Fraction.limit_denominator(100) 转换
    """
    if value == int(value):
        return str(int(value))

    frac = Fraction(value).limit_denominator(100)
    return f"{frac.numerator}/{frac.denominator}"


def format_rounded(value: float, n: int) -> str:
    """格式化为保留 n 位小数（四舍五入）"""
    if n == 0:
        return str(int(value + 0.5) if value >= 0 else int(value - 0.5))
    # 使用 Decimal 做精确���舍五入，避免浮点 banker's rounding
    from decimal import Decimal, ROUND_HALF_UP
    d = Decimal(str(value))
    quantize_str = '0.' + '0' * n
    rounded = d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)
    return str(rounded)


def format_auto(value: float) -> str:
    """自动格式化：整数去尾零，小数保留合理位数"""
    if value == int(value):
        return str(int(value))
    # 保留有效位数，去除尾零
    formatted = f"{value:.6f}".rstrip('0').rstrip('.')
    return formatted


# ============================================================
# 主入口
# ============================================================

def postprocess_answer(raw_answer: str, question: str) -> str:
    """
    根据题目上下文对模型输出答案进行后处理

    Args:
        raw_answer: 模型输出的原始答案字符串
        question: 原始题目文本

    Returns:
        格式化后的答案字符串
    """
    if not raw_answer or not raw_answer.strip():
        return "0"

    raw_answer = raw_answer.strip()

    # 基础清理：去除换行、单位、LaTeX
    answer = _basic_clean(raw_answer)

    # 检测答案类型
    answer_type, param = detect_answer_type(question)

    # 如果已经是目标格式，直接通过
    if _is_already_formatted(answer, answer_type):
        return _final_clean(answer)

    # 解析数值
    numeric_value = parse_numeric(answer)

    # 无法解析为数字，返回清理后的原始答案
    if numeric_value is None:
        return _final_clean(answer)

    # 按类型格式化
    if answer_type == "percentage":
        return format_percentage(numeric_value, question)
    elif answer_type == "fraction":
        return format_fraction(numeric_value)
    elif answer_type == "round_n":
        return format_rounded(numeric_value, param)
    elif answer_type == "ceil_integer":
        # 仅小数时才 ceil，已是整数直接通过
        if numeric_value != int(numeric_value):
            return str(math.ceil(numeric_value))
        else:
            return str(int(numeric_value))
    elif answer_type == "floor_integer":
        if numeric_value != int(numeric_value):
            return str(math.floor(numeric_value))
        else:
            return str(int(numeric_value))
    elif answer_type == "integer":
        if numeric_value == int(numeric_value):
            return str(int(numeric_value))
        else:
            return str(round(numeric_value))
    else:
        return format_auto(numeric_value)


# ============================================================
# 基础清理
# ============================================================

_UNITS = [
    "千克", "公斤", "克", "吨", "米", "厘米", "毫米", "千米", "公里",
    "平方米", "平方厘米", "立方米", "元", "角", "分", "块",
    "个", "只", "条", "本", "台", "辆", "棵", "支", "张", "把",
    "小时", "分钟", "秒", "天", "年", "月", "周",
    "人", "名", "位", "双", "对", "箱", "盒", "包", "瓶", "页",
    "km", "m", "cm", "mm", "kg", "g",
]

_LATEX_CMD = re.compile(r'\\[a-zA-Z]+')
_LATEX_BRACES = re.compile(r'[{}$]')


def _basic_clean(answer: str) -> str:
    """基础清理：去除单位、LaTeX、换行等"""
    answer = answer.replace("\n", " ").strip()

    # 中文日期 → 分数格式
    answer = re.sub(r'(\d+)\s*月\s*(\d+)\s*日?', r'\1/\2', answer)

    # 去除单位（长单位优先匹配）
    for unit in sorted(_UNITS, key=len, reverse=True):
        answer = answer.replace(unit, "")

    # 去除 LaTeX
    answer = _LATEX_CMD.sub('', answer)
    answer = _LATEX_BRACES.sub('', answer)

    return answer.strip()


def _final_clean(answer: str) -> str:
    """最终清理：去除非法字符"""
    # 保留数字、小数点、负号、斜杠、百分号
    # 但如果整体是合法格式就不要过度清理
    if re.match(r'^-?\d+\.?\d*%?$', answer):
        return answer
    if re.match(r'^-?\d+/\d+$', answer):
        return answer
    # 尝试提取数字部分
    nums = re.findall(r'-?\d+\.?\d*(?:/\d+)?%?', answer)
    if nums:
        return nums[-1]
    return answer
