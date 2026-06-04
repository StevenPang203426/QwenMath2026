"""
Inference-facing compatibility wrapper for shared answer normalization.
"""
from src.utils.answer_normalizer import (
    ANSWER_TYPES,
    basic_clean_answer as _basic_clean,
    detect_answer_type,
    final_clean_answer as _final_clean,
    format_auto,
    format_fraction,
    format_percentage,
    format_rounded,
    get_last_clause as _get_last_clause,
    normalize_answer_for_question,
    parse_numeric,
)


def postprocess_answer(raw_answer: str, question: str) -> str:
    return normalize_answer_for_question(raw_answer, question)
