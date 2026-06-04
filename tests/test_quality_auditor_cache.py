"""
Cache-friendly request construction tests for quality_auditor.
"""
import sys
from types import SimpleNamespace

torch_stub = SimpleNamespace()
torch_utils_data_stub = SimpleNamespace(Dataset=object)
torch_utils_stub = SimpleNamespace(data=torch_utils_data_stub)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.utils", torch_utils_stub)
sys.modules.setdefault("torch.utils.data", torch_utils_data_stub)

sys.path.insert(0, ".")

from src.data.quality_auditor import (  # noqa: E402
    _AUDIT_CACHE_PREFIX,
    _AUDIT_CANDIDATE_SEPARATOR,
    _build_audit_messages,
    _record_cache_usage,
)


def _user_text(messages):
    content = messages[1]["content"]
    assert isinstance(content, list)
    return content[0]["text"]


def _common_prefix_len(left: str, right: str) -> int:
    for idx, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return idx
    return min(len(left), len(right))


def test_audit_messages_keep_dynamic_candidate_after_long_static_prefix():
    first = {
        "id": "1",
        "question": "甲有12个苹果，乙有8个苹果，一共有多少个？",
        "answer": "20",
        "signals": ["expr_correct_failed"],
        "risk_score": 0.35,
    }
    second = {
        "id": "2",
        "question": "一件商品原价80元，打八折后多少元？",
        "answer": "64",
        "signals": ["cot_answer_disagreement"],
        "risk_score": 0.45,
    }

    first_text = _user_text(_build_audit_messages(first))
    second_text = _user_text(_build_audit_messages(second))
    common_len = _common_prefix_len(first_text, second_text)

    assert first_text.startswith(_AUDIT_CACHE_PREFIX)
    assert _AUDIT_CANDIDATE_SEPARATOR in first_text
    assert common_len >= len(_AUDIT_CACHE_PREFIX) + len(_AUDIT_CANDIDATE_SEPARATOR)
    assert common_len > 1000
    assert first["question"] in first_text[common_len:]
    assert second["question"] in second_text[common_len:]


def test_record_cache_usage_accumulates_deepseek_usage_fields():
    stats = {"requests": 0, "hit_tokens": 0, "miss_tokens": 0}
    first = SimpleNamespace(
        usage=SimpleNamespace(prompt_cache_hit_tokens=128, prompt_cache_miss_tokens=512)
    )
    second = SimpleNamespace(
        usage=SimpleNamespace(prompt_cache_hit_tokens=256, prompt_cache_miss_tokens=64)
    )
    third = {"usage": {"prompt_cache_hit_tokens": 32, "prompt_cache_miss_tokens": 16}}

    _record_cache_usage(first, stats)
    _record_cache_usage(second, stats)
    _record_cache_usage(SimpleNamespace(**third), stats)

    assert stats == {"requests": 3, "hit_tokens": 416, "miss_tokens": 592}


if __name__ == "__main__":
    test_audit_messages_keep_dynamic_candidate_after_long_static_prefix()
    test_record_cache_usage_accumulates_deepseek_usage_fields()
    print("quality_auditor cache tests: ALL PASSED")
