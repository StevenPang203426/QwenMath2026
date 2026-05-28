"""
Shared helpers for TRL-based RL trainers.
"""
import inspect
import json
from typing import Any


def to_text(value: Any) -> str:
    """Normalize JSON text fields before Arrow infers column types."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(to_text(part) for part in value)
    if isinstance(value, dict):
        if "text" in value:
            return to_text(value["text"])
        if "content" in value:
            return to_text(value["content"])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Render chat messages to a plain prompt string for TRL datasets."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def filter_supported_kwargs(cls_or_fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs supported by the installed TRL version."""
    signature = inspect.signature(cls_or_fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def add_tokenizer_kwarg(trainer_cls: Any, kwargs: dict[str, Any], tokenizer: Any) -> None:
    """Add tokenizer under the name expected by this TRL trainer version."""
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
