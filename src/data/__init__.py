"""
Data package exports.

Keep training dependencies lazy so lightweight data processing commands can
run without torch installed.
"""
from __future__ import annotations

from .answer_extractor import extract_answer
from .preprocessor import prepare_dpo_data, prepare_sft_data, split_train_val

__all__ = [
    "MathDataset",
    "MathDPODataset",
    "extract_answer",
    "prepare_sft_data",
    "prepare_dpo_data",
    "split_train_val",
]


def __getattr__(name: str):
    if name in {"MathDataset", "MathDPODataset"}:
        from .dataset import MathDPODataset, MathDataset

        return {"MathDataset": MathDataset, "MathDPODataset": MathDPODataset}[name]
    raise AttributeError(name)
