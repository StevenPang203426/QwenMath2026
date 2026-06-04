"""
Utility package exports with lazy optional dependencies.
"""
from __future__ import annotations

__all__ = ["load_config", "set_seed", "compute_accuracy", "setup_wandb"]


def __getattr__(name: str):
    if name == "load_config":
        from .config import load_config

        return load_config
    if name == "set_seed":
        from .seed import set_seed

        return set_seed
    if name == "compute_accuracy":
        from .metrics import compute_accuracy

        return compute_accuracy
    if name == "setup_wandb":
        from .logger import setup_wandb

        return setup_wandb
    raise AttributeError(name)
