"""
配置加载模块
支持 YAML 继承（inherit 字段）和命令行覆盖
"""
import os
import copy
import yaml
import argparse
from typing import Any
from pathlib import Path


class Config:
    """配置对象，支持点号访问（config.model.name）"""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """转换回字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """安全获取属性"""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并两个字典，override 中的值覆盖 base"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _resolve_inherit(config_data: dict, config_dir: str) -> dict:
    """解析配置继承链"""
    if "inherit" not in config_data:
        return config_data

    parent_name = config_data.pop("inherit")
    parent_path = os.path.join(config_dir, f"{parent_name}.yaml")

    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"父配置文件不存在: {parent_path}")

    with open(parent_path, "r", encoding="utf-8") as f:
        parent_data = yaml.safe_load(f)

    # 递归解析父配置的继承
    parent_data = _resolve_inherit(parent_data, config_dir)

    return _deep_merge(parent_data, config_data)


def load_config(config_path: str, overrides: dict | None = None) -> Config:
    """
    加载 YAML 配置文件

    Args:
        config_path: YAML 配置文件路径
        overrides: 命令行覆盖参数（可选）

    Returns:
        Config 对象
    """
    config_path = Path(config_path)
    config_dir = str(config_path.parent)

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # 解析继承
    config_data = _resolve_inherit(config_data, config_dir)

    # 应用命令行覆盖
    if overrides:
        config_data = _deep_merge(config_data, overrides)

    return Config(config_data)


def parse_args_with_config() -> Config:
    """
    从命令行参数解析配置

    用法: python script.py --config configs/sft_cot.yaml --training.learning_rate 1e-5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    args, unknown = parser.parse_known_args()

    # 解析额外的命令行覆盖参数
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                value = unknown[i + 1]
                # 尝试类型转换
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                i += 2
            else:
                value = True
                i += 1

            # 支持点号分隔的嵌套键（如 training.learning_rate）
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
        else:
            i += 1

    return load_config(args.config, overrides)
