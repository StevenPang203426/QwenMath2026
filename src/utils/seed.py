"""
随机种子固定模块
确保实验可复现
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    固定所有随机种子以保证可复现性

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 确保 CUDA 卷积算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
