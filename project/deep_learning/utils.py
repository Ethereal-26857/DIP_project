"""通用工具：随机种子、原图/增强图判定、按 lesion 分组。"""
import re
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def base_lesion_id(image_id: str) -> str:
    """从 image_id 提取原图编号，如 '42_aug2' -> '42'。"""
    m = re.match(r"^(\d+)", str(image_id))
    return m.group(1) if m else str(image_id)


def is_augmented(image_id: str) -> bool:
    return "_aug" in str(image_id)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
