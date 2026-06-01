"""ResNet18 / MobileNetV2 迁移学习模型构建。"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models

from . import config


def build_model(backbone: str, num_classes: int = config.NUM_CLASSES) -> nn.Module:
    backbone = backbone.lower()
    if backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(
            f"不支持的 backbone: {backbone}，可选: {config.BACKBONES}"
        )
    return model
