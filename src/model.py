from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(
    *,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights: Any = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")

    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
