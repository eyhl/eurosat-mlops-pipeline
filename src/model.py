from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def _build_backbone(name: str, pretrained: bool) -> nn.Module:
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        return resnet18(weights=weights)
    raise ValueError(f"Unknown backbone: {name}")


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    if hasattr(model, "fc"):  # ResNet
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unsupported model architecture")


def _get_head_module(model: nn.Module) -> nn.Module:
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError("Cannot find classifier head")


def build_model(
    *,
    backbone_name: str = "resnet18",
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    model = _build_backbone(backbone_name, pretrained=pretrained)

    _replace_classifier(model, num_classes)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        head = _get_head_module(model)
        for p in head.parameters():
            p.requires_grad = True

    return model


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]
