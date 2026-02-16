import json

import pytest
import torch
from torchvision import transforms

from src.model import build_model, trainable_parameters

def test_forward_pass() -> None:
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)


def test_freeze_backbone_only_fc_trainable() -> None:
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=True)
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_trainable_parameters() -> None:
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=True)
    trainable_params = list(trainable_parameters(model))
    set_1 = set(id(p) for p in trainable_params)
    set_2 = set(id(p) for p in model.fc.parameters())
    assert set_1 == set_2


@pytest.mark.slow
def test_train_run_smoke(trained_run_dir):
    run_dir = trained_run_dir

    model_path = run_dir / "model.pt"
    assert model_path.exists()

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.exists()

    with open(metrics_path) as f:
        metrics = json.load(f)

    assert metrics["epochs"] == 1
    assert metrics["best_epoch"] == 1

    assert len(metrics["history"]["train_loss"]) == 1
    assert len(metrics["history"]["val_loss"]) == 1
