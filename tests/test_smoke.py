import torch

from src.model import build_model, trainable_parameters


def test_forward_pass() -> None:
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 10)


def test_one_train_step_head_only() -> None:
    model = build_model(num_classes=10, pretrained=False, freeze_backbone=True)
    optimizer = torch.optim.SGD(trainable_parameters(model), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    x = torch.randn(2, 3, 224, 224)
    targets = torch.tensor([0, 1])

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item() is True
