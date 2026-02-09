import math

import torch
from torch import nn

from src.train import _eval_one_epoch, _train_one_epoch, accuracy_from_logits


def test_train_one_epoch_random_data() -> None:
    # create random data
    N = 10
    C = 3
    X = torch.randn(N, 3, 32, 32)
    y = torch.randint(0, C, (N,))
    dataset = torch.utils.data.TensorDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, C))

    # save model parameters before training
    initial_model_parameters = [p.detach().clone() for p in model.parameters()]

    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr_loss, tr_acc = _train_one_epoch(
        model, train_loader, device=device, criterion=criterion, optimizer=optimizer
    )

    assert isinstance(tr_loss, float)
    assert math.isfinite(tr_loss)
    assert isinstance(tr_acc, float)
    assert 0.0 <= tr_acc <= 1.0

    # check if model parameters have been updated
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial_model_parameters, model.parameters(), strict=True)
    )


def test_eval_one_epoch_random_data() -> None:
    # create random data
    N = 10
    C = 3
    X = torch.randn(N, 3, 32, 32)
    y = torch.randint(0, C, (N,))
    dataset = torch.utils.data.TensorDataset(X, y)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, C))

    # save model parameters before evaluation
    initial_model_parameters = [p.detach().clone() for p in model.parameters()]

    device = torch.device("cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    va_loss, va_acc = _eval_one_epoch(model, val_loader, device=device, criterion=criterion)

    assert isinstance(va_loss, float)
    assert math.isfinite(va_loss)
    assert isinstance(va_acc, float)
    assert 0.0 <= va_acc <= 1.0

    no_change_in_parameters = all(
        torch.equal(before, after)
        for before, after in zip(initial_model_parameters, model.parameters(), strict=True)
    )
    assert no_change_in_parameters


# def test_train_one_epoch_zero_loss() -> None:
def test_accuracy_from_logits() -> None:
    # create dummy logits and targets
    targets = torch.tensor([1, 0, 1])
    logits_mixed = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2]])
    logits_correct = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8]])
    logits_wrong = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]])

    acc_mixed = accuracy_from_logits(logits_mixed, targets)
    acc_correct = accuracy_from_logits(logits_correct, targets)
    acc_wrong = accuracy_from_logits(logits_wrong, targets)

    assert all(isinstance(x, float) for x in [acc_mixed, acc_correct, acc_wrong])
    assert math.isclose(acc_mixed, 2 / 3, rel_tol=1e-5)
    assert acc_correct == 1.0
    assert acc_wrong == 0.0
