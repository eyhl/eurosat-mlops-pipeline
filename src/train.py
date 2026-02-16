from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src import data as data_mod
from src.model import build_model, trainable_parameters
from src.utils import (
    configure_logging,
    ensure_dir,
    get_device,
    get_git_info,
    get_logger,
    load_yaml,
    make_run_id,
    save_json,
    save_yaml,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 on EuroSAT RGB")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0

    for inputs, targets in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        batches += 1

    return running_loss / max(batches, 1), running_acc / max(batches, 1)


@torch.no_grad()
def _eval_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    batches = 0

    for inputs, targets in tqdm(loader, desc="val", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, targets)
        batches += 1

    return running_loss / max(batches, 1), running_acc / max(batches, 1)


def run(config_path: str | Path) -> None:
    config = load_yaml(config_path)

    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    log = get_logger("train")

    seed = int(config.get("seed", 42))
    set_seed(seed)

    git_info = get_git_info()
    run_id = make_run_id(git_info)

    artifacts_dir = Path(config.get("artifacts", {}).get("dir", "artifacts"))
    run_dir = ensure_dir(artifacts_dir / run_id)

    log.info("run_id=%s", run_id)
    log.info("artifacts_dir=%s", run_dir)

    device = get_device()
    log.info("device=%s", device)

    save_yaml(run_dir / "config_used.yaml", config)
    save_json(run_dir / "git_info.json", git_info.to_dict())

    data_cfg = config.get("data", {})
    split_cfg = data_cfg.get("split", {})

    dataset = data_mod.load_dataset(
        data_cfg.get("root_dir", "data/eurosat_rgb"),
        image_size=int(data_cfg.get("image_size", 224)),
    )

    split = data_mod.make_split_indices(
        len(dataset),
        train_ratio=float(split_cfg.get("train", 0.8)),
        val_ratio=float(split_cfg.get("val", 0.1)),
        test_ratio=float(split_cfg.get("test", 0.1)),
        seed=int(split_cfg.get("seed", seed)),
    )
    save_json(run_dir / "split.json", {"train": split.train, "val": split.val, "test": split.test})
    save_json(run_dir / "class_to_idx.json", dataset.class_to_idx)

    train_loader, val_loader, _ = data_mod.build_dataloaders(
        dataset,
        split,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})

    model = build_model(
        backbone_name=str(model_cfg.get("name", "resnet18")),
        num_classes=int(model_cfg.get("num_classes", 10)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        freeze_backbone=bool(train_cfg.get("freeze_backbone", True)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        trainable_parameters(model),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    epochs = int(train_cfg.get("epochs", 3))
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_epoch = -1
    model_path = run_dir / "model.pt"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _train_one_epoch(
            model, train_loader, device=device, criterion=criterion, optimizer=optimizer
        )
        va_loss, va_acc = _eval_one_epoch(model, val_loader, device=device, criterion=criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        log.info(
            "epoch=%d/%d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            epochs,
            tr_loss,
            tr_acc,
            va_loss,
            va_acc,
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": int(model_cfg.get("num_classes", 10)),
                    "image_size": int(data_cfg.get("image_size", 224)),
                },
                model_path,
            )

    metrics = {
        "run_id": run_id,
        "device": str(device),
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "history": history,
    }
    save_json(run_dir / "metrics.json", metrics)

    log.info("saved model=%s", model_path)
    log.info("saved metrics=%s", run_dir / "metrics.json")


def main() -> None: # pragma: no cover
    args = _parse_args()
    run(config_path=args.config)


if __name__ == "__main__": # pragma: no cover
    main()
