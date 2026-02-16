from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src import data as data_mod
from src.model import build_model
from src.utils import configure_logging, get_device, get_logger, load_json, load_yaml, save_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained run")
    parser.add_argument("--run", required=True, help="Path to artifacts/<run_id> directory")
    return parser.parse_args()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for inputs, targets in tqdm(loader, desc="test", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        preds = logits.argmax(dim=1)
        running_loss += loss.item()
        correct += int((preds == targets).sum().item())
        total += int(targets.numel())  # could be len(loader.dataset) outside loop

        for t, p in zip(targets.cpu().numpy(), preds.cpu().numpy(), strict=False):
            confusion[int(t), int(p)] += 1

    loss_avg = running_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return loss_avg, acc, confusion


def _plot_confusion_matrix(
    confusion: np.ndarray,
    *,
    class_names: list[str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = confusion.max() / 2.0 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            ax.text(
                j,
                i,
                format(int(confusion[i, j])),
                ha="center",
                va="center",
                color="white" if confusion[i, j] > thresh else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run(*, run_dir: Path) -> None:
    config = load_yaml(run_dir / "config_used.yaml")
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    log = get_logger("evaluate")

    device = get_device()

    log.info("device=%s", device)

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    dataset = data_mod.load_dataset(
        data_cfg.get("root_dir", "data/eurosat_rgb"),
        image_size=int(data_cfg.get("image_size", 224)),
    )
    split_json = load_json(run_dir / "split.json")
    split = data_mod.SplitIndices(
        train=list(split_json["train"]),
        val=list(split_json["val"]),
        test=list(split_json["test"]),
    )

    _, _, test_loader = data_mod.build_dataloaders(
        dataset,
        split,
        batch_size=int(data_cfg.get("batch_size", 32)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    num_classes = int(checkpoint.get("num_classes", model_cfg.get("num_classes", 10)))

    model = build_model(
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, confusion = _evaluate(
        model, test_loader, device=device, criterion=criterion, num_classes=num_classes
    )

    metrics_path = run_dir / "metrics.json"
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    metrics["test"] = {"loss": test_loss, "accuracy": test_acc}
    save_json(metrics_path, metrics)

    log.info("test_loss=%.4f test_acc=%.4f", test_loss, test_acc)

    if bool(config.get("eval", {}).get("confusion_matrix", True)):
        class_to_idx = load_json(run_dir / "class_to_idx.json")
        class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
        out_path = run_dir / "confusion_matrix.png"
        _plot_confusion_matrix(confusion, class_names=class_names, out_path=out_path)
        log.info("saved confusion_matrix=%s", out_path)

def main() -> None: # pragma: no cover
    args = _parse_args()
    run_dir = Path(args.run)
    run(run_dir=run_dir)


if __name__ == "__main__": # pragma: no cover
    main()
