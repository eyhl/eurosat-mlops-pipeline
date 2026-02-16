from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model
from src.utils import configure_logging, get_device, get_logger, load_json, load_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a folder of images")
    parser.add_argument("--run", required=True, help="Path to artifacts/<run_id> directory")
    parser.add_argument("--input", required=True, help="Folder with images")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args()


def _build_infer_transform(image_size: int) -> torch.nn.Module:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _iter_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)

@torch.no_grad()
def run(*, run_dir: Path, input_dir: Path, output_path: Path) -> None:
    config = load_yaml(run_dir / "config_used.yaml")
    configure_logging(str(config.get("logging", {}).get("level", "INFO")))
    log = get_logger("predict")

    device = get_device()

    log.info("device=%s", device)

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu")
    num_classes = int(checkpoint.get("num_classes", 10))

    model = build_model(num_classes=num_classes, pretrained=False, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    data_cfg = config.get("data", {})
    image_size = int(data_cfg.get("image_size", checkpoint.get("image_size", 224)))
    transform = _build_infer_transform(image_size)

    class_to_idx = load_json(run_dir / "class_to_idx.json")
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}

    images = _iter_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found under: {input_dir}")

    prob_cols = [f"prob_{idx_to_class[i]}" for i in range(num_classes)]

    rows: list[dict[str, object]] = []
    for path in images:
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy().tolist()

        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        pred_class = idx_to_class.get(pred_idx, str(pred_idx))
        confidence = float(probs[pred_idx])

        row: dict[str, object] = {
            "path": str(path),
            "pred_index": pred_idx,
            "pred_class": pred_class,
            "confidence": confidence,
        }
        row.update({col: float(probs[i]) for i, col in enumerate(prob_cols)})
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    log.info("wrote predictions=%s (n=%d)", output_path, len(df))

def main() -> None: # pragma: no cover
    args = _parse_args()
    run_dir = Path(args.run)
    input_dir = Path(args.input)
    output_path = Path(args.output)

    run(run_dir=run_dir, input_dir=input_dir, output_path=output_path)



if __name__ == "__main__": # pragma: no cover
    main()
