import json
import pytest
import torch
from torchvision import transforms
import yaml
from src.model import build_model, trainable_parameters
from src.train import run


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
def test_train_run_smoke(tmp_path):
    image_size = 32
    num_classes = 2
    num_samples = 10

    data_root = tmp_path / "data"
    artifacts_root = tmp_path / "artifacts"

    # create dummy data directory
    for i in range(num_classes):
        class_dir = data_root / "class_{}".format(i)
        class_dir.mkdir(parents=True)
        for j in range(num_samples):
            img_path = class_dir / "img_{}.jpg".format(j)
            img = torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8)
            img[i, :, :] = 128
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)

    # create config file
    config = {
        "data": {
            "root_dir": str(data_root),
            "image_size": image_size,
            "batch_size": 2,
            "num_workers": 0,
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "model": {"pretrained": False, "num_classes": num_classes},
        "train": {
            "epochs": 1,
            "freeze_backbone": True,
        },
        "artifacts": {"dir": str(artifacts_root)},
    }

    config_path = tmp_path / "config.yaml"

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    # run training
    run(config_path)

    entries = list(artifacts_root.iterdir())
    run_dirs = [p for p in entries if p.is_dir()]

    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

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

    # config_path = "configs/test.yaml"

    # # read config and read image size and batch size, number of classes
    # import yaml
    # with open(config_path) as f:
    #     config = yaml.safe_load(f)
    # image_size = config["data"]["image_size"]
    # num_classes = config["model"]["num_classes"]
