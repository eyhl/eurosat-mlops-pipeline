
import pytest
import torch
import yaml
from torchvision import transforms

from src import train


def dummy_dataset_root(workspace_root):
    root = workspace_root / "dummy_dataset"
    root.mkdir(parents=True, exist_ok=True)

    for i in range(2):
        class_dir = root / f"class_{i}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for j in range(20):
            img_path = class_dir / f"img_{j}.jpg"
            img = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
            img[i, :, :] = 128
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(img_path)

    return root


def dummy_config_path(workspace_root, dataset_root):
    root = workspace_root / "dummy_config"
    root.mkdir(parents=True, exist_ok=True)
    artifacts_root = root / "artifacts"
    artifacts_root.mkdir(parents=True, exist_ok=True)

    config = {
        "data": {
            "root_dir": str(dataset_root),
            "image_size": 32,
            "batch_size": 2,
            "num_workers": 0,
            "split": {"train": 0.8, "val": 0.1, "test": 0.1},
        },
        "model": {"pretrained": False, "num_classes": 2},
        "train": {
            "epochs": 1,
            "freeze_backbone": True,
        },
        "artifacts": {"dir": str(artifacts_root)},
    }
    config_path = root / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    return {'config_path': config_path, 'artifacts_root': artifacts_root}

@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    workspace_root = tmp_path_factory.mktemp("workspace")

    dataset_root = dummy_dataset_root(workspace_root)
    cfg = dummy_config_path(workspace_root, dataset_root)
    
    return {'workspace_root': workspace_root, 
            'dataset_root': dataset_root, 
            'config_path': cfg['config_path'], 
            'artifacts_root': cfg['artifacts_root']
    }

@pytest.fixture(scope="session")
def trained_run_dir(workspace):
    artifacts_root = workspace["artifacts_root"]
    config_path = workspace["config_path"]

    before = {p.name for p in artifacts_root.iterdir() if p.is_dir()}
    train.run(config_path)
    after = [p for p in artifacts_root.iterdir() if p.is_dir() and p.name not in before]

    assert len(after) == 1
    return after[0]