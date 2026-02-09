import pytest
import torch
from PIL import Image
from src.data import load_dataset


def test_load_dataset_raises_if_root_missing(tmp_path) -> None:
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_dataset(root_dir=missing_dir, image_size=64)


def test_load_dataset_with_minimal_imagefolder(tmp_path) -> None:
    # create class dir
    class_dir = tmp_path / "class_0"
    class_dir.mkdir()

    # 2. create a tiny rgb image
    img_path = class_dir / "img.jpg"
    img = Image.new("RGB", (8, 8), color=(128, 128, 128))
    img.save(img_path)

    # 3. load dataset
    dataset = load_dataset(root_dir=tmp_path, image_size=32)

    # 4. assert
    assert len(dataset) == 1
    assert dataset.class_to_idx == {"class_0": 0}

    # check that transform is applied
    img_tensor, label = dataset[0]
    assert img_tensor.shape == (3, 32, 32)
    assert label == 0
    assert isinstance(img_tensor, torch.Tensor)

    # remove temporary data
    img_path.unlink()
    class_dir.rmdir()
