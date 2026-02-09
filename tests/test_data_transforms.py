from torchvision import transforms
from src.data import build_transforms, IMAGENET_MEAN, IMAGENET_STD

def test_build_transforms_resize() -> None:
    compose = build_transforms(image_size=128)
    assert compose.transforms[0].size == (128, 128)

def test_build_transforms_imagenet_normalization() -> None:
    compose = build_transforms(image_size=256)
    # find the Normalize transform
    normalize_transform = None
    for transform in compose.transforms:
        if isinstance(transform, transforms.Normalize):
            normalize_transform = transform
            break

    assert normalize_transform is not None, "Normalize transform not found in compose transforms"
    assert tuple(normalize_transform.mean) == IMAGENET_MEAN
    assert tuple(normalize_transform.std) == IMAGENET_STD
