import pytest
from src.data import make_split_indices

dataset_size = 100
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1
seed = 42


def test_make_split_indices_lengths() -> None:
    split = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    assert len(split.train) == int(
        train_ratio * dataset_size
    )  # note: assuming int() as rounding policy
    assert len(split.val) == int(val_ratio * dataset_size)
    assert len(split.test) == dataset_size - len(split.train) - len(split.val)


def test_make_split_indices_raises_on_bad_ratios() -> None:
    with pytest.raises(ValueError):
        make_split_indices(
            dataset_size=dataset_size,
            train_ratio=0.6,
            val_ratio=0.0,
            test_ratio=0.2,
            seed=seed,
        )


def test_make_split_indices_same_seed() -> None:
    split_1 = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=1,
    )

    split_2 = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=1,
    )

    assert split_1 == split_2


def test_make_split_indices_different_seed() -> None:
    split_1 = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=1,
    )

    split_2 = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=2,
    )

    assert split_1 != split_2


def test_make_split_indices_disjointness() -> None:
    split = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    total_indices = split.train + split.val + split.test
    assert len(total_indices) == len(set(total_indices))


def test_make_split_indices_bounds() -> None:
    split = make_split_indices(
        dataset_size=dataset_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    for idx in split.train + split.val + split.test:
        assert 0 <= idx < dataset_size
