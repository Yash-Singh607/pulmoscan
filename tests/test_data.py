"""Data loading, transforms, and class-weight tests."""

import pytest
import torch

from chestxray.config import DataConfig
from chestxray.data import build_transforms, compute_class_weights, load_data


def test_build_transforms_output_shape(rgb_image):
    train_tf, eval_tf = build_transforms(image_size=224)
    assert train_tf(rgb_image).shape == (3, 224, 224)
    assert eval_tf(rgb_image).shape == (3, 224, 224)


def test_compute_class_weights_inverse_frequency():
    # Minority class should receive a larger weight.
    weights = compute_class_weights([100, 300])
    assert weights[0] > weights[1]
    assert torch.isfinite(weights).all()


def test_compute_class_weights_handles_zero():
    weights = compute_class_weights([0, 10])
    assert torch.isfinite(weights).all()


def test_load_data_counts(tiny_dataset):
    cfg = DataConfig(
        data_dir=tiny_dataset,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        resplit_val=False,
    )
    bundle = load_data(cfg)
    assert bundle.class_names == ["NORMAL", "PNEUMONIA"]
    assert bundle.class_counts == [3, 5]
    images, labels = next(iter(bundle.train_loader))
    assert images.shape[1:] == (3, 224, 224)
    assert labels.ndim == 1


def test_load_data_missing_split_raises(tmp_path):
    cfg = DataConfig(data_dir=str(tmp_path), num_workers=0)
    with pytest.raises(FileNotFoundError):
        load_data(cfg)


def test_load_data_limit_caps_split_size(tiny_dataset):
    cfg = DataConfig(
        data_dir=tiny_dataset, batch_size=2, num_workers=0, pin_memory=False, limit=4
    )
    bundle = load_data(cfg)
    assert len(bundle.train_loader.dataset) == 4  # train has 8 images, capped to 4
    assert sum(bundle.class_counts) == 4
    assert bundle.class_names == ["NORMAL", "PNEUMONIA"]
