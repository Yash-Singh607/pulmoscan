"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from chestxray.checkpoint import save_checkpoint
from chestxray.config import ModelConfig
from chestxray.model import build_model


@pytest.fixture
def rgb_image() -> Image.Image:
    arr = (np.random.rand(256, 256, 3) * 255).astype("uint8")
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def checkpoint(tmp_path):
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    path = tmp_path / "best_model.pth"
    save_checkpoint(str(path), model, ["NORMAL", "PNEUMONIA"], ModelConfig(num_classes=2))
    return str(path)


@pytest.fixture
def tiny_dataset(tmp_path):
    """Create a minimal ImageFolder-style dataset with 2 classes/splits."""
    root = tmp_path / "chest_xray"
    for split in ("train", "val", "test"):
        for cls, n in (("NORMAL", 3), ("PNEUMONIA", 5)):
            d = root / split / cls
            d.mkdir(parents=True)
            for i in range(n):
                arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
                Image.fromarray(arr, mode="RGB").save(d / f"{cls}_{i}.png")
    return str(root)
