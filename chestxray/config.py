"""Centralised configuration.

All tunable parameters live here as typed dataclasses so that training,
inference, and serving share a single source of truth. Values can be
overridden via CLI flags or environment variables (see ``from_env``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

# ImageNet normalisation statistics (ResNet-50 was pretrained with these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CLASS_NAMES = ("NORMAL", "PNEUMONIA")


@dataclass
class DataConfig:
    data_dir: str = "data/chest_xray"
    image_size: int = 224
    batch_size: int = 32
    # On Windows, multiprocessing DataLoaders frequently fail; default to 0
    # there and let users opt into workers explicitly.
    num_workers: int = 0 if os.name == "nt" else 4
    pin_memory: bool = True
    # Cap the number of images per split (for quick CPU smoke runs). None = full.
    limit: int | None = None
    # The Kaggle dataset ships a tiny (16-image) val folder which makes model
    # selection unreliable. When True we carve a proper stratified validation
    # set out of the training data instead (and fold the tiny val folder in).
    resplit_val: bool = True
    val_split: float = 0.15


@dataclass
class ModelConfig:
    num_classes: int = 2
    dropout1: float = 0.4
    dropout2: float = 0.3
    hidden_dim: int = 256
    pretrained: bool = True


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-3
    weight_decay: float = 1e-4
    unfreeze_epoch: int = 6
    scheduler_step: int = 5
    scheduler_gamma: float = 0.5
    use_class_weights: bool = True
    # Label smoothing regularises the loss and typically improves generalisation
    # and probability calibration on this dataset.
    label_smoothing: float = 0.05
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    checkpoint_name: str = "best_model.pth"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(cls) -> "Config":
        """Build a config, overriding selected fields from environment vars."""
        cfg = cls()
        cfg.data.data_dir = os.getenv("CXR_DATA_DIR", cfg.data.data_dir)
        cfg.train.checkpoint_dir = os.getenv(
            "CXR_CHECKPOINT_DIR", cfg.train.checkpoint_dir
        )
        cfg.train.output_dir = os.getenv("CXR_OUTPUT_DIR", cfg.train.output_dir)
        if (bs := os.getenv("CXR_BATCH_SIZE")) is not None:
            cfg.data.batch_size = int(bs)
        if (nw := os.getenv("CXR_NUM_WORKERS")) is not None:
            cfg.data.num_workers = int(nw)
        return cfg

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.train.checkpoint_dir, self.train.checkpoint_name)
