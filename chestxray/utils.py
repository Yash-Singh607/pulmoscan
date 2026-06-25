"""Cross-cutting utilities: logging, reproducibility, and device selection."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging once, idempotently.

    Honours the ``CXR_LOG_LEVEL`` environment variable when set.
    """
    env_level = os.getenv("CXR_LOG_LEVEL")
    if env_level:
        level = env_level.upper()
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or honour an explicit preference."""
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
