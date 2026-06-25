"""Dataset loading, transforms, and class-imbalance helpers."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

from .config import IMAGENET_MEAN, IMAGENET_STD, DataConfig
from .utils import get_logger

logger = get_logger(__name__)


def build_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transforms, eval_transforms)``."""
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            # Mild random crop/zoom helps the model generalise to differently
            # framed radiographs without distorting anatomy too much.
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def inference_transform(image_size: int = 224) -> transforms.Compose:
    """Transform used for a single image at inference time."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    class_names: list[str]
    class_counts: list[int]


def _require_split(data_dir: str, split: str) -> str:
    path = os.path.join(data_dir, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Expected dataset split '{split}' at '{path}'. "
            f"Run `python -m chestxray.cli setup-data` first."
        )
    return path


def _maybe_subset(dataset, limit: int | None):
    """Optionally return a random Subset capped at ``limit`` items.

    Returns ``(dataset_or_subset, indices_or_None)``. Relies on the caller
    having seeded ``random`` for reproducibility.
    """
    if not limit or limit >= len(dataset):
        return dataset, None
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    indices = indices[:limit]
    return Subset(dataset, indices), indices


def _stratified_split(samples, val_split: float) -> tuple[list[int], list[int]]:
    """Stratified train/val index split that preserves per-class ratios.

    Uses the global ``random`` state (seeded by the caller) for reproducibility.
    """
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)

    train_idx, val_idx = [], []
    for idxs in by_class.values():
        idxs = idxs[:]
        random.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_split)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    random.shuffle(train_idx)
    return train_idx, val_idx


def load_data(cfg: DataConfig) -> DataBundle:
    """Load the train/val/test splits into DataLoaders.

    When ``cfg.resplit_val`` is set, a proper stratified validation set is
    carved out of the training data (and the tiny shipped ``val`` folder is
    folded in), which makes model selection far more reliable than the
    dataset's default 16-image validation split.
    """
    train_tf, eval_tf = build_transforms(cfg.image_size)
    train_dir = _require_split(cfg.data_dir, "train")
    test_dir = _require_split(cfg.data_dir, "test")

    # Two views of the training folder: augmented (train) and clean (val/eval).
    train_tf_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    train_eval_ds = datasets.ImageFolder(train_dir, transform=eval_tf)
    test_full = datasets.ImageFolder(test_dir, transform=eval_tf)
    class_names = list(train_tf_ds.classes)
    samples = train_tf_ds.samples

    if cfg.resplit_val:
        tr_idx, va_idx = _stratified_split(samples, cfg.val_split)
        train_ds = Subset(train_tf_ds, tr_idx)
        train_label_idx = tr_idx

        val_parts: list = [Subset(train_eval_ds, va_idx)]
        val_dir = os.path.join(cfg.data_dir, "val")
        if os.path.isdir(val_dir):  # fold the shipped tiny val set in
            val_parts.append(datasets.ImageFolder(val_dir, transform=eval_tf))
        val_ds = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    else:
        val_dir = _require_split(cfg.data_dir, "val")
        train_ds = train_tf_ds
        train_label_idx = list(range(len(samples)))
        val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)

    # Quick-mode caps (CPU smoke runs).
    train_ds, sub_idx = _maybe_subset(train_ds, cfg.limit)
    if sub_idx is not None:
        train_label_idx = [train_label_idx[i] for i in sub_idx]
    val_ds, _ = _maybe_subset(val_ds, cfg.limit)
    test_ds, _ = _maybe_subset(test_full, cfg.limit)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    class_counts = [0] * len(class_names)
    for i in train_label_idx:
        class_counts[samples[i][1]] += 1

    if cfg.limit:
        logger.info("Quick mode: capped to %d images per split", cfg.limit)
    if cfg.resplit_val:
        logger.info("Validation re-split from train (val_split=%.2f)", cfg.val_split)
    logger.info("Classes : %s", class_names)
    logger.info("Train   : %d images %s", len(train_ds), class_counts)
    logger.info("Val     : %d images", len(val_ds))
    logger.info("Test    : %d images", len(test_ds))

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        class_counts=class_counts,
    )


def compute_class_weights(class_counts: list[int]) -> torch.Tensor:
    """Inverse-frequency class weights for a weighted CrossEntropyLoss.

    This addresses the dataset's strong PNEUMONIA/NORMAL imbalance, which the
    original training code ignored despite the docs claiming otherwise.
    """
    counts = torch.tensor(class_counts, dtype=torch.float)
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights
