"""Dataset and prediction visualisation helpers."""

from __future__ import annotations

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from .utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def show_sample_images(loader, class_names, n=8, save_path=None):
    """Display/save a grid of sample training images with labels."""
    images, labels = next(iter(loader))
    images, labels = images[:n], labels[:n]

    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)

    _, axes = plt.subplots(2, max(1, n // 2), figsize=(14, 6))
    axes = np.array(axes).flatten()

    for i, (img, lbl) in enumerate(zip(images, labels)):
        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(std * img_np + mean, 0, 1)
        axes[i].imshow(img_np, cmap="gray")
        axes[i].set_title(
            class_names[lbl.item()],
            fontsize=9,
            fontweight="bold",
            color="#dc2626" if lbl.item() == 1 else "#16a34a",
        )
        axes[i].axis("off")

    plt.suptitle("Sample Training Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "sample images")


def visualize_class_distribution(data_dir, save_path=None):
    """Bar chart showing per-split class distribution."""
    splits = ["train", "val", "test"]
    colors = {"NORMAL": "#16a34a", "PNEUMONIA": "#dc2626"}

    data = {}
    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.isdir(split_path):
            continue
        counts = {}
        for cls in os.listdir(split_path):
            cls_path = os.path.join(split_path, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len(os.listdir(cls_path))
        data[split] = counts

    x = np.arange(len(splits))
    width = 0.35
    _, ax = plt.subplots(figsize=(8, 5))
    class_names = sorted({cls for s in data.values() for cls in s})

    for i, cls in enumerate(class_names):
        counts = [data.get(s, {}).get(cls, 0) for s in splits]
        offset = (i - len(class_names) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            counts,
            width,
            label=cls,
            color=colors.get(cls, None),
            alpha=0.85,
        )
        for j, cnt in enumerate(counts):
            ax.text(x[j] + offset, cnt + 20, str(cnt), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution per Split", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path, "class distribution")


def apply_clahe(image_path, clip_limit=2.0, tile_size=(8, 8)):
    """Apply CLAHE contrast enhancement to a grayscale X-ray."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    return img, clahe.apply(img)


def show_clahe_comparison(image_path, save_path=None):
    """Side-by-side original vs CLAHE-enhanced X-ray."""
    original, enhanced = apply_clahe(image_path)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original")
    ax1.axis("off")
    ax2.imshow(enhanced, cmap="gray")
    ax2.set_title("CLAHE Enhanced")
    ax2.axis("off")
    plt.suptitle("Preprocessing: CLAHE Contrast Enhancement", fontweight="bold")
    plt.tight_layout()
    _save_or_show(save_path, "CLAHE comparison")


def _save_or_show(save_path, label):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved %s -> %s", label, save_path)
    else:
        plt.show()
    plt.close()
