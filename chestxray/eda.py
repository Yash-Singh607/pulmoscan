"""Exploratory Data Analysis for the Chest X-Ray dataset."""

from __future__ import annotations

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .utils import get_logger  # noqa: E402
from .visualize import visualize_class_distribution  # noqa: E402

logger = get_logger(__name__)

CLASSES = ("NORMAL", "PNEUMONIA")
COLORS = {"NORMAL": "#16a34a", "PNEUMONIA": "#dc2626"}


def pixel_intensity_analysis(data_dir, n_samples=200, save_dir="outputs/eda"):
    os.makedirs(save_dir, exist_ok=True)
    all_pixels = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, "train", cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir)[:n_samples]:
            img = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                all_pixels[cls].extend(img.flatten().tolist())

    plt.figure(figsize=(9, 4))
    for cls in CLASSES:
        pixels = np.array(all_pixels[cls])
        plt.hist(
            pixels,
            bins=64,
            alpha=0.6,
            color=COLORS[cls],
            label=f"{cls} (n={len(pixels) // 1000}k)",
            density=True,
        )
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Density")
    plt.title("Pixel Intensity Distribution — NORMAL vs PNEUMONIA", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "intensity_distribution.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved intensity distribution -> %s", out)


def image_size_analysis(data_dir, save_dir="outputs/eda"):
    os.makedirs(save_dir, exist_ok=True)
    widths, heights, labels = [], [], []

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, "train", cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir)[:300]:
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is not None:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                labels.append(cls)

    plt.figure(figsize=(7, 5))
    for cls in CLASSES:
        idxs = [i for i, l in enumerate(labels) if l == cls]
        plt.scatter(
            [widths[i] for i in idxs],
            [heights[i] for i in idxs],
            c=COLORS[cls],
            label=cls,
            alpha=0.5,
            s=15,
        )
    plt.xlabel("Width (px)")
    plt.ylabel("Height (px)")
    plt.title("Image Dimensions in Training Set", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "image_sizes.png")
    plt.savefig(out, dpi=150)
    plt.close()
    logger.info("Saved image size scatter -> %s", out)


def sample_grid(data_dir, save_dir="outputs/eda", n_per_class=6):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(
        len(CLASSES), n_per_class, figsize=(n_per_class * 2.5, len(CLASSES) * 2.8)
    )

    for r, cls in enumerate(CLASSES):
        cls_dir = os.path.join(data_dir, "train", cls)
        if not os.path.isdir(cls_dir):
            continue
        for c, fname in enumerate(os.listdir(cls_dir)[:n_per_class]):
            img = cv2.imread(os.path.join(cls_dir, fname), cv2.IMREAD_GRAYSCALE)
            axes[r][c].imshow(img, cmap="gray")
            axes[r][c].axis("off")
            if c == 0:
                axes[r][c].set_ylabel(
                    cls, fontsize=10, fontweight="bold", rotation=0, labelpad=60, color=COLORS[cls]
                )

    plt.suptitle("Raw X-Ray Samples from Kaggle Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "sample_grid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample grid -> %s", out)


def run_eda(data_dir: str, save_dir: str = "outputs/eda") -> None:
    logger.info("Running EDA on '%s' ...", data_dir)
    visualize_class_distribution(
        data_dir, save_path=os.path.join(save_dir, "class_distribution.png")
    )
    pixel_intensity_analysis(data_dir, save_dir=save_dir)
    image_size_analysis(data_dir, save_dir=save_dir)
    sample_grid(data_dir, save_dir=save_dir)
    logger.info("All EDA plots saved to '%s'", save_dir)
