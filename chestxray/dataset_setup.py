"""Download and verify the Kaggle Chest X-Ray dataset.

Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

from .utils import get_logger

logger = get_logger(__name__)

DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_TARGET = "data/chest_xray"
SPLITS = ("train", "val", "test")
CLASSES = ("NORMAL", "PNEUMONIA")


def _kaggle_available() -> bool:
    try:
        import kaggle  # noqa: F401

        return True
    except (ImportError, OSError):
        # OSError is raised by kaggle when credentials are missing on import.
        return False


def download_dataset(target_dir: str = DEFAULT_TARGET) -> None:
    logger.info("Chest X-Ray dataset setup -> %s", target_dir)

    if all(os.path.isdir(os.path.join(target_dir, s)) for s in SPLITS):
        logger.info("Dataset already present. Nothing to do.")
        print_stats(target_dir)
        return

    if not _kaggle_available():
        logger.warning("kaggle package not importable; attempting install ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle", "-q"], check=False)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise SystemExit(
            "Kaggle API credentials not found at ~/.kaggle/kaggle.json.\n"
            "Create a token at https://www.kaggle.com (Account -> API -> Create New Token), "
            "place it there, then re-run."
        )

    logger.info("Downloading dataset '%s' ...", DATASET_SLUG)
    os.makedirs("data", exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", "data", "--unzip"],
        check=True,
    )

    if not os.path.isdir(target_dir):
        for candidate in ("data/chest-xray-pneumonia/chest_xray", "data/chest_xray"):
            if os.path.isdir(candidate) and candidate != target_dir:
                shutil.move(candidate, target_dir)
                break

    if not os.path.isdir(target_dir):
        raise SystemExit(
            f"Could not locate dataset after download. Extract it to '{target_dir}' manually."
        )

    logger.info("Dataset ready at '%s'", target_dir)
    print_stats(target_dir)


def print_stats(target_dir: str = DEFAULT_TARGET) -> None:
    print("\nDataset statistics:")
    print(f"  {'Split':<8}  {'NORMAL':>8}  {'PNEUMONIA':>10}  {'Total':>7}")
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 10}  {'-' * 7}")
    for split in SPLITS:
        split_dir = os.path.join(target_dir, split)
        if not os.path.isdir(split_dir):
            continue
        counts = {}
        for cls in CLASSES:
            cls_dir = os.path.join(split_dir, cls)
            counts[cls] = len(os.listdir(cls_dir)) if os.path.isdir(cls_dir) else 0
        total = sum(counts.values())
        print(f"  {split.capitalize():<8}  {counts['NORMAL']:>8}  {counts['PNEUMONIA']:>10}  {total:>7}")
    print()
