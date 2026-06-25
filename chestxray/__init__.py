"""Chest X-Ray Pneumonia Classifier — production package.

A transfer-learning pipeline (ResNet-50) for detecting pneumonia from chest
X-rays, with training, evaluation, Grad-CAM interpretability, batch/single
inference, and a REST serving API.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chestxray")
except PackageNotFoundError:  # pragma: no cover - editable/source checkout
    __version__ = "0.1.0"

__all__ = ["__version__"]
