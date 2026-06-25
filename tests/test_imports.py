"""Smoke tests: the package and all submodules import cleanly.

This directly guards against the original bug where ``train.py``/``eda.py``
imported a non-existent ``utils`` package.
"""

import importlib

import pytest

MODULES = [
    "chestxray",
    "chestxray.config",
    "chestxray.utils",
    "chestxray.model",
    "chestxray.data",
    "chestxray.metrics",
    "chestxray.visualize",
    "chestxray.gradcam",
    "chestxray.checkpoint",
    "chestxray.engine",
    "chestxray.inference",
    "chestxray.eda",
    "chestxray.dataset_setup",
    "chestxray.cli",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    assert importlib.import_module(name) is not None
