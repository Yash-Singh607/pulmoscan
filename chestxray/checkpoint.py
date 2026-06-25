"""Checkpoint serialisation with embedded metadata.

Checkpoints store the model weights *and* the class names + config used to
train them, so inference no longer has to hard-code ``["NORMAL", "PNEUMONIA"]``.
Loading is backward-compatible with bare ``state_dict`` checkpoints.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from .config import DEFAULT_CLASS_NAMES, ModelConfig


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    class_names: list[str],
    model_cfg: ModelConfig,
    extra: dict[str, Any] | None = None,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "class_names": list(class_names),
        "model_config": vars(model_cfg),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu") -> dict[str, Any]:
    """Return a normalised dict with ``state_dict``, ``class_names``,
    and ``model_config`` keys regardless of checkpoint format."""
    raw = torch.load(path, map_location=map_location)

    if isinstance(raw, dict) and "state_dict" in raw:
        class_names = raw.get("class_names", list(DEFAULT_CLASS_NAMES))
        model_config = raw.get("model_config", vars(ModelConfig()))
        return {
            "state_dict": raw["state_dict"],
            "class_names": class_names,
            "model_config": model_config,
        }

    # Legacy: a bare state_dict.
    return {
        "state_dict": raw,
        "class_names": list(DEFAULT_CLASS_NAMES),
        "model_config": vars(ModelConfig()),
    }
