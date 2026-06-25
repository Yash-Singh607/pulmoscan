"""Model registry — list and load versioned checkpoints."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .inference import Classifier


def _checkpoint_dir() -> Path:
    return Path(os.getenv("CXR_CHECKPOINT_DIR", "checkpoints"))


def _default_path() -> str:
    return os.getenv("CXR_CHECKPOINT_PATH", str(_checkpoint_dir() / "best_model.pth"))


def list_models() -> list[dict]:
    """Scan the checkpoint directory and return metadata for each ``.pth`` file."""
    models: list[dict] = []
    ckpt_dir = _checkpoint_dir()
    if not ckpt_dir.is_dir():
        return models

    default = Path(_default_path()).resolve()
    for path in sorted(ckpt_dir.glob("*.pth")):
        entry = {
            "id": path.stem,
            "path": str(path),
            "is_default": path.resolve() == default,
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
        try:
            meta = load_checkpoint(str(path), map_location="cpu")
            entry["class_names"] = meta.get("class_names", [])
            extra = meta.get("extra") or {}
            entry["val_acc"] = extra.get("val_acc")
            entry["val_balanced_acc"] = extra.get("val_balanced_acc")
            entry["epoch"] = extra.get("epoch")
        except Exception:
            entry["class_names"] = []
        models.append(entry)
    return models


def resolve_checkpoint(model_id: str | None) -> str:
    if not model_id or model_id in ("default", "best"):
        return _default_path()
    ckpt_dir = _checkpoint_dir()
    direct = ckpt_dir / f"{model_id}.pth"
    if direct.is_file():
        return str(direct)
    for path in ckpt_dir.glob("*.pth"):
        if path.stem == model_id:
            return str(path)
    raise FileNotFoundError(f"Model '{model_id}' not found in {ckpt_dir}")


@lru_cache(maxsize=4)
def get_classifier(model_id: str | None = None) -> Classifier:
    return Classifier(resolve_checkpoint(model_id))


def clear_classifier_cache() -> None:
    get_classifier.cache_clear()
