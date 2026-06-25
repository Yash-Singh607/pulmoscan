"""Checkpoint round-trip and backward-compatibility tests."""

import torch

from chestxray.checkpoint import load_checkpoint, save_checkpoint
from chestxray.config import ModelConfig
from chestxray.model import build_model


def test_checkpoint_roundtrip(tmp_path):
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    path = tmp_path / "model.pth"
    save_checkpoint(str(path), model, ["NORMAL", "PNEUMONIA"], ModelConfig(num_classes=2))

    ckpt = load_checkpoint(str(path))
    assert ckpt["class_names"] == ["NORMAL", "PNEUMONIA"]
    assert "state_dict" in ckpt

    reloaded = build_model(ModelConfig(num_classes=2), pretrained=False)
    reloaded.load_state_dict(ckpt["state_dict"])


def test_load_legacy_bare_state_dict(tmp_path):
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    path = tmp_path / "legacy.pth"
    torch.save(model.state_dict(), path)

    ckpt = load_checkpoint(str(path))
    assert ckpt["class_names"] == ["NORMAL", "PNEUMONIA"]
    reloaded = build_model(ModelConfig(num_classes=2), pretrained=False)
    reloaded.load_state_dict(ckpt["state_dict"])
