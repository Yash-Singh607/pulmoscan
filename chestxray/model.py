"""Model definition shared by training and inference.

A single ``build_model`` here eliminates the previous duplication between
``train.py`` and ``predict.py`` (the classifier head was defined twice).
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

from .config import ModelConfig


def build_model(cfg: ModelConfig | None = None, pretrained: bool | None = None) -> nn.Module:
    """Construct a ResNet-50 with a custom two-layer classifier head.

    Args:
        cfg: Model hyper-parameters. Defaults to :class:`ModelConfig`.
        pretrained: Override ``cfg.pretrained``. Use ``False`` at inference
            time when weights are loaded from a checkpoint.
    """
    cfg = cfg or ModelConfig()
    use_pretrained = cfg.pretrained if pretrained is None else pretrained

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = models.resnet50(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=cfg.dropout1),
        nn.Linear(in_features, cfg.hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=cfg.dropout2),
        nn.Linear(cfg.hidden_dim, cfg.num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze every backbone parameter (the classifier head
    in ``model.fc`` is always left trainable)."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable
