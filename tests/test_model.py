"""Model construction and forward-pass tests."""

import torch

from chestxray.config import ModelConfig
from chestxray.model import build_model, set_backbone_trainable


def test_build_model_forward_shape():
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)


def test_freeze_unfreeze_backbone():
    model = build_model(ModelConfig(), pretrained=False)

    set_backbone_trainable(model, trainable=False)
    backbone_grads = [p.requires_grad for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_grads = [p.requires_grad for n, p in model.named_parameters() if n.startswith("fc.")]
    assert not any(backbone_grads)
    assert all(head_grads)

    set_backbone_trainable(model, trainable=True)
    assert all(p.requires_grad for p in model.parameters())


def test_custom_num_classes():
    model = build_model(ModelConfig(num_classes=5), pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 5)
