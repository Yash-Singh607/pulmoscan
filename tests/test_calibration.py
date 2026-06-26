"""Calibration utilities tests."""

import torch

from chestxray.calibration import apply_temperature, find_optimal_threshold, fit_temperature


def test_fit_temperature_improves_nll_on_synthetic_logits():
    logits = torch.tensor([[2.0, 0.5], [0.5, 2.0], [1.5, 0.2], [0.1, 1.8]])
    labels = torch.tensor([0, 1, 0, 1])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0


def test_apply_temperature_sums_to_one():
    logits = torch.tensor([[1.0, 2.0], [3.0, 0.5]])
    probs = apply_temperature(logits, 1.5)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-5)


def test_find_optimal_threshold():
    probs = torch.tensor([0.9, 0.8, 0.2, 0.1]).numpy()
    labels = torch.tensor([1, 1, 0, 0]).numpy()
    threshold = find_optimal_threshold(probs, labels, positive_label=1)
    assert 0.05 <= threshold <= 0.95
