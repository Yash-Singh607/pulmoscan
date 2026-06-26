"""Post-training calibration: temperature scaling and threshold tuning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import ImageFolder

from .utils import get_logger

logger = get_logger(__name__)


class _TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)


@torch.no_grad()
def _collect_logits(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device)
        logits_list.append(model(images).cpu())
        labels_list.append(labels.cpu())
    return torch.cat(logits_list), torch.cat(labels_list)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a single temperature on validation logits (temperature scaling)."""
    scaler = _TemperatureScaler()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.05, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = criterion(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(scaler.temperature.item())
    logger.info("Fitted temperature scaling: T=%.4f", temperature)
    return max(temperature, 1e-3)


def find_optimal_threshold(
    probs_positive: np.ndarray,
    labels: np.ndarray,
    *,
    positive_label: int = 1,
) -> float:
    """Pick the threshold on the positive class that maximises F1 on validation."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs_positive >= t).astype(int)
        binary_labels = (labels == positive_label).astype(int)
        tp = int(((preds == 1) & (binary_labels == 1)).sum())
        fp = int(((preds == 1) & (binary_labels == 0)).sum())
        fn = int(((preds == 0) & (binary_labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    logger.info("Optimal pneumonia threshold on val: %.3f (F1=%.4f)", best_t, best_f1)
    return best_t


def calibrate_model(
    model,
    val_loader: DataLoader,
    device,
    class_names: list[str],
) -> dict[str, float]:
    """Fit temperature and decision threshold on the validation loader."""
    logits, labels = _collect_logits(model, val_loader, device)
    temperature = fit_temperature(logits, labels)
    scaled = torch.softmax(logits / temperature, dim=1).numpy()
    positive_idx = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1
    threshold = find_optimal_threshold(scaled[:, positive_idx], labels.numpy(), positive_label=positive_idx)
    return {"temperature": temperature, "optimal_threshold": threshold, "positive_class": class_names[positive_idx]}


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(logits / max(temperature, 1e-3), dim=-1)
