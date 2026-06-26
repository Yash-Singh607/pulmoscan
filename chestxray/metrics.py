"""Evaluation metrics, confusion matrix, ROC curve, and training curves."""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for servers/CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from torch.utils.data import ConcatDataset, Subset
from torchvision.datasets import ImageFolder

from .calibration import apply_temperature
from .utils import get_logger  # noqa: E402

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_model(
    model,
    loader,
    class_names,
    device,
    save_dir="outputs",
    tta: bool = False,
    *,
    temperature: float = 1.0,
    threshold: float | None = None,
    positive_idx: int | None = None,
    export_errors: bool = True,
) -> dict:
    """Full evaluation: accuracy, F1, AUC, confusion matrix, ROC curve.

    With ``tta=True``, averages predictions with a horizontal flip (small boost).
    When ``threshold`` is set, binary decisions use the positive-class probability.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    pos_idx = positive_idx if positive_idx is not None else (
        class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1
    )

    all_preds, all_labels, all_probs = [], [], []
    error_rows: list[dict] = []
    sample_idx = 0

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        if tta:
            flipped = torch.flip(images, dims=[3])
            outputs = (outputs + model(flipped)) / 2
        probs = apply_temperature(outputs, temperature).cpu().numpy()
        batch_size = probs.shape[0]

        if threshold is not None:
            other_idx = 0 if pos_idx == 1 else 1
            batch_preds = np.where(probs[:, pos_idx] >= threshold, pos_idx, other_idx)
        else:
            batch_preds = np.argmax(probs, axis=1)

        all_probs.extend(probs[:, pos_idx])
        all_preds.extend(batch_preds)
        all_labels.extend(labels.numpy())

        if export_errors:
            for i in range(batch_size):
                true_l = int(labels[i].item())
                pred_l = int(batch_preds[i])
                if pred_l != true_l:
                    path = _resolve_sample_path(loader.dataset, sample_idx + i)
                    error_rows.append(
                        {
                            "path": path,
                            "true_label": class_names[true_l],
                            "predicted_label": class_names[pred_l],
                            "confidence": float(probs[i, pred_l]),
                            "prob_pneumonia": float(probs[i, pos_idx]),
                        }
                    )
            sample_idx += batch_size

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "accuracy": float((all_preds == all_labels).mean()),
        "f1": float(f1_score(all_labels, all_preds, average="weighted")),
        "precision": float(
            precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        ),
        "auc": float(roc_auc_score(all_labels, all_probs)),
        "temperature": float(temperature),
    }
    if threshold is not None:
        metrics["optimal_threshold"] = float(threshold)

    if tta:
        logger.info("Test evaluation used horizontal-flip TTA.")
    logger.info("Test Accuracy : %.4f", metrics["accuracy"])
    logger.info("F1 Score      : %.4f", metrics["f1"])
    logger.info("Precision     : %.4f", metrics["precision"])
    logger.info("Recall        : %.4f", metrics["recall"])
    logger.info("ROC-AUC       : %.4f", metrics["auc"])

    report = classification_report(all_labels, all_preds, target_names=class_names)
    logger.info("Classification report:\n%s", report)

    _plot_confusion_matrix(
        all_labels, all_preds, class_names, os.path.join(save_dir, "confusion_matrix.png")
    )
    _plot_roc_curve(all_labels, all_probs, os.path.join(save_dir, "roc_curve.png"))

    with open(os.path.join(save_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key:<10}: {value:.4f}\n")
        f.write("\n")
        f.write(report)

    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if export_errors and error_rows:
        error_path = os.path.join(save_dir, "error_analysis.json")
        payload = {
            "count": len(error_rows),
            "misclassified": error_rows[:100],
        }
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        metrics["error_analysis_path"] = error_path
        metrics["misclassified_count"] = len(error_rows)
        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved error analysis (%d items) -> %s", len(error_rows), error_path)

    return metrics


def _resolve_sample_path(dataset, index: int) -> str:
    """Best-effort path lookup for ImageFolder / Subset / ConcatDataset."""
    if isinstance(dataset, Subset):
        return _resolve_sample_path(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, ConcatDataset):
        offset = 0
        for child in dataset.datasets:
            if index < offset + len(child):
                return _resolve_sample_path(child, index - offset)
            offset += len(child)
    if isinstance(dataset, ImageFolder):
        return dataset.samples[index][0]
    return f"index:{index}"


def _plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix -> %s", save_path)


def _plot_roc_curve(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved ROC curve -> %s", save_path)


def plot_training_curves(history, save_path="outputs/training_curves.png"):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], "o-", color="#ef4444", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "s-", color="#f97316", label="Val Loss")
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "o-", color="#2563eb", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "s-", color="#16a34a", label="Val Acc")
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle("Training Curves — Chest X-Ray Classifier", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Saved training curves -> %s", save_path)
