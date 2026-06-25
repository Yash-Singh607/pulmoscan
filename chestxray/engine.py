"""Training and validation loops."""

from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from .checkpoint import save_checkpoint
from .config import Config
from .data import compute_class_weights, load_data
from .metrics import evaluate_model, plot_training_curves
from .model import build_model, set_backbone_trainable
from .utils import get_device, get_logger, set_seed
from .visualize import show_sample_images

logger = get_logger(__name__)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    running_loss, correct, total = 0.0, 0, 0
    class_correct: dict[int, int] = defaultdict(int)
    class_total: dict[int, int] = defaultdict(int)

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(1)
            running_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            for t, p in zip(labels.view(-1).tolist(), preds.view(-1).tolist()):
                class_total[t] += 1
                class_correct[t] += int(t == p)

    # Balanced accuracy = mean per-class recall (robust to class imbalance).
    per_class = [class_correct[c] / class_total[c] for c in class_total if class_total[c]]
    balanced_acc = sum(per_class) / len(per_class) if per_class else 0.0
    return running_loss / max(total, 1), correct / max(total, 1), balanced_acc


def train(cfg: Config) -> dict:
    """Full training pipeline. Returns the test-set metrics dict."""
    set_seed(cfg.train.seed)
    device = get_device()
    logger.info("Device: %s", device)

    logger.info("[1/4] Loading dataset ...")
    bundle = load_data(cfg.data)

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)
    show_sample_images(
        bundle.train_loader,
        bundle.class_names,
        save_path=os.path.join(cfg.train.output_dir, "sample_images.png"),
    )

    logger.info("[2/4] Building model (ResNet-50 pretrained) ...")
    cfg.model.num_classes = len(bundle.class_names)
    model = build_model(cfg.model).to(device)
    set_backbone_trainable(model, trainable=False)

    smoothing = cfg.train.label_smoothing
    if cfg.train.use_class_weights:
        weights = compute_class_weights(bundle.class_counts).to(device)
        logger.info("Using class weights: %s", weights.tolist())
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = StepLR(
        optimizer, step_size=cfg.train.scheduler_step, gamma=cfg.train.scheduler_gamma
    )

    logger.info("[3/4] Training for %d epochs ...", cfg.train.epochs)
    best_val_score = 0.0
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()

        if epoch == cfg.train.unfreeze_epoch:
            logger.info("Epoch %d: unfreezing backbone for full fine-tuning ...", epoch)
            set_backbone_trainable(model, trainable=True)
            optimizer = optim.Adam(
                model.parameters(),
                lr=cfg.train.lr * 0.1,
                weight_decay=cfg.train.weight_decay,
            )
            scheduler = StepLR(optimizer, step_size=3, gamma=cfg.train.scheduler_gamma)

        train_loss, train_acc, _ = _run_epoch(
            model, bundle.train_loader, criterion, device, optimizer
        )
        val_loss, val_acc, val_bal_acc = _run_epoch(model, bundle.val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(
            "Epoch [%3d/%d]  train_loss=%.4f acc=%.4f | "
            "val_loss=%.4f acc=%.4f bal_acc=%.4f (%.1fs)",
            epoch,
            cfg.train.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_bal_acc,
            time.time() - t0,
        )

        # Select on balanced accuracy (robust to the dataset's class imbalance).
        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(
                cfg.checkpoint_path,
                model,
                bundle.class_names,
                cfg.model,
                extra={"val_acc": val_acc, "val_balanced_acc": val_bal_acc, "epoch": epoch},
            )
            logger.info(
                "Best model saved (val_bal_acc=%.4f, val_acc=%.4f)", val_bal_acc, val_acc
            )

    logger.info("[4/4] Evaluating on test set ...")
    model.load_state_dict(best_weights)
    metrics = evaluate_model(
        model, bundle.test_loader, bundle.class_names, device, save_dir=cfg.train.output_dir
    )
    plot_training_curves(
        history,
        save_path=f"{cfg.train.output_dir}/training_curves.png",
    )

    logger.info(
        "Training complete. Best Val Bal-Acc=%.4f (Val Acc=%.4f)", best_val_score, best_val_acc
    )
    return metrics
