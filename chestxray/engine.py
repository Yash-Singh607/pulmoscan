"""Training and validation loops."""

from __future__ import annotations

import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from .calibration import apply_temperature, calibrate_model
from .checkpoint import save_checkpoint
from .config import Config
from .data import compute_class_weights, load_data
from .metrics import evaluate_model, plot_training_curves
from .model import build_model, set_backbone_trainable
from .utils import get_device, get_logger, set_seed
from .visualize import show_sample_images

logger = get_logger(__name__)


def _mixup_batch(images, labels, alpha: float):
    """Mixup augmentation; returns mixed images and label pair + lambda."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def _mixup_loss(criterion, outputs, y_a, y_b, lam: float):
    return lam * criterion(outputs, y_a) + (1.0 - lam) * criterion(outputs, y_b)


def _run_epoch(
    model, loader, criterion, device, optimizer=None, *, use_mixup: bool = False, mixup_alpha: float = 0.2
):
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
                if use_mixup:
                    images, ya, yb, lam = _mixup_batch(images, labels, mixup_alpha)
            outputs = model(images)
            if is_train and use_mixup:
                loss = _mixup_loss(criterion, outputs, ya, yb, lam)
            else:
                loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(1)
            eval_labels = labels if not (is_train and use_mixup) else ya
            running_loss += loss.item() * images.size(0)
            correct += (preds == eval_labels).sum().item()
            total += eval_labels.size(0)
            for t, p in zip(eval_labels.view(-1).tolist(), preds.view(-1).tolist()):
                class_total[t] += 1
                class_correct[t] += int(t == p)

    per_class = [class_correct[c] / class_total[c] for c in class_total if class_total[c]]
    balanced_acc = sum(per_class) / len(per_class) if per_class else 0.0
    return running_loss / max(total, 1), correct / max(total, 1), balanced_acc


def _build_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, epochs_remaining: int):
    return CosineAnnealingLR(optimizer, T_max=max(1, epochs_remaining))


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

    optimizer = _build_optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        cfg.train.lr,
        cfg.train.weight_decay,
    )
    scheduler = _build_scheduler(optimizer, cfg.train.unfreeze_epoch - 1)

    logger.info("[3/4] Training for %d epochs (mixup=%s, early_stop=%d) ...",
                cfg.train.epochs, cfg.train.use_mixup, cfg.train.early_stop_patience)
    best_val_score = 0.0
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    no_improve = 0
    swa_weights: list[dict] = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()

        if epoch == cfg.train.unfreeze_epoch:
            logger.info("Epoch %d: unfreezing backbone for full fine-tuning ...", epoch)
            set_backbone_trainable(model, trainable=True)
            fine_lr = cfg.train.lr * 0.1
            optimizer = _build_optimizer(model, fine_lr, cfg.train.weight_decay)
            remaining = cfg.train.epochs - cfg.train.unfreeze_epoch + 1
            scheduler = _build_scheduler(optimizer, remaining)

        train_loss, train_acc, _ = _run_epoch(
            model,
            bundle.train_loader,
            criterion,
            device,
            optimizer,
            use_mixup=cfg.train.use_mixup,
            mixup_alpha=cfg.train.mixup_alpha,
        )
        val_loss, val_acc, val_bal_acc = _run_epoch(model, bundle.val_loader, criterion, device)
        scheduler.step()

        if cfg.train.use_swa and epoch > cfg.train.epochs - cfg.train.swa_epochs:
            swa_weights.append(copy.deepcopy(model.state_dict()))

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

        if val_bal_acc > best_val_score:
            best_val_score = val_bal_acc
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            save_checkpoint(
                cfg.checkpoint_path,
                model,
                bundle.class_names,
                cfg.model,
                extra={
                    "val_acc": val_acc,
                    "val_balanced_acc": val_bal_acc,
                    "epoch": epoch,
                    "image_size": cfg.data.image_size,
                },
            )
            logger.info(
                "Best model saved (val_bal_acc=%.4f, val_acc=%.4f)", val_bal_acc, val_acc
            )
        else:
            no_improve += 1
            if no_improve >= cfg.train.early_stop_patience:
                logger.info("Early stopping — no val improvement for %d epochs.", no_improve)
                break

    if swa_weights:
        logger.info("Applying SWA over last %d epoch weights ...", len(swa_weights))
        avg = copy.deepcopy(swa_weights[0])
        for key in avg:
            avg[key] = sum(w[key].float() for w in swa_weights) / len(swa_weights)
        model.load_state_dict(avg)
        best_weights = copy.deepcopy(avg)

    logger.info("[4/4] Calibrating on validation set and evaluating on test set ...")
    model.load_state_dict(best_weights)
    cal = calibrate_model(model, bundle.val_loader, device, bundle.class_names)
    temperature = cal["temperature"]
    threshold = cal["optimal_threshold"]

    save_checkpoint(
        cfg.checkpoint_path,
        model,
        bundle.class_names,
        cfg.model,
        extra={
            "val_acc": best_val_acc,
            "val_balanced_acc": best_val_score,
            "epoch": len(history["val_acc"]),
            "image_size": cfg.data.image_size,
            "temperature": temperature,
            "optimal_threshold": threshold,
        },
    )

    metrics = evaluate_model(
        model,
        bundle.test_loader,
        bundle.class_names,
        device,
        save_dir=cfg.train.output_dir,
        tta=cfg.train.eval_tta,
        temperature=temperature,
        threshold=threshold,
    )
    plot_training_curves(
        history,
        save_path=f"{cfg.train.output_dir}/training_curves.png",
    )

    logger.info(
        "Training complete. Best Val Bal-Acc=%.4f (Val Acc=%.4f)", best_val_score, best_val_acc
    )
    return metrics
