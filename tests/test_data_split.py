"""Stratified train/val split tests."""

import random

from chestxray.data import _stratified_split


def test_stratified_split_preserves_classes_and_is_disjoint():
    random.seed(0)
    # 100 class-0, 50 class-1
    samples = [("a", 0)] * 100 + [("b", 1)] * 50
    train_idx, val_idx = _stratified_split(samples, val_split=0.2)

    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(samples)

    val_labels = [samples[i][1] for i in val_idx]
    assert val_labels.count(0) == 20  # 20% of 100
    assert val_labels.count(1) == 10  # 20% of 50


def test_stratified_split_always_keeps_at_least_one_val_per_class():
    random.seed(1)
    samples = [("a", 0)] * 3 + [("b", 1)] * 3
    _, val_idx = _stratified_split(samples, val_split=0.01)
    val_labels = [samples[i][1] for i in val_idx]
    assert 0 in val_labels and 1 in val_labels
