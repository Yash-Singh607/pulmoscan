"""SQLite store and release bundle tests."""

import importlib

import pytest

from chestxray.checkpoint import save_checkpoint
from chestxray.config import ModelConfig
from chestxray.model import build_model
from chestxray.release import prepare_release_bundle


def test_sqlite_append_and_read(tmp_path, monkeypatch):
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CXR_STORE", "sqlite")
    monkeypatch.setenv("CXR_SQLITE_PATH", str(tmp_path / "test.db"))

    import chestxray.sqlite_store as sqlite_store
    import chestxray.store as store

    importlib.reload(sqlite_store)
    sqlite_store.reset_connection()
    store = importlib.reload(store)

    store.append_jsonl("audit_log.jsonl", {"id": "a", "created_at": "2025-01-01T00:00:00Z"})
    store.append_jsonl("audit_log.jsonl", {"id": "b", "created_at": "2025-01-02T00:00:00Z"})

    recent = store.read_recent_jsonl("audit_log.jsonl", limit=10)
    assert [r["id"] for r in recent] == ["b", "a"]

    sqlite_store.reset_connection()


def test_models_registry_lists_checkpoint(tmp_path, monkeypatch):
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    ckpt = tmp_path / "best_model.pth"
    save_checkpoint(
        str(ckpt),
        model,
        ["NORMAL", "PNEUMONIA"],
        ModelConfig(num_classes=2),
        extra={"val_acc": 0.91, "val_balanced_acc": 0.90, "epoch": 10, "temperature": 1.2},
    )
    monkeypatch.setenv("CXR_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", str(ckpt))

    import chestxray.models_registry as registry

    registry = importlib.reload(registry)
    models = registry.list_models()
    assert len(models) == 1
    assert models[0]["class_names"] == ["NORMAL", "PNEUMONIA"]
    assert models[0]["val_acc"] == pytest.approx(0.91)
    assert models[0]["temperature"] == pytest.approx(1.2)


def test_prepare_release_bundle(tmp_path):
    ckpt = tmp_path / "best_model.pth"
    model = build_model(ModelConfig(num_classes=2), pretrained=False)
    save_checkpoint(str(ckpt), model, ["NORMAL", "PNEUMONIA"], ModelConfig(num_classes=2))

    out = tmp_path / "bundle"
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"accuracy": 0.93}', encoding="utf-8")

    manifest = prepare_release_bundle(str(out), checkpoint=str(ckpt), metrics=str(metrics))
    assert (out / "best_model.pth").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "checksums.sha256").is_file()
    assert "best_model.pth" in manifest["checksums"]
