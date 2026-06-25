"""Audit log + feedback store tests."""

import importlib


def _fresh_audit(tmp_path, monkeypatch):
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path))
    import chestxray.audit as audit

    return importlib.reload(audit)


def test_append_and_read_audit(tmp_path, monkeypatch):
    audit = _fresh_audit(tmp_path, monkeypatch)
    audit.append_audit({"id": "a", "label": "NORMAL"})
    audit.append_audit({"id": "b", "label": "PNEUMONIA"})
    recent = audit.recent_audit(limit=10)
    assert [r["id"] for r in recent] == ["b", "a"]  # most recent first


def test_read_recent_respects_limit(tmp_path, monkeypatch):
    audit = _fresh_audit(tmp_path, monkeypatch)
    for i in range(5):
        audit.append_audit({"id": str(i)})
    recent = audit.recent_audit(limit=2)
    assert len(recent) == 2
    assert recent[0]["id"] == "4"


def test_feedback_store(tmp_path, monkeypatch):
    audit = _fresh_audit(tmp_path, monkeypatch)
    audit.append_feedback({"id": "x", "is_correct": False})
    items = audit.read_recent("feedback.jsonl", limit=10)
    assert items[0]["id"] == "x"


def test_read_missing_file_returns_empty(tmp_path, monkeypatch):
    audit = _fresh_audit(tmp_path, monkeypatch)
    assert audit.recent_audit() == []
