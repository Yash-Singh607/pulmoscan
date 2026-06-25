"""Append-only audit log and feedback store (JSONL)."""

from __future__ import annotations

from typing import Any

from .store import append_jsonl, read_recent_jsonl, utcnow

__all__ = ["append_audit", "append_feedback", "recent_audit", "utcnow"]


def append_audit(entry: dict[str, Any]) -> None:
    append_jsonl("audit_log.jsonl", entry)


def append_feedback(entry: dict[str, Any]) -> None:
    append_jsonl("feedback.jsonl", entry)


def recent_audit(limit: int = 50) -> list[dict[str, Any]]:
    return read_recent_jsonl("audit_log.jsonl", limit)
