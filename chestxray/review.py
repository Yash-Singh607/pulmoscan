"""Radiologist review queue for uncertain or flagged studies."""

from __future__ import annotations

import uuid
from typing import Any

from .store import append_jsonl, read_all_jsonl, utcnow


def _latest_by_review_id() -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in read_all_jsonl("review_queue.jsonl"):
        rid = item.get("review_id")
        if rid:
            by_id[rid] = item
    return by_id


def enqueue(
    *,
    study_id: str,
    reason: str,
    label: str,
    confidence: float,
    triage: str,
    case_id: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "review_id": uuid.uuid4().hex[:12],
        "study_id": study_id,
        "case_id": case_id,
        "reason": reason,
        "label": label,
        "confidence": round(confidence, 4),
        "triage": triage,
        "status": "pending",
        "created_at": utcnow(),
        "meta": meta or {},
    }
    append_jsonl("review_queue.jsonl", entry)
    return entry


def list_pending(limit: int = 50) -> list[dict[str, Any]]:
    pending = [i for i in _latest_by_review_id().values() if i.get("status") == "pending"]
    pending.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return pending[:limit]


def resolve(
    review_id: str,
    *,
    decision: str,
    reviewer: str,
    notes: str | None = None,
) -> dict[str, Any] | None:
    item = _latest_by_review_id().get(review_id)
    if not item or item.get("status") != "pending":
        return None
    updated = dict(item)
    updated["status"] = "resolved"
    updated["decision"] = decision
    updated["reviewer"] = reviewer
    updated["notes"] = notes or ""
    updated["resolved_at"] = utcnow()
    append_jsonl("review_queue.jsonl", updated)
    return updated
