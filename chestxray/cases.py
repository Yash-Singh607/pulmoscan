"""De-identified case / study workflow store."""

from __future__ import annotations

import uuid
from typing import Any

from .store import append_jsonl, read_all_jsonl, read_recent_jsonl, utcnow


def create_case(
    *,
    patient_ref: str,
    study_date: str | None = None,
    notes: str | None = None,
    created_by: str | None = None,
) -> dict[str, Any]:
    case_id = uuid.uuid4().hex[:12]
    entry = {
        "case_id": case_id,
        "patient_ref": patient_ref.strip() or "anonymous",
        "study_date": study_date or utcnow()[:10],
        "notes": notes or "",
        "created_by": created_by,
        "created_at": utcnow(),
        "studies": [],
        "status": "open",
    }
    append_jsonl("cases.jsonl", entry)
    return entry


def get_case(case_id: str) -> dict[str, Any] | None:
    for case in reversed(read_all_jsonl("cases.jsonl")):
        if case.get("case_id") == case_id:
            return case
    return None


def attach_study(case_id: str, study: dict[str, Any]) -> bool:
    case = get_case(case_id)
    if not case:
        return False
    updated = dict(case)
    studies = list(updated.get("studies", []))
    studies.append(study)
    updated["studies"] = studies
    updated["updated_at"] = utcnow()
    append_jsonl("cases.jsonl", updated)
    return True


def list_cases(limit: int = 50) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for case in read_all_jsonl("cases.jsonl"):
        cid = case.get("case_id")
        if cid:
            by_id[cid] = case
    items = sorted(by_id.values(), key=lambda c: c.get("updated_at", c.get("created_at", "")), reverse=True)
    return items[:limit]
