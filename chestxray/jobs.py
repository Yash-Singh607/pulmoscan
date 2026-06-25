"""In-memory async analysis jobs (hospital batch workflow prototype)."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

_lock = threading.Lock()
_jobs: dict[str, "Job"] = {}


@dataclass
class Job:
    id: str
    status: str  # queued | running | done | failed
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=lambda: __import__("time").time())


def submit(fn: Callable[[], dict[str, Any]]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = Job(id=job_id, status="queued")
    with _lock:
        _jobs[job_id] = job

    def _run():
        with _lock:
            job.status = "running"
        try:
            job.result = fn()
            job.status = "done"
        except Exception as exc:  # pragma: no cover - worker safety
            job.error = str(exc)
            job.status = "failed"

    threading.Thread(target=_run, daemon=True).start()
    return job_id


def get(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def public_view(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "status": job.status,
        "result": job.result,
        "error": job.error,
    }
