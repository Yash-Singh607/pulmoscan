"""Append-only audit log and feedback store (JSONL).

Every prediction is recorded for traceability; clinicians/users can submit
feedback (correct / incorrect) which is collected for future retraining.
Files live under the output directory:
    <output_dir>/audit_log.jsonl
    <output_dir>/feedback.jsonl
"""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def _output_dir() -> Path:
    return Path(os.getenv("CXR_OUTPUT_DIR", "outputs"))


def _path(name: str) -> Path:
    d = _output_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(name: str, entry: dict[str, Any]) -> None:
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        with open(_path(name), "a", encoding="utf-8") as f:
            f.write(line + "\n")


def append_audit(entry: dict[str, Any]) -> None:
    _append("audit_log.jsonl", entry)


def append_feedback(entry: dict[str, Any]) -> None:
    _append("feedback.jsonl", entry)


def read_recent(name: str, limit: int = 50) -> list[dict[str, Any]]:
    path = _output_dir() / name
    if not path.is_file():
        return []
    tail: deque[str] = deque(maxlen=limit)
    with _lock:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tail.append(line)
    out = []
    for line in tail:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()  # most recent first
    return out


def recent_audit(limit: int = 50) -> list[dict[str, Any]]:
    return read_recent("audit_log.jsonl", limit)
