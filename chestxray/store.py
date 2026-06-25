"""Shared JSONL persistence helpers for audit, cases, and review queues."""

from __future__ import annotations

import json
import os
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_lock = threading.Lock()


def output_dir() -> Path:
    return Path(os.getenv("CXR_OUTPUT_DIR", "outputs"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(name: str, entry: dict[str, Any]) -> None:
    d = output_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def read_recent_jsonl(name: str, limit: int = 50) -> list[dict[str, Any]]:
    path = output_dir() / name
    if not path.is_file():
        return []
    tail: deque[str] = deque(maxlen=limit)
    with _lock:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tail.append(line)
    out: list[dict[str, Any]] = []
    for line in tail:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    out.reverse()
    return out


def read_all_jsonl(name: str) -> list[dict[str, Any]]:
    path = output_dir() / name
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with _lock:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
