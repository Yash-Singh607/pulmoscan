"""SQLite-backed persistence (durable alternative to JSONL files)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _db_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def init_db(path: Path) -> sqlite3.Connection:
    global _conn
    db = _db_path(path)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stream_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stream TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stream_id ON stream_entries(stream, id DESC)")
    conn.commit()
    _conn = conn
    return conn


def get_connection(path: Path) -> sqlite3.Connection:
    global _conn
    if _conn is None:
        return init_db(path)
    return _conn


def reset_connection() -> None:
    """Close the module-level connection (used in tests)."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def migrate_jsonl_files(output_dir: Path, db_path: Path) -> None:
    """Import existing JSONL files into SQLite on first use."""
    conn = get_connection(db_path)
    for name in ("audit_log.jsonl", "feedback.jsonl", "cases.jsonl", "review_queue.jsonl"):
        src = output_dir / name
        if not src.is_file():
            continue
        count = conn.execute("SELECT COUNT(*) FROM stream_entries WHERE stream=?", (name,)).fetchone()[0]
        if count:
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                created = payload.get("created_at") or payload.get("ts") or payload.get("resolved_at") or ""
                conn.execute(
                    "INSERT INTO stream_entries(stream, payload, created_at) VALUES (?,?,?)",
                    (name, json.dumps(payload, ensure_ascii=False), created),
                )
        conn.commit()


def append_stream(conn: sqlite3.Connection, stream: str, entry: dict[str, Any], created_at: str) -> None:
    with _lock:
        conn.execute(
            "INSERT INTO stream_entries(stream, payload, created_at) VALUES (?,?,?)",
            (stream, json.dumps(entry, ensure_ascii=False), created_at),
        )
        conn.commit()


def read_recent_stream(conn: sqlite3.Connection, stream: str, limit: int) -> list[dict[str, Any]]:
    with _lock:
        rows = conn.execute(
            "SELECT payload FROM stream_entries WHERE stream=? ORDER BY id DESC LIMIT ?",
            (stream, limit),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for (payload,) in rows:
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out


def read_all_stream(conn: sqlite3.Connection, stream: str) -> list[dict[str, Any]]:
    with _lock:
        rows = conn.execute(
            "SELECT payload FROM stream_entries WHERE stream=? ORDER BY id ASC",
            (stream,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for (payload,) in rows:
        try:
            out.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return out
