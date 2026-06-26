"""Export clinician feedback for fine-tuning or active learning."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

from .store import read_all_jsonl

from .utils import get_logger

logger = get_logger(__name__)


def export_feedback(
    dest_dir: str,
    *,
    only_incorrect: bool = True,
    include_correct: bool = False,
) -> dict:
    """Write feedback rows to ``dest_dir/manifest.csv`` and optional image copies.

    Feedback entries may include an ``image_path`` field when recorded from the UI.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    images_dir = dest / "images"
    images_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    for entry in read_all_jsonl("feedback.jsonl"):
        is_correct = entry.get("is_correct")
        if only_incorrect and not include_correct and is_correct is True:
            continue
        if only_incorrect and not include_correct and is_correct is None:
            if entry.get("predicted_label") == entry.get("correct_label"):
                continue

        row = {
            "id": entry.get("id", ""),
            "predicted_label": entry.get("predicted_label", ""),
            "correct_label": entry.get("correct_label", ""),
            "comment": entry.get("comment") or "",
            "user": entry.get("user") or "",
            "ts": entry.get("ts") or "",
            "image_path": entry.get("image_path") or "",
            "local_image": "",
        }

        src_path = entry.get("image_path")
        if src_path and os.path.isfile(src_path):
            ext = Path(src_path).suffix or ".png"
            target = images_dir / f"{row['id']}{ext}"
            if not target.exists():
                shutil.copy2(src_path, target)
            row["local_image"] = str(target)

        rows.append(row)

    manifest = dest / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "id", "predicted_label", "correct_label", "comment", "user", "ts", "image_path", "local_image"
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "exported": len(rows),
        "dest_dir": str(dest),
        "manifest": str(manifest),
    }
    with open(dest / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Exported %d feedback rows -> %s", len(rows), manifest)
    return summary
