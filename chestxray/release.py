"""Bundle trained artifacts for GitHub Release or Hugging Face upload."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)


def prepare_release_bundle(
    out_dir: str = "release_bundle",
    *,
    checkpoint: str = "checkpoints/best_model.pth",
    metrics: str = "outputs/metrics.json",
    model_card: str = "MODEL_CARD.md",
) -> dict:
    """Copy checkpoint + metrics into a release folder with checksums."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ckpt_src = Path(checkpoint)
    if not ckpt_src.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_src}. Train first: chestxray train --profile high-accuracy"
        )

    ckpt_dest = out / "best_model.pth"
    shutil.copy2(ckpt_src, ckpt_dest)

    metrics_src = Path(metrics)
    if metrics_src.is_file():
        shutil.copy2(metrics_src, out / "metrics.json")
    else:
        logger.warning("No metrics.json at %s — bundle will omit training metrics.", metrics_src)

    card_src = Path(model_card)
    if card_src.is_file():
        shutil.copy2(card_src, out / "MODEL_CARD.md")

    readme = out / "README.txt"
    readme.write_text(
        "PulmoScan model release bundle\n\n"
        "1. Copy best_model.pth to checkpoints/best_model.pth\n"
        "2. Copy metrics.json to outputs/metrics.json\n"
        "3. Run: chestxray serve\n\n"
        "Upload this folder as a GitHub Release asset or to Hugging Face.\n",
        encoding="utf-8",
    )

    checksums: dict[str, str] = {}
    for path in sorted(out.glob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums[path.name] = digest

    checksum_path = out / "checksums.sha256"
    lines = [f"{digest}  {name}\n" for name, digest in sorted(checksums.items())]
    checksum_path.write_text("".join(lines), encoding="utf-8")

    manifest = {
        "files": sorted(checksums.keys()) + ["checksums.sha256"],
        "checksums": checksums,
        "github_release_hint": (
            f"gh release create v1.0.0 {ckpt_dest} {out / 'metrics.json'} "
            f"--title 'PulmoScan v1.0.0 weights' --notes 'Pre-trained ResNet-50 checkpoint'"
        ),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Release bundle ready at %s (%d files)", out, len(manifest["files"]))
    logger.info("Suggested: %s", manifest["github_release_hint"])
    return manifest
