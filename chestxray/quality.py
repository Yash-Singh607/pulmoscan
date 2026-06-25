"""Pre-inference image quality assessment for radiograph uploads."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityCheck:
    score: float
    blur_variance: float
    mean_intensity: float
    acceptable: bool
    warnings: list[str] = field(default_factory=list)
    triage: str = "routine"  # routine | review | reject

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 3),
            "blur_variance": round(self.blur_variance, 2),
            "mean_intensity": round(self.mean_intensity, 2),
            "acceptable": self.acceptable,
            "warnings": self.warnings,
            "triage": self.triage,
        }


def assess_quality(
    image: Image.Image,
    *,
    blur_min: float = 80.0,
    dark_max: float = 35.0,
    bright_min: float = 220.0,
) -> QualityCheck:
    """Score blur and exposure; flag studies that may need rescan or manual review."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_int = float(gray.mean())

    warnings: list[str] = []
    score = 1.0

    if blur_var < blur_min:
        warnings.append("Image appears blurry — consider rescanning.")
        score -= 0.35
    if mean_int < dark_max:
        warnings.append("Image underexposed (too dark).")
        score -= 0.25
    if mean_int > bright_min:
        warnings.append("Image overexposed (too bright).")
        score -= 0.25

    score = max(0.0, min(1.0, score))
    acceptable = score >= 0.5 and blur_var >= blur_min * 0.5

    if not acceptable or len(warnings) >= 2:
        triage = "reject"
    elif warnings:
        triage = "review"
    else:
        triage = "routine"

    return QualityCheck(
        score=score,
        blur_variance=blur_var,
        mean_intensity=mean_int,
        acceptable=acceptable,
        warnings=warnings,
        triage=triage,
    )
