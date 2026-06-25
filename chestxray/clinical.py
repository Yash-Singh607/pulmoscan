"""Clinical orchestration: prediction + quality + triage + audit + review queue."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

from PIL import Image

from . import cases, review
from .audit import append_audit
from .imaging import InputCheck, check_input
from .inference import Classifier, Prediction, Uncertainty
from .quality import QualityCheck, assess_quality
from .store import utcnow


@dataclass
class ClinicalResult:
    study_id: str
    prediction: Prediction
    input_check: InputCheck
    quality: QualityCheck
    triage: str
    triage_reasons: list[str]
    uncertainty: Uncertainty | None
    case_id: str | None = None

    def to_audit_dict(self, filename: str, model_id: str | None, user: str | None) -> dict[str, Any]:
        return {
            "id": self.study_id,
            "ts": utcnow(),
            "file": filename,
            "label": self.prediction.label,
            "confidence": round(self.prediction.confidence, 4),
            "probabilities": self.prediction.probabilities,
            "is_xray_like": self.input_check.is_xray_like,
            "quality": self.quality.to_dict(),
            "triage": self.triage,
            "triage_reasons": self.triage_reasons,
            "case_id": self.case_id,
            "model_id": model_id or "default",
            "user": user,
            "abstain": self.uncertainty.abstain if self.uncertainty else False,
        }


def _mc_passes() -> int:
    return int(os.getenv("CXR_MC_PASSES", "10"))


def compute_triage(
    input_check: InputCheck,
    quality: QualityCheck,
    uncertainty: Uncertainty | None,
    confidence: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    level = "routine"

    if not input_check.is_xray_like:
        reasons.append(input_check.reason or "Input may not be a chest X-ray")
        level = "review"

    if quality.triage == "reject":
        reasons.extend(quality.warnings)
        return "reject", reasons
    if quality.triage == "review":
        reasons.extend(quality.warnings)
        level = "review"

    if uncertainty and uncertainty.abstain:
        reasons.append("Model uncertainty — recommend radiologist review")
        level = "review"

    if confidence < 0.65:
        reasons.append("Low model confidence")
        level = "review"

    return level, reasons


def run_study(
    clf: Classifier,
    image: Image.Image,
    filename: str,
    *,
    tta: bool = False,
    with_uncertainty: bool = True,
    case_id: str | None = None,
    model_id: str | None = None,
    user: str | None = None,
) -> ClinicalResult:
    input_check = check_input(image)
    quality = assess_quality(image)
    pred = clf.predict(image, tta=tta)

    uncertainty = None
    if with_uncertainty and _mc_passes() > 0:
        uncertainty = clf.estimate_uncertainty(image, passes=_mc_passes())

    triage, reasons = compute_triage(input_check, quality, uncertainty, pred.confidence)
    study_id = uuid.uuid4().hex

    result = ClinicalResult(
        study_id=study_id,
        prediction=pred,
        input_check=input_check,
        quality=quality,
        triage=triage,
        triage_reasons=reasons,
        uncertainty=uncertainty,
        case_id=case_id,
    )

    audit_entry = result.to_audit_dict(filename, model_id, user)
    try:
        append_audit(audit_entry)
    except OSError:
        pass

    if case_id:
        cases.attach_study(
            case_id,
            {
                "study_id": study_id,
                "label": pred.label,
                "confidence": pred.confidence,
                "triage": triage,
                "ts": utcnow(),
            },
        )

    if triage in ("review", "reject") or (uncertainty and uncertainty.abstain):
        review.enqueue(
            study_id=study_id,
            reason="; ".join(reasons) or triage,
            label=pred.label,
            confidence=pred.confidence,
            triage=triage,
            case_id=case_id,
            meta={"filename": filename, "abstain": uncertainty.abstain if uncertainty else False},
        )

    return result
