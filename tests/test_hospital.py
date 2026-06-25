"""Hospital workflow: cases, review queue, clinical triage."""

import importlib

import numpy as np
import pytest
from PIL import Image


def _fresh_cases(tmp_path, monkeypatch):
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path))
    import chestxray.cases as cases

    return importlib.reload(cases)


def _fresh_review(tmp_path, monkeypatch):
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path))
    import chestxray.review as review

    return importlib.reload(review)


def test_case_create_and_attach(tmp_path, monkeypatch):
    cases = _fresh_cases(tmp_path, monkeypatch)
    case = cases.create_case(patient_ref="MRN-001", notes="test")
    assert case["case_id"]
    ok = cases.attach_study(case["case_id"], {"study_id": "s1", "label": "NORMAL"})
    assert ok
    loaded = cases.get_case(case["case_id"])
    assert len(loaded["studies"]) == 1


def test_review_enqueue_and_resolve(tmp_path, monkeypatch):
    review = _fresh_review(tmp_path, monkeypatch)
    item = review.enqueue(
        study_id="s1", reason="low confidence", label="PNEUMONIA",
        confidence=0.55, triage="review",
    )
    pending = review.list_pending()
    assert len(pending) == 1
    resolved = review.resolve(item["review_id"], decision="agree", reviewer="dr_a")
    assert resolved["status"] == "resolved"
    assert review.list_pending() == []


def test_clinical_triage_low_confidence(checkpoint, rgb_image):
    from chestxray.clinical import compute_triage
    from chestxray.imaging import check_input
    from chestxray.quality import assess_quality

    check = check_input(rgb_image)
    quality = assess_quality(rgb_image)
    triage, reasons = compute_triage(check, quality, None, confidence=0.4)
    assert triage == "review"
    assert reasons
