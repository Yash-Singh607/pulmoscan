"""FHIR R4 DiagnosticReport export (research/education stub).

Produces a valid JSON bundle suitable for hospital integration prototyping.
Not certified for production EMR use.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_diagnostic_report(
    *,
    study_id: str,
    patient_ref: str,
    label: str,
    confidence: float,
    probabilities: dict[str, float],
    triage: str,
    quality: dict,
    performer: str = "PulmoScan AI",
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conclusion = (
        f"AI screening result: {label} ({confidence * 100:.1f}% confidence). "
        f"Triage: {triage.upper()}. "
        "For research/education only — not for sole clinical use."
    )
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": now,
        "entry": [
            {
                "fullUrl": f"urn:uuid:{study_id}",
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": study_id,
                    "status": "preliminary",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                                    "code": "RAD",
                                    "display": "Radiology",
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": "36643-5",
                                "display": "Chest X-ray pneumonia assessment",
                            }
                        ],
                        "text": "Chest X-ray pneumonia screening",
                    },
                    "subject": {"reference": f"Patient/{patient_ref}"},
                    "effectiveDateTime": now,
                    "issued": now,
                    "performer": [{"display": performer}],
                    "conclusion": conclusion,
                    "conclusionCode": [
                        {
                            "coding": [{"code": label, "display": label}],
                            "text": label,
                        }
                    ],
                    "extension": [
                        {
                            "url": "https://pulmoscan.local/fhir/probabilities",
                            "valueString": str(probabilities),
                        },
                        {
                            "url": "https://pulmoscan.local/fhir/quality",
                            "valueString": str(quality),
                        },
                    ],
                },
            }
        ],
    }
