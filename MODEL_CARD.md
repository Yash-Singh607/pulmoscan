# PulmoScan Model Card

> Research and education only — **not a certified medical device**.

## Model overview

| Field | Value |
|-------|--------|
| **Name** | PulmoScan Pneumonia Classifier |
| **Version** | 1.0 (ResNet-50 backbone) |
| **Task** | Binary chest X-ray screening (NORMAL vs PNEUMONIA) |
| **Architecture** | ResNet-50 (ImageNet) + custom classification head |
| **Input** | Frontal chest radiograph, 224×224 RGB |
| **Output** | Class label + temperature-scaled probabilities + Grad-CAM |

## Calibration

After training, PulmoScan fits **temperature scaling** on the validation set and tunes a **pneumonia decision threshold** (F1-max on val). Both values are stored in the checkpoint and applied at inference.

## Intended use

- **Intended:** Research, education, and workflow prototyping for AI-assisted radiograph screening demos.
- **Out of scope:** Sole clinical diagnosis, treatment decisions, pediatric emergency triage without human review, non-chest radiographs.

## Training data

- **Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Paul Mooney, Kaggle, CC BY 4.0)
- **Classes:** NORMAL, PNEUMONIA (bacterial + viral combined)
- **Known limitations:** Pediatric-focused dataset; limited demographic metadata; shipped validation split is tiny (16 images) — PulmoScan re-splits validation from train.

## Performance (typical full training run)

| Metric | Approx. range |
|--------|----------------|
| Test accuracy | 90–94% |
| F1 | 0.93–0.95 |
| ROC-AUC | 0.95–0.99 |

Exact metrics are written to `outputs/metrics.json` after training.

## Safety features (software)

- OOD input guard (non-radiograph detection)
- Image quality assessment (blur / exposure)
- MC-Dropout uncertainty + abstention recommendation
- Clinical triage: `routine` | `review` | `reject`
- Radiologist review queue for flagged studies
- Audit log + feedback collection
- FHIR DiagnosticReport export (prototype)

## Ethical considerations

- Model may reflect dataset bias (age, geography, acquisition devices).
- False negatives (missed pneumonia) and false positives (unnecessary follow-up) both carry clinical risk.
- Always require qualified human review before any clinical action.

## Regulatory status

This software has **not** been submitted to FDA, CE, or other regulatory bodies.

## Maintainer

PulmoScan open-source project — [github.com/Yash-Singh607/pulmoscan](https://github.com/Yash-Singh607/pulmoscan)
