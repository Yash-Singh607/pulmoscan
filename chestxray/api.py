"""FastAPI inference service + web UI.

Features: standard image + DICOM upload, OOD input guard, Grad-CAM, MC-Dropout
uncertainty/abstention, PDF reports, audit log, feedback collection, optional
API-key auth, and in-memory rate limiting.

Run with::

    python -m chestxray.cli serve --checkpoint checkpoints/best_model.pth
    # or
    uvicorn chestxray.api:app
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from . import audit, report
from .imaging import ImageDecodeError, check_input, load_image_from_bytes
from .inference import Classifier
from .utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"
_rate_state: dict[str, list[float]] = {}

app = FastAPI(
    title="Chest X-Ray Pneumonia Classifier",
    version="2.0.0",
    description="Detect pneumonia from chest X-ray images. For research/education only.",
)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ----------------------------- models ---------------------------------------
class PredictionResponse(BaseModel):
    id: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    input_check: dict


class AnalyzeResponse(PredictionResponse):
    images: dict[str, str]
    uncertainty: dict | None = None


class MetadataResponse(BaseModel):
    class_names: list[str]
    checkpoint_path: str
    device: str


class FeedbackRequest(BaseModel):
    id: str
    predicted_label: str
    correct_label: str
    comment: str | None = None


# ----------------------------- config helpers -------------------------------
def _checkpoint_path() -> str:
    return os.getenv("CXR_CHECKPOINT_PATH", "checkpoints/best_model.pth")


def _mc_passes() -> int:
    return int(os.getenv("CXR_MC_PASSES", "10"))


@lru_cache(maxsize=1)
def get_classifier() -> Classifier:
    return Classifier(_checkpoint_path())


def _require_classifier() -> Classifier:
    try:
        return get_classifier()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model is loaded. Train one with `chestxray train` "
                f"so a checkpoint exists at '{_checkpoint_path()}'."
            ),
        ) from exc


# ----------------------------- security -------------------------------------
def rate_limit(request: Request) -> None:
    limit = int(os.getenv("CXR_RATE_LIMIT", "120"))  # requests/min/IP, 0 disables
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_state.setdefault(ip, [])
    cutoff = now - 60
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    bucket.append(now)


def require_auth(x_api_key: str | None = Header(default=None)) -> None:
    keys = {k for k in os.getenv("CXR_API_KEYS", "").split(",") if k}
    if keys and x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


SECURED = [Depends(rate_limit), Depends(require_auth)]


# ----------------------------- helpers --------------------------------------
def _decode(raw: bytes, filename: str) -> Image.Image:
    try:
        return load_image_from_bytes(raw, filename)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _audit(entry_id: str, filename: str, label: str, confidence: float, check, extra=None):
    record = {
        "id": entry_id,
        "ts": audit.utcnow(),
        "file": filename,
        "label": label,
        "confidence": round(confidence, 4),
        "is_xray_like": check.is_xray_like,
    }
    if extra:
        record.update(extra)
    try:
        audit.append_audit(record)
    except OSError:  # never fail a prediction because logging failed
        logger.warning("Failed to write audit log entry")


# ----------------------------- routes ---------------------------------------
@app.get("/", include_in_schema=False)
def index():
    index_file = WEB_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Web UI not found.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    path = Path(os.getenv("CXR_OUTPUT_DIR", "outputs")) / "metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No metrics available yet. Train first.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="Could not read metrics file.") from exc


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    clf = _require_classifier()
    return MetadataResponse(
        class_names=clf.class_names, checkpoint_path=_checkpoint_path(), device=str(clf.device)
    )


@app.get("/history")
def history(limit: int = 50) -> dict:
    return {"items": audit.recent_audit(min(max(limit, 1), 200))}


@app.post("/predict", response_model=PredictionResponse, dependencies=SECURED)
async def predict(file: UploadFile = File(...), tta: bool = False) -> PredictionResponse:
    image = _decode(await file.read(), file.filename or "")
    check = check_input(image)
    pred = _require_classifier().predict(image, tta=tta)
    entry_id = uuid.uuid4().hex
    _audit(entry_id, file.filename or "", pred.label, pred.confidence, check)
    return PredictionResponse(
        id=entry_id,
        label=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        input_check=check.to_dict(),
    )


@app.post("/predict/analyze", response_model=AnalyzeResponse, dependencies=SECURED)
async def analyze(file: UploadFile = File(...)) -> AnalyzeResponse:
    clf = _require_classifier()
    image = _decode(await file.read(), file.filename or "")
    check = check_input(image)

    pred, original, heatmap, overlay = clf.analyze(image)

    uncertainty = None
    passes = _mc_passes()
    if passes > 0:
        uncertainty = clf.estimate_uncertainty(image, passes=passes).to_dict()

    entry_id = uuid.uuid4().hex
    _audit(
        entry_id, file.filename or "", pred.label, pred.confidence, check,
        extra={"abstain": uncertainty["abstain"] if uncertainty else None},
    )
    return AnalyzeResponse(
        id=entry_id,
        label=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        input_check=check.to_dict(),
        uncertainty=uncertainty,
        images={
            "original": _to_data_url(original),
            "heatmap": _to_data_url(heatmap),
            "overlay": _to_data_url(overlay),
        },
    )


@app.post("/predict/report", dependencies=SECURED)
async def predict_report(
    file: UploadFile = File(...), threshold: float = Form(0.5)
) -> Response:
    clf = _require_classifier()
    raw = await file.read()
    image = _decode(raw, file.filename or "")
    check = check_input(image)
    pred, original, heatmap, overlay = clf.analyze(image)

    uncertainty = None
    if _mc_passes() > 0:
        uncertainty = clf.estimate_uncertainty(image, passes=_mc_passes()).to_dict()

    pneu = pred.probabilities.get("PNEUMONIA", 0.0)
    label = "PNEUMONIA" if pneu >= threshold else "NORMAL"
    confidence = pred.probabilities.get(label, pred.confidence)

    pdf = report.build_pdf(
        original=original,
        heatmap=heatmap,
        overlay=overlay,
        label=label,
        confidence=confidence,
        probabilities=pred.probabilities,
        threshold=threshold,
        meta={
            "file": file.filename or "-",
            "architecture": "ResNet-50",
            "device": str(clf.device),
            "uncertainty": uncertainty,
            "input_check": check.to_dict(),
        },
    )
    headers = {"Content-Disposition": 'attachment; filename="pulmoscan_report.pdf"'}
    return Response(content=pdf, media_type="application/pdf", headers=headers)


@app.post("/feedback", dependencies=SECURED)
def feedback(payload: FeedbackRequest) -> dict:
    entry = payload.model_dump()
    entry["ts"] = audit.utcnow()
    entry["is_correct"] = payload.predicted_label == payload.correct_label
    try:
        audit.append_feedback(entry)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Could not store feedback.") from exc
    return {"status": "recorded"}
