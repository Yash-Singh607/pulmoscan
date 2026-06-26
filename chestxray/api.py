"""FastAPI inference service — hospital-grade workflow prototype.

Research/education only — not a certified medical device.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from . import audit, cases, fhir, jobs, report, review
from .auth import User, auth_enabled, authenticate, create_token, get_optional_user, require_role
from .clinical import run_study
from .imaging import ImageDecodeError, load_image_from_bytes
from .models_registry import clear_classifier_cache, get_classifier, list_models
from .observability import ObservabilityMiddleware, inc, render_prometheus
from .utils import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

WEB_DIR = Path(__file__).parent / "web"
_rate_state: dict[str, list[float]] = {}
MAX_UPLOAD = int(os.getenv("CXR_MAX_UPLOAD_MB", "15")) * 1024 * 1024

app = FastAPI(
    title="PulmoScan — Chest X-Ray AI",
    version="3.0.0",
    description="Hospital workflow prototype: cases, triage, review queue, FHIR export.",
)
app.add_middleware(ObservabilityMiddleware)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# ----------------------------- schemas --------------------------------------
class PredictionResponse(BaseModel):
    id: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    input_check: dict
    quality: dict
    triage: str
    triage_reasons: list[str] = Field(default_factory=list)
    case_id: str | None = None
    model_id: str | None = None


class AnalyzeResponse(PredictionResponse):
    images: dict[str, str]
    uncertainty: dict | None = None


class MetadataResponse(BaseModel):
    class_names: list[str]
    checkpoint_path: str
    device: str
    auth_enabled: bool


class FeedbackRequest(BaseModel):
    id: str
    predicted_label: str
    correct_label: str
    comment: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class CaseCreate(BaseModel):
    patient_ref: str = Field(..., description="De-identified external patient reference")
    study_date: str | None = None
    notes: str | None = None


class ReviewResolve(BaseModel):
    decision: str = Field(..., description="agree | disagree | needs_rescan")
    notes: str | None = None


# ----------------------------- helpers --------------------------------------
def _mc_passes() -> int:
    return int(os.getenv("CXR_MC_PASSES", "10"))


def _default_model_id() -> str | None:
    return os.getenv("CXR_MODEL_ID", "default")


def _require_classifier(model_id: str | None = None):
    mid = model_id or _default_model_id()
    try:
        return get_classifier(None if mid in (None, "default", "best") else mid), mid
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "No trained model loaded. Run `chestxray train` or set CXR_CHECKPOINT_PATH."
            ),
        ) from exc


def rate_limit(request: Request) -> None:
    limit = int(os.getenv("CXR_RATE_LIMIT", "120"))
    if limit <= 0:
        return
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _rate_state.setdefault(ip, [])
    cutoff = now - 60
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")
    bucket.append(now)


def require_auth(x_api_key: str | None = Header(default=None)) -> None:
    if auth_enabled():
        return
    keys = {k for k in os.getenv("CXR_API_KEYS", "").split(",") if k}
    if keys and x_api_key not in keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


SECURED = [Depends(rate_limit), Depends(require_auth)]


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_UPLOAD // (1024 * 1024)} MB limit.",
        )
    return raw, file.filename or ""


def _decode(raw: bytes, filename: str) -> Image.Image:
    try:
        return load_image_from_bytes(raw, filename)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def _clinical_response(result, model_id: str | None) -> dict:
    inc("predictions_total")
    if result.triage in ("review", "reject"):
        inc("review_enqueued_total")
    return {
        "id": result.study_id,
        "label": result.prediction.label,
        "confidence": result.prediction.confidence,
        "probabilities": result.prediction.probabilities,
        "input_check": result.input_check.to_dict(),
        "quality": result.quality.to_dict(),
        "triage": result.triage,
        "triage_reasons": result.triage_reasons,
        "case_id": result.case_id,
        "model_id": model_id,
    }


# ----------------------------- routes ---------------------------------------
@app.get("/", include_in_schema=False)
def index():
    index_file = WEB_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail="Web UI not found.")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "auth_enabled": auth_enabled()}


@app.get("/ready")
def ready() -> dict:
    try:
        clf, _ = _require_classifier()
        return {"status": "ready", "device": str(clf.device), "classes": clf.class_names}
    except HTTPException:
        return {"status": "not_ready", "reason": "checkpoint missing"}


@app.get("/metrics/prometheus")
def metrics_prometheus() -> PlainTextResponse:
    return PlainTextResponse(render_prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/metrics")
def metrics() -> dict:
    path = Path(os.getenv("CXR_OUTPUT_DIR", "outputs")) / "metrics.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No training metrics yet.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/auth/login")
def login(body: LoginRequest) -> dict:
    if not auth_enabled():
        return {"token": None, "role": "admin", "auth_enabled": False, "message": "Auth disabled"}
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": create_token(user), "role": user.role, "auth_enabled": True}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    clf, _ = _require_classifier()
    return MetadataResponse(
        class_names=clf.class_names,
        checkpoint_path=os.getenv("CXR_CHECKPOINT_PATH", "checkpoints/best_model.pth"),
        device=str(clf.device),
        auth_enabled=auth_enabled(),
    )


@app.get("/models")
def models() -> dict:
    return {"models": list_models(), "default": _default_model_id()}


@app.get("/demo/sample")
def demo_sample(label: str = "PNEUMONIA"):
    """Return a sample test X-ray for demo mode (requires downloaded dataset)."""
    data_dir = Path(os.getenv("CXR_DATA_DIR", "data/chest_xray"))
    sub = "PNEUMONIA" if label.upper().startswith("P") else "NORMAL"
    folder = data_dir / "test" / sub
    if not folder.is_dir():
        raise HTTPException(
            status_code=404,
            detail="Demo samples need the dataset. Run: chestxray setup-data",
        )
    for pattern in ("*.jpeg", "*.jpg", "*.png"):
        matches = sorted(folder.glob(pattern))
        if matches:
            return FileResponse(str(matches[0]), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail=f"No sample images in {folder}")


@app.get("/history")
def history(limit: int = 50) -> dict:
    return {"items": audit.recent_audit(min(max(limit, 1), 200))}


@app.post("/cases")
def create_case(
    body: CaseCreate,
    user: User = Depends(require_role("clinician", "admin")),
) -> dict:
    entry = cases.create_case(
        patient_ref=body.patient_ref,
        study_date=body.study_date,
        notes=body.notes,
        created_by=user.username,
    )
    return entry


@app.get("/cases")
def list_all_cases(limit: int = 50) -> dict:
    return {"items": cases.list_cases(min(max(limit, 1), 200))}


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict:
    case = cases.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@app.get("/review/queue")
def review_queue(limit: int = 50) -> dict:
    return {"items": review.list_pending(min(max(limit, 1), 200))}


@app.post("/review/{review_id}/resolve")
def resolve_review(
    review_id: str,
    body: ReviewResolve,
    user: User = Depends(require_role("clinician", "admin")),
) -> dict:
    item = review.resolve(
        review_id, decision=body.decision, reviewer=user.username, notes=body.notes
    )
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found or already resolved")
    return item


@app.get("/studies/{study_id}/fhir")
def study_fhir(study_id: str, patient_ref: str = "anonymous") -> dict:
    for item in audit.recent_audit(limit=500):
        if item.get("id") == study_id:
            return fhir.build_diagnostic_report(
                study_id=study_id,
                patient_ref=patient_ref,
                label=item.get("label", "UNKNOWN"),
                confidence=float(item.get("confidence", 0)),
                probabilities=item.get("probabilities", {}),
                triage=item.get("triage", "routine"),
                quality=item.get("quality", {}),
            )
    raise HTTPException(status_code=404, detail="Study not found in audit log")


@app.post("/predict", response_model=PredictionResponse, dependencies=SECURED)
async def predict(
    file: UploadFile = File(...),
    tta: bool = False,
    case_id: str | None = Form(None),
    model_id: str | None = Form(None),
    user: User = Depends(require_role("clinician", "admin", "viewer")),
) -> PredictionResponse:
    raw, filename = await _read_upload(file)
    image = _decode(raw, filename)
    clf, mid = _require_classifier(model_id)
    result = run_study(
        clf, image, filename, tta=tta, with_uncertainty=False,
        case_id=case_id, model_id=mid, user=user.username,
    )
    return PredictionResponse(**_clinical_response(result, mid))


@app.post("/predict/analyze", response_model=AnalyzeResponse, dependencies=SECURED)
async def analyze(
    file: UploadFile = File(...),
    case_id: str | None = Form(None),
    model_id: str | None = Form(None),
    user: User = Depends(require_role("clinician", "admin")),
) -> AnalyzeResponse:
    raw, filename = await _read_upload(file)
    image = _decode(raw, filename)
    clf, mid = _require_classifier(model_id)
    result = run_study(
        clf, image, filename, case_id=case_id, model_id=mid, user=user.username,
    )
    pred, original, heatmap, overlay = clf.analyze(image)
    payload = _clinical_response(result, mid)
    payload["uncertainty"] = result.uncertainty.to_dict() if result.uncertainty else None
    payload["images"] = {
        "original": _to_data_url(original),
        "heatmap": _to_data_url(heatmap),
        "overlay": _to_data_url(overlay),
    }
    return AnalyzeResponse(**payload)


@app.post("/jobs/analyze", dependencies=SECURED)
async def job_analyze(
    file: UploadFile = File(...),
    case_id: str | None = Form(None),
    model_id: str | None = Form(None),
    user: User = Depends(require_role("clinician", "admin")),
) -> dict:
    raw, filename = await _read_upload(file)
    image = _decode(raw, filename)
    clf, mid = _require_classifier(model_id)
    username = user.username

    def _work():
        result = run_study(
            clf, image, filename, case_id=case_id, model_id=mid, user=username,
        )
        pred, original, heatmap, overlay = clf.analyze(image)
        out = _clinical_response(result, mid)
        out["uncertainty"] = result.uncertainty.to_dict() if result.uncertainty else None
        out["images"] = {
            "original": _to_data_url(original),
            "heatmap": _to_data_url(heatmap),
            "overlay": _to_data_url(overlay),
        }
        return out

    job_id = jobs.submit(_work)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs.public_view(job)


@app.post("/predict/report", dependencies=SECURED)
async def predict_report(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    case_id: str | None = Form(None),
    user: User = Depends(require_role("clinician", "admin")),
) -> Response:
    clf, mid = _require_classifier()
    raw, filename = await _read_upload(file)
    image = _decode(raw, filename)
    result = run_study(clf, image, filename, case_id=case_id, model_id=mid, user=user.username)
    _, original, heatmap, overlay = clf.analyze(image)

    pneu = result.prediction.probabilities.get("PNEUMONIA", 0.0)
    label = "PNEUMONIA" if pneu >= threshold else "NORMAL"
    confidence = result.prediction.probabilities.get(label, result.prediction.confidence)

    pdf = report.build_pdf(
        original=original,
        heatmap=heatmap,
        overlay=overlay,
        label=label,
        confidence=confidence,
        probabilities=result.prediction.probabilities,
        threshold=threshold,
        meta={
            "file": filename,
            "architecture": "ResNet-50",
            "device": str(clf.device),
            "uncertainty": result.uncertainty.to_dict() if result.uncertainty else None,
            "input_check": result.input_check.to_dict(),
            "quality": result.quality.to_dict(),
            "triage": result.triage,
            "case_id": case_id,
        },
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="pulmoscan_report.pdf"'},
    )


@app.post("/feedback", dependencies=SECURED)
def feedback(payload: FeedbackRequest, user: User = Depends(get_optional_user)) -> dict:
    entry = payload.model_dump()
    entry["ts"] = audit.utcnow()
    entry["is_correct"] = payload.predicted_label == payload.correct_label
    entry["user"] = user.username if user else None
    audit.append_feedback(entry)
    return {"status": "recorded"}


# Re-export for tests / backward compatibility
__all__ = ["app", "get_classifier", "clear_classifier_cache"]
