"""End-to-end inference and API tests using a randomly-initialised model."""

import io

import pytest
from PIL import Image

from chestxray.inference import Classifier


def test_classifier_predict(checkpoint, rgb_image):
    clf = Classifier(checkpoint, device="cpu")
    pred = clf.predict(rgb_image)
    assert pred.label in ("NORMAL", "PNEUMONIA")
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_classifier_analyze_returns_images(checkpoint, rgb_image):
    clf = Classifier(checkpoint, device="cpu")
    pred, original, heatmap, overlay = clf.analyze(rgb_image)
    assert pred.label in ("NORMAL", "PNEUMONIA")
    assert original.size == (224, 224)
    assert heatmap.size == (224, 224)
    assert overlay.size == (224, 224)


def test_classifier_missing_checkpoint():
    with pytest.raises(FileNotFoundError):
        Classifier("does/not/exist.pth", device="cpu")


def test_api_predict(checkpoint, rgb_image, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)

    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)

    assert client.get("/health").json()["status"] == "ok"

    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    buf.seek(0)
    resp = client.post("/predict", files={"file": ("xray.png", buf, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in ("NORMAL", "PNEUMONIA")
    assert set(body["probabilities"]) == {"NORMAL", "PNEUMONIA"}
    assert "quality" in body and "triage" in body


def test_api_metrics_missing_returns_404(checkpoint, tmp_path, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path / "empty_outputs"))

    from chestxray import api

    client = fastapi_testclient.TestClient(api.app)
    assert client.get("/metrics").status_code == 404


def test_api_metrics_returns_json(checkpoint, tmp_path, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    import json

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    payload = {"accuracy": 0.93, "f1": 0.94, "precision": 0.95, "recall": 0.92, "auc": 0.98}
    (out_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(out_dir))

    from chestxray import api

    client = fastapi_testclient.TestClient(api.app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["accuracy"] == 0.93


def test_api_serves_web_ui(checkpoint, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)

    from chestxray import api

    client = fastapi_testclient.TestClient(api.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_api_analyze_returns_images(checkpoint, rgb_image, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)

    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)

    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    buf.seek(0)
    resp = client.post("/predict/analyze", files={"file": ("xray.png", buf, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["images"]) == {"original", "heatmap", "overlay"}
    assert body["images"]["overlay"].startswith("data:image/png;base64,")


def test_classifier_predict_tta(checkpoint, rgb_image):
    clf = Classifier(checkpoint, device="cpu")
    pred = clf.predict(rgb_image, tta=True)
    assert pred.label in ("NORMAL", "PNEUMONIA")
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_classifier_uncertainty(checkpoint, rgb_image):
    clf = Classifier(checkpoint, device="cpu")
    u = clf.estimate_uncertainty(rgb_image, passes=3)
    assert u.entropy >= 0.0
    assert set(u.std) == {"NORMAL", "PNEUMONIA"}
    assert isinstance(u.abstain, bool)


def _png(rgb_image):
    buf = io.BytesIO()
    rgb_image.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_api_predict_includes_input_check(checkpoint, rgb_image, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)
    resp = client.post("/predict", files={"file": ("xray.png", _png(rgb_image), "image/png")})
    body = resp.json()
    assert "input_check" in body and "is_xray_like" in body["input_check"]
    assert "id" in body


def test_api_history_and_feedback(checkpoint, rgb_image, tmp_path, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path / "out"))
    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)

    pred = client.post("/predict", files={"file": ("xray.png", _png(rgb_image), "image/png")})
    rec_id = pred.json()["id"]

    hist = client.get("/history")
    assert hist.status_code == 200
    assert any(item["id"] == rec_id for item in hist.json()["items"])

    fb = client.post(
        "/feedback",
        json={"id": rec_id, "predicted_label": "NORMAL", "correct_label": "PNEUMONIA"},
    )
    assert fb.status_code == 200
    assert fb.json()["status"] == "recorded"


def test_api_report_returns_pdf(checkpoint, rgb_image, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_MC_PASSES", "0")
    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)
    resp = client.post(
        "/predict/report",
        files={"file": ("xray.png", _png(rgb_image), "image/png")},
        data={"threshold": "0.5"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content[:4] == b"%PDF"


def test_api_auth_enforced_when_keys_set(checkpoint, rgb_image, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_API_KEYS", "secret123")
    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)

    no_key = client.post("/predict", files={"file": ("x.png", _png(rgb_image), "image/png")})
    assert no_key.status_code == 401

    ok = client.post(
        "/predict",
        files={"file": ("x.png", _png(rgb_image), "image/png")},
        headers={"X-API-Key": "secret123"},
    )
    assert ok.status_code == 200


def test_api_ready_and_hospital_routes(checkpoint, rgb_image, tmp_path, monkeypatch):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    monkeypatch.setenv("CXR_CHECKPOINT_PATH", checkpoint)
    monkeypatch.setenv("CXR_OUTPUT_DIR", str(tmp_path / "out"))
    from chestxray import api

    api.clear_classifier_cache()
    client = fastapi_testclient.TestClient(api.app)

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    prom = client.get("/metrics/prometheus")
    assert prom.status_code == 200
    assert "cxr_requests_total" in prom.text

    case = client.post("/cases", json={"patient_ref": "MRN-001"})
    assert case.status_code == 200
    case_id = case.json()["case_id"]

    pred = client.post(
        "/predict/analyze",
        files={"file": ("xray.png", _png(rgb_image), "image/png")},
        data={"case_id": case_id},
    )
    assert pred.status_code == 200
    study_id = pred.json()["id"]

    fhir = client.get(f"/studies/{study_id}/fhir?patient_ref=MRN-001")
    assert fhir.status_code == 200
    assert fhir.json()["resourceType"] == "Bundle"
