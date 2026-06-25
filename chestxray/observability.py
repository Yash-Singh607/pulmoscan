"""Prometheus metrics and request observability."""

from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Lightweight in-process counters (no prometheus_client dep required).
_metrics = {
    "requests_total": 0,
    "errors_total": 0,
    "predictions_total": 0,
    "review_enqueued_total": 0,
    "latency_ms_sum": 0.0,
    "latency_ms_count": 0,
}


def inc(name: str, amount: int = 1) -> None:
    _metrics[name] = _metrics.get(name, 0) + amount


def observe_latency(ms: float) -> None:
    _metrics["latency_ms_sum"] += ms
    _metrics["latency_ms_count"] += 1


def render_prometheus() -> str:
    avg = 0.0
    if _metrics["latency_ms_count"]:
        avg = _metrics["latency_ms_sum"] / _metrics["latency_ms_count"]
    lines = [
        "# HELP cxr_requests_total Total HTTP requests",
        "# TYPE cxr_requests_total counter",
        f"cxr_requests_total {_metrics['requests_total']}",
        "# HELP cxr_errors_total Total HTTP 5xx responses",
        "# TYPE cxr_errors_total counter",
        f"cxr_errors_total {_metrics['errors_total']}",
        "# HELP cxr_predictions_total Total predictions served",
        "# TYPE cxr_predictions_total counter",
        f"cxr_predictions_total {_metrics['predictions_total']}",
        "# HELP cxr_review_enqueued_total Studies sent to review queue",
        "# TYPE cxr_review_enqueued_total counter",
        f"cxr_review_enqueued_total {_metrics['review_enqueued_total']}",
        "# HELP cxr_request_latency_ms_avg Average request latency",
        "# TYPE cxr_request_latency_ms_avg gauge",
        f"cxr_request_latency_ms_avg {avg:.2f}",
    ]
    return "\n".join(lines) + "\n"


class ObservabilityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:16])
        request.state.request_id = request_id
        t0 = time.perf_counter()
        inc("requests_total")
        try:
            response = await call_next(request)
        except Exception:
            inc("errors_total")
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000
        observe_latency(elapsed_ms)
        if response.status_code >= 500:
            inc("errors_total")
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response
