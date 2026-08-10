"""Request-ID assignment, structured JSON access logging, latency
measurement, and the in-process metrics store `GET /metrics` reads.

Never logs the request body -- only the request_id, method, path, status
code and latency, per the Phase 8 brief's "never log full applicant
payloads" rule. Derived aggregates (credit_score, risk_category, etc.) are
logged by the route handlers themselves if/when useful, not here.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"

logger = logging.getLogger("creditguard.api")

_LOG_RECORD_EXTRA_KEYS = ("request_id", "method", "path", "status_code", "latency_ms")


class JsonLogFormatter(logging.Formatter):
    """Renders each log record as one JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key in _LOG_RECORD_EXTRA_KEYS:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload)


def configure_logging() -> None:
    """Attach a JSON-formatted stream handler to the API's logger, once
    (idempotent -- safe to call from `main.py`'s lifespan even if the
    process reloads).
    """
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = rank - lower
    return float(
        sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


class MetricsStore:
    """Bounded in-memory request/error counters and latency samples --
    backs `GET /metrics`. Bounded (not an ever-growing list) so a
    long-running process doesn't leak memory; the most recent
    `max_samples` requests are representative enough for p50/p95/p99
    without keeping every sample forever.
    """

    def __init__(self, max_samples: int = 2000) -> None:
        self._lock = Lock()
        self._latencies: list[float] = []
        self._max_samples = max_samples
        self._request_count = 0
        self._error_count = 0

    def record(self, latency_ms: float, status_code: int) -> None:
        with self._lock:
            self._request_count += 1
            if status_code >= 400:
                self._error_count += 1
            self._latencies.append(latency_ms)
            if len(self._latencies) > self._max_samples:
                self._latencies = self._latencies[-self._max_samples :]

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            latencies = sorted(self._latencies)
            request_count = self._request_count
            error_count = self._error_count
        return {
            "request_count": request_count,
            "error_count": error_count,
            "latency_p50_ms": round(_percentile(latencies, 50), 2),
            "latency_p95_ms": round(_percentile(latencies, 95), 2),
            "latency_p99_ms": round(_percentile(latencies, 99), 2),
        }

    def reset(self) -> None:
        """Test-only hook: clear all recorded metrics."""
        with self._lock:
            self._latencies.clear()
            self._request_count = 0
            self._error_count = 0


metrics_store = MetricsStore()


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request_id (reusing an inbound `X-Request-ID` if the
    caller supplied one, otherwise a fresh UUID4), attaches it to
    `request.state` (so route handlers and exception handlers can read it)
    and the response header, measures latency, records it into
    `metrics_store`, and logs one structured line per request.
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # Only reachable for a failure FastAPI's own exception handlers
            # didn't convert into a Response (e.g. a bug in a handler
            # itself) -- normal application errors are handled by
            # creditguard.api.errors and arrive here as an ordinary
            # Response with a 4xx/5xx status, in the branch below.
            latency_ms = (time.perf_counter() - start) * 1000
            metrics_store.record(latency_ms, 500)
            logger.error(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        metrics_store.record(latency_ms, response.status_code)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response
