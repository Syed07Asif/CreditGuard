"""`GET /health` (liveness), `GET /health/ready` (readiness: database
reachable AND model loaded), `GET /metrics`.

`/health` and `/health/ready` are unauthenticated by design (an
orchestrator's liveness/readiness probe doesn't carry an API key);
`/metrics` is authenticated like every other endpoint, since it exposes
internal operational counts.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import text

from creditguard.api.dependencies import enforce_rate_limit
from creditguard.api.middleware import metrics_store
from creditguard.api.schemas import HealthResponse, MetricsResponse, ReadinessResponse
from creditguard.db.engine import get_engine
from creditguard.scoring import engine

logger = logging.getLogger("creditguard.api")

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Always 200 if the process is alive and able to handle a request --
    doesn't touch the database or the model.
    """
    return HealthResponse(status="ok")


def _database_reachable() -> bool:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("readiness_db_check_failed", exc_info=True)
        return False


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response) -> ReadinessResponse:
    """Ready only when the database is reachable AND the active model
    (feature pipeline + SHAP explainer) is loaded. A non-200 status (503)
    is what tells an orchestrator not to route traffic yet -- the JSON body
    alone isn't sufficient for standard readiness-probe semantics.
    """
    database_ok = _database_reachable()
    model_ok = engine.is_model_loaded()
    if not (database_ok and model_ok):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if (database_ok and model_ok) else "not_ready",
        database=database_ok,
        model_loaded=model_ok,
    )


@router.get("/metrics", response_model=MetricsResponse)
def metrics(_api_key: str = Depends(enforce_rate_limit)) -> MetricsResponse:
    """Request count, error count, and p50/p95/p99 latency over the most
    recent requests (see `creditguard.api.middleware.MetricsStore`).
    """
    return MetricsResponse(**metrics_store.snapshot())
