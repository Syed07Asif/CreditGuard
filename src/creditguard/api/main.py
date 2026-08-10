"""FastAPI application factory: lifespan (warm the active model once),
CORS, request-context middleware, exception handlers, and router
registration.

Run with:

    uvicorn creditguard.api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from creditguard.api.errors import register_exception_handlers
from creditguard.api.middleware import RequestContextMiddleware, configure_logging
from creditguard.api.routes import applications, health, model, predict
from creditguard.config import get_settings
from creditguard.scoring import engine

logger = logging.getLogger("creditguard.api")

API_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Eagerly load the active model, its refit feature pipeline and its
    SHAP explainer once at startup, so the first real request doesn't pay
    that one-time cost (typically 10-25s -- see `creditguard.scoring.
    engine`'s module docstring).

    A failed load does NOT stop the app from starting -- liveness
    (`/health`) stays healthy either way. It's surfaced by `/health/ready`
    reporting not-ready instead, per the Phase 8 brief ("fail readiness,
    not liveness, if loading fails").
    """
    configure_logging()
    try:
        engine.warm_up()
        logger.info("startup_model_loaded")
    except Exception:
        logger.error("startup_model_load_failed", exc_info=True)
    yield


def create_app() -> FastAPI:
    """Build the FastAPI app: middleware, exception handlers, routers."""
    settings = get_settings()
    app = FastAPI(
        title="CreditGuard Scoring API",
        description=(
            "Real-time credit scoring: submit a loan application, get back "
            "a credit score, risk category, lending recommendation and "
            "SHAP-based explanation. Educational simulation on synthetic "
            "data -- not a real lending system; see docs/scoring_methodology.md."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    app.include_router(predict.router, prefix=API_PREFIX)
    app.include_router(applications.router, prefix=API_PREFIX)
    app.include_router(model.router, prefix=API_PREFIX)
    # /health, /health/ready, /metrics are unversioned operational endpoints.
    app.include_router(health.router)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "creditguard.api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        workers=int(os.environ.get("API_WORKERS", "1")),
    )
