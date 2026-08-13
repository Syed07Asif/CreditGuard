"""Central exception -> HTTP response mapping for the API.

Every handler returns a `creditguard.api.schemas.ErrorResponse` and
includes the request_id from `request.state.request_id` (set by
`creditguard.api.middleware.RequestContextMiddleware`, which runs before
routing/exception handling). Nothing here ever includes a stack trace, a
raw exception message from a database driver, or any other internal detail
in the response body -- the generic 500 handler logs the real exception
server-side only and returns a fixed, safe message.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from creditguard.api.schemas import ErrorDetailItem, ErrorResponse
from creditguard.explain.shap_explainer import ExplainabilityError
from creditguard.features.leakage import LeakageError
from creditguard.scoring.engine import ScoringEngineError, ScoringInputError

logger = logging.getLogger("creditguard.api")


class LoanNotFoundError(Exception):
    """Raised when a requested `loan_id` has no stored application --
    mapped to 404, not the generic 500.
    """

    def __init__(self, loan_id: str) -> None:
        self.loan_id = loan_id
        super().__init__(f"No application found for loan_id={loan_id!r}")


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _json_error(
    request: Request,
    status_code: int,
    error: str,
    detail: list[ErrorDetailItem] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(request_id=_request_id(request), error=error, detail=detail)
    return JSONResponse(
        status_code=status_code, content=payload.model_dump(mode="json")
    )


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """A malformed/out-of-range request body: 422 with a field-level detail
    list (`loc` names the offending field, e.g. `["body", "age"]`).
    """
    detail = [
        ErrorDetailItem(
            loc=[str(part) for part in error["loc"]],
            msg=error["msg"],
            type=error["type"],
        )
        for error in exc.errors()
    ]
    return _json_error(
        request, status.HTTP_422_UNPROCESSABLE_ENTITY, "Validation failed", detail
    )


async def handle_scoring_input_error(
    request: Request, exc: ScoringInputError
) -> JSONResponse:
    """Defense in depth: the engine's own input validation rejected
    something even though the API schema accepted it (schema/engine
    constraints should stay in sync, but this keeps the response honest if
    they ever drift).
    """
    return _json_error(request, status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc))


async def handle_service_unavailable(request: Request, exc: Exception) -> JSONResponse:
    """The active model, feature pipeline or SHAP explainer isn't ready
    (no model registered, a leakage-guard trip, or a missing explainability
    artifact) -- 503, not 500: this is the service not being ready to
    serve, not the request being wrong.
    """
    logger.error(
        "service_unavailable",
        extra={"request_id": _request_id(request), "error_type": type(exc).__name__},
        exc_info=exc,
    )
    return _json_error(
        request,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Scoring service temporarily unavailable -- the active model or "
        "explainer isn't ready",
    )


async def handle_loan_not_found(
    request: Request, exc: LoanNotFoundError
) -> JSONResponse:
    return _json_error(request, status.HTTP_404_NOT_FOUND, str(exc))


async def handle_integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
    """A database constraint violation (e.g. a duplicate `customer_id`/
    `loan_id` on `POST /applications`) -- 422, a client-input problem, not
    a server failure. The raw driver error text is logged server-side only.
    """
    logger.warning(
        "integrity_error", extra={"request_id": _request_id(request)}, exc_info=exc
    )
    return _json_error(
        request,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "Request conflicts with an existing record (duplicate identifier "
        "or constraint violation)",
    )


async def handle_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    """Anything raised as a plain `HTTPException` (auth, rate limiting)
    keeps its own status code, reshaped into the same error envelope.
    """
    return _json_error(request, exc.status_code, str(exc.detail))


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log the real exception (server-side only, with
    traceback) and return a fixed, generic 500 message plus the request ID
    -- never the exception's own text, which could contain a database
    error string or other internal detail.
    """
    logger.error(
        "unhandled_exception",
        extra={"request_id": _request_id(request), "error_type": type(exc).__name__},
        exc_info=exc,
    )
    return _json_error(
        request,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "An unexpected error occurred. Please contact support with the request ID.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every handler above onto `app`. Starlette dispatches by the
    most specific matching exception type in the MRO, so registration
    order doesn't affect behaviour -- listed most- to least-specific here
    purely for readability.
    """
    # Starlette's own type stub for add_exception_handler expects every
    # handler typed as Callable[[Request, Exception], ...] regardless of
    # which exception type it's registered for -- dispatch at runtime is by
    # the specific type given here (that's the whole point of registering
    # per-type handlers), so these narrower handler signatures are correct
    # in practice; only the stub's variance is imprecise for this pattern.
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)  # type: ignore[arg-type]
    app.add_exception_handler(ScoringInputError, handle_scoring_input_error)  # type: ignore[arg-type]
    app.add_exception_handler(LeakageError, handle_service_unavailable)
    app.add_exception_handler(ScoringEngineError, handle_service_unavailable)
    app.add_exception_handler(ExplainabilityError, handle_service_unavailable)
    app.add_exception_handler(LoanNotFoundError, handle_loan_not_found)  # type: ignore[arg-type]
    app.add_exception_handler(IntegrityError, handle_integrity_error)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, handle_http_exception)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, handle_unexpected_error)
