"""Typed HTTP client for the Phase 8 CreditGuard API -- the dashboard's only
way of reaching scored predictions, model info and portfolio data. Every
public method wraps exactly one API endpoint (see docs/api.md) and returns
the parsed JSON body as a plain dict/list; nothing here recomputes anything
the API already returns.

Base URL and API key come from environment variables (`API_BASE_URL`,
default `http://localhost:8000`; `API_KEY`), not `creditguard.config.
Settings` -- the dashboard has no business requiring database/MLflow
credentials just to make HTTP calls, and coupling it to that full settings
object would force every dashboard deployment to also carry DB config it
never uses.

Every 5xx response is retried with exponential backoff (`MAX_RETRIES`
attempts); 4xx responses are not retried -- retrying a client error can't
succeed. `/predict`, `/predict/batch`, `/explain` and `/applications`
(state-changing or explicitly "never cache" per the API's own semantics)
are never wrapped in `st.cache_data`; every read-only GET endpoint is,
via the module-level `get_*`/`list_*` functions below, with a TTL short
enough that a demo session still sees new predictions show up.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
import streamlit as st

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 0.5
READ_CACHE_TTL_SECONDS = 30


class ApiClientError(Exception):
    """Base class for every typed exception this client raises. Pages
    should catch this (or a specific subclass) and render a friendly
    message via `creditguard.dashboard.components.cards.render_api_error`
    rather than letting a traceback reach the screen.
    """


class ApiConnectionError(ApiClientError):
    """The API couldn't be reached at all: connection refused, DNS
    failure, or a request that timed out.
    """


class ApiAuthError(ApiClientError):
    """401 -- missing or invalid `X-API-Key`."""


class ApiValidationError(ApiClientError):
    """422 -- the request was rejected. Carries the API's field-level
    `detail` list (see `creditguard.api.schemas.ErrorResponse`) so a page
    can show exactly which field failed, not just a generic message.
    """

    def __init__(
        self, message: str, detail: list[dict[str, Any]] | None = None
    ) -> None:
        super().__init__(message)
        self.detail = detail or []


class ApiNotFoundError(ApiClientError):
    """404 -- e.g. an unknown `loan_id`."""


class ApiRateLimitError(ApiClientError):
    """429 -- rate limit exceeded."""


class ApiServerError(ApiClientError):
    """5xx that persisted through every retry, or a 503 (the active model/
    explainer isn't ready, or -- for `/model/performance` -- its backfill
    hasn't been run yet).
    """


def _base_url() -> str:
    return os.environ.get("API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str:
    return os.environ.get("API_KEY", "")


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(body, dict) and "error" in body:
        return str(body["error"])
    return f"HTTP {response.status_code}"


def _error_detail(response: requests.Response) -> list[dict[str, Any]]:
    try:
        body = response.json()
    except ValueError:
        return []
    if isinstance(body, dict) and isinstance(body.get("detail"), list):
        return body["detail"]
    return []


class ApiClient:
    """One instance per call site is fine -- it holds no connection state
    beyond `requests`' own defaults. `base_url`/`api_key` default to the
    environment (see module docstring) but can be overridden, mainly for
    tests.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.base_url = (base_url if base_url is not None else _base_url()).rstrip("/")
        self.api_key = api_key if api_key is not None else _api_key()
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        clean_params = (
            {k: v for k, v in params.items() if v is not None} if params else None
        )
        last_error: ApiServerError | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.request(
                    method,
                    url,
                    json=json_body,
                    params=clean_params,
                    headers=self._headers(),
                    timeout=self.timeout,
                )
            except requests.ConnectionError as exc:
                raise ApiConnectionError(
                    f"Could not reach the CreditGuard API at {self.base_url}"
                ) from exc
            except requests.Timeout as exc:
                raise ApiConnectionError(
                    f"CreditGuard API did not respond within {self.timeout:g}s"
                ) from exc

            if response.status_code >= 500:
                last_error = ApiServerError(
                    f"{method} {path} -> HTTP {response.status_code}: "
                    f"{_error_message(response)}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                    continue
                raise last_error

            return self._parse(response, method, path)

        raise last_error or ApiServerError(f"{method} {path} failed with no response")

    def _parse(self, response: requests.Response, method: str, path: str) -> Any:
        if response.status_code == 401:
            raise ApiAuthError("Missing or invalid API key")
        if response.status_code == 404:
            raise ApiNotFoundError(_error_message(response))
        if response.status_code == 422:
            raise ApiValidationError(
                _error_message(response), detail=_error_detail(response)
            )
        if response.status_code == 429:
            raise ApiRateLimitError("Rate limit exceeded -- try again shortly")
        if not response.ok:
            raise ApiServerError(
                f"{method} {path} -> HTTP {response.status_code}: "
                f"{_error_message(response)}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # -- predict/explain/applications: never cached ------------------------

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`POST /api/v1/predict` -- score one application (stateless)."""
        return self._request("POST", "/api/v1/predict", json_body=payload)

    def predict_batch(self, payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """`POST /api/v1/predict/batch` -- score up to 1000 applications."""
        return self._request("POST", "/api/v1/predict/batch", json_body=payloads)

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`POST /api/v1/explain` -- full per-feature SHAP breakdown."""
        return self._request("POST", "/api/v1/explain", json_body=payload)

    def create_application(self, payload: dict[str, Any]) -> dict[str, Any]:
        """`POST /api/v1/applications` -- persist + score a real application."""
        return self._request("POST", "/api/v1/applications", json_body=payload)

    # -- reads: wrapped in st.cache_data below, not here --------------------

    def get_application(self, loan_id: str) -> dict[str, Any]:
        """`GET /api/v1/applications/{loan_id}`."""
        return self._request("GET", f"/api/v1/applications/{loan_id}")

    def list_predictions(
        self,
        *,
        page: int = 1,
        page_size: int = 20,
        date_from: str | None = None,
        date_to: str | None = None,
        risk_category: str | None = None,
        recommendation: str | None = None,
        model_id: str | None = None,
        loan_type: str | None = None,
    ) -> dict[str, Any]:
        """`GET /api/v1/predictions` -- paginated, filterable prediction log."""
        return self._request(
            "GET",
            "/api/v1/predictions",
            params={
                "page": page,
                "page_size": page_size,
                "date_from": date_from,
                "date_to": date_to,
                "risk_category": risk_category,
                "recommendation": recommendation,
                "model_id": model_id,
                "loan_type": loan_type,
            },
        )

    def model_info(self) -> dict[str, Any]:
        """`GET /api/v1/model/info`."""
        return self._request("GET", "/api/v1/model/info")

    def model_versions(self) -> dict[str, Any]:
        """`GET /api/v1/model/versions`."""
        return self._request("GET", "/api/v1/model/versions")

    def model_performance(self) -> dict[str, Any]:
        """`GET /api/v1/model/performance`."""
        return self._request("GET", "/api/v1/model/performance")

    def health(self) -> dict[str, Any]:
        """`GET /health` -- liveness, unauthenticated."""
        return self._request("GET", "/health")

    def health_ready(self) -> dict[str, Any]:
        """`GET /health/ready` -- readiness, unauthenticated."""
        return self._request("GET", "/health/ready")

    # -- Phase 10: monitoring ------------------------------------------------

    def monitoring_drift(self, model_id: str | None = None) -> dict[str, Any]:
        """`GET /api/v1/monitoring/drift`."""
        return self._request(
            "GET", "/api/v1/monitoring/drift", params={"model_id": model_id}
        )

    def monitoring_performance(self, model_id: str | None = None) -> dict[str, Any]:
        """`GET /api/v1/monitoring/performance`."""
        return self._request(
            "GET", "/api/v1/monitoring/performance", params={"model_id": model_id}
        )

    def monitoring_data_quality(self, window_days: int = 90) -> dict[str, Any]:
        """`GET /api/v1/monitoring/data-quality`."""
        return self._request(
            "GET",
            "/api/v1/monitoring/data-quality",
            params={"window_days": window_days},
        )


def get_client() -> ApiClient:
    """One client per call -- cheap (no persistent connection state)."""
    return ApiClient()


# -- cached wrappers for read-only endpoints ---------------------------------
# `st.cache_data` keys on argument values; every wrapper takes only
# JSON-serialisable filter arguments (no client instance) so the cache key
# stays stable and independent of `API_BASE_URL`/`API_KEY` changing mid-session.


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_model_info() -> dict[str, Any]:
    return get_client().model_info()


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_model_versions() -> dict[str, Any]:
    return get_client().model_versions()


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_model_performance() -> dict[str, Any]:
    return get_client().model_performance()


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_list_predictions(
    *,
    page: int = 1,
    page_size: int = 20,
    date_from: str | None = None,
    date_to: str | None = None,
    risk_category: str | None = None,
    recommendation: str | None = None,
    model_id: str | None = None,
    loan_type: str | None = None,
) -> dict[str, Any]:
    return get_client().list_predictions(
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        risk_category=risk_category,
        recommendation=recommendation,
        model_id=model_id,
        loan_type=loan_type,
    )


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_get_application(loan_id: str) -> dict[str, Any]:
    return get_client().get_application(loan_id)


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_drift(model_id: str | None = None) -> dict[str, Any]:
    return get_client().monitoring_drift(model_id)


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_performance(model_id: str | None = None) -> dict[str, Any]:
    return get_client().monitoring_performance(model_id)


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def cached_monitoring_data_quality(window_days: int = 90) -> dict[str, Any]:
    return get_client().monitoring_data_quality(window_days)


@st.cache_data(ttl=READ_CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all_predictions(
    *,
    date_from: str | None = None,
    date_to: str | None = None,
    max_rows: int = 2000,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Page through `GET /predictions` (max `page_size` of 100 per the API's
    own cap) until every matching row is collected or `max_rows` is hit --
    there is no bulk "all predictions" endpoint, so Portfolio Analytics'
    aggregate views are built client-side from these paginated reads.
    Additional filters (risk category, recommendation, loan type) are
    applied by the caller after fetching, since the API only accepts one
    value per filter but the dashboard's sidebar offers multi-select.
    """
    client = get_client()
    rows: list[dict[str, Any]] = []
    page = 1
    while len(rows) < max_rows:
        response = client.list_predictions(
            page=page, page_size=page_size, date_from=date_from, date_to=date_to
        )
        items = response["items"]
        rows.extend(items)
        if len(items) < page_size or response["total"] <= len(rows):
            break
        page += 1
    return rows[:max_rows]
