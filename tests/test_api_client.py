"""Tests for creditguard.dashboard.api_client: URL/parsing correctness,
retry-with-backoff on 5xx, and typed exception mapping. HTTP is mocked with
`responses` -- nothing here talks to a real server, and nothing here
imports Streamlit's runtime (only `st.cache_data`-decorated functions are
touched indirectly, which is safe to call outside a running app).
"""

from __future__ import annotations

import pytest
import responses

from creditguard.dashboard.api_client import (
    MAX_RETRIES,
    ApiAuthError,
    ApiClient,
    ApiConnectionError,
    ApiNotFoundError,
    ApiRateLimitError,
    ApiServerError,
    ApiValidationError,
)

BASE_URL = "http://testserver"


def make_client() -> ApiClient:
    return ApiClient(base_url=BASE_URL, api_key="test-key", timeout=1.0)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff would otherwise add real delay to these tests."""
    monkeypatch.setattr(
        "creditguard.dashboard.api_client.time.sleep", lambda _seconds: None
    )


@responses.activate
def test_predict_hits_correct_url_sends_api_key_and_parses_response() -> None:
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/v1/predict",
        json={"credit_score": 700, "risk_category": "LOW"},
        status=200,
    )
    result = make_client().predict({"age": 30})
    assert result == {"credit_score": 700, "risk_category": "LOW"}
    assert responses.calls[0].request.url == f"{BASE_URL}/api/v1/predict"
    assert responses.calls[0].request.headers["X-API-Key"] == "test-key"


@responses.activate
def test_list_predictions_passes_filters_as_query_params() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/predictions",
        json={"items": [], "total": 0, "page": 1, "page_size": 20},
        status=200,
    )
    make_client().list_predictions(risk_category="HIGH", loan_type="AUTO")
    request_url = responses.calls[0].request.url
    assert "risk_category=HIGH" in request_url
    assert "loan_type=AUTO" in request_url
    # None-valued filters are dropped, not sent as "recommendation=None".
    assert "recommendation=" not in request_url


@responses.activate
def test_model_info_hits_correct_url() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/info",
        json={"model_id": "m1"},
        status=200,
    )
    assert make_client().model_info() == {"model_id": "m1"}


@responses.activate
def test_get_application_hits_correct_url() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/applications/LOAN-1",
        json={"loan_id": "LOAN-1"},
        status=200,
    )
    assert make_client().get_application("LOAN-1") == {"loan_id": "LOAN-1"}


@responses.activate
def test_401_raises_auth_error() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/info",
        json={"error": "unauthorized"},
        status=401,
    )
    with pytest.raises(ApiAuthError):
        make_client().model_info()


@responses.activate
def test_404_raises_not_found_error() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/applications/UNKNOWN",
        json={"error": "not found"},
        status=404,
    )
    with pytest.raises(ApiNotFoundError):
        make_client().get_application("UNKNOWN")


@responses.activate
def test_422_raises_validation_error_with_field_detail() -> None:
    responses.add(
        responses.POST,
        f"{BASE_URL}/api/v1/predict",
        json={
            "error": "Validation failed",
            "detail": [
                {"loc": ["body", "age"], "msg": "bad value", "type": "value_error"}
            ],
        },
        status=422,
    )
    with pytest.raises(ApiValidationError) as exc_info:
        make_client().predict({})
    assert exc_info.value.detail == [
        {"loc": ["body", "age"], "msg": "bad value", "type": "value_error"}
    ]


@responses.activate
def test_429_raises_rate_limit_error() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/info",
        json={"error": "rate limited"},
        status=429,
    )
    with pytest.raises(ApiRateLimitError):
        make_client().model_info()


@responses.activate
def test_503_raises_server_error() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/performance",
        json={"error": "not ready"},
        status=503,
    )
    with pytest.raises(ApiServerError):
        make_client().model_performance()


@responses.activate
def test_500_retries_up_to_max_retries_then_raises() -> None:
    for _ in range(MAX_RETRIES):
        responses.add(
            responses.GET,
            f"{BASE_URL}/api/v1/model/info",
            json={"error": "boom"},
            status=500,
        )
    with pytest.raises(ApiServerError):
        make_client().model_info()
    assert len(responses.calls) == MAX_RETRIES


@responses.activate
def test_500_then_200_succeeds_after_one_retry() -> None:
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/info",
        json={"error": "boom"},
        status=500,
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/info",
        json={"model_id": "m1"},
        status=200,
    )
    result = make_client().model_info()
    assert result == {"model_id": "m1"}
    assert len(responses.calls) == 2


@responses.activate
def test_422_is_never_retried() -> None:
    """4xx responses are the caller's problem -- retrying can't fix a bad
    request, so exactly one call should be made.
    """
    responses.add(
        responses.POST, f"{BASE_URL}/api/v1/predict", json={"error": "bad"}, status=422
    )
    with pytest.raises(ApiValidationError):
        make_client().predict({})
    assert len(responses.calls) == 1


@responses.activate
def test_unmatched_request_raises_connection_error() -> None:
    """No endpoint registered at all -- `responses` simulates the API being
    completely unreachable, which the client should surface as
    `ApiConnectionError`, not let a raw `requests` exception through.
    """
    with pytest.raises(ApiConnectionError):
        make_client().model_info()
