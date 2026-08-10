"""Tests for the API's authentication layer: X-API-Key required on every
protected endpoint, /health exempt, no internal detail ever leaked in an
error body.
"""

from __future__ import annotations

from conftest import API_KEY

HEADERS = {"X-API-Key": API_KEY}


def test_health_is_unauthenticated(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_ready_is_unauthenticated(api_client) -> None:
    response = api_client.get("/health/ready")
    # No X-API-Key header sent at all -- must not 401.
    assert response.status_code in (200, 503)


def test_predict_without_api_key_is_401(api_client, _example_payload) -> None:
    response = api_client.post("/api/v1/predict", json=_example_payload)
    assert response.status_code == 401


def test_predict_with_wrong_api_key_is_401(api_client, _example_payload) -> None:
    response = api_client.post(
        "/api/v1/predict",
        json=_example_payload,
        headers={"X-API-Key": "definitely-not-the-right-key"},
    )
    assert response.status_code == 401


def test_predict_with_correct_api_key_succeeds(api_client, _example_payload) -> None:
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 200


def test_metrics_requires_api_key(api_client) -> None:
    response = api_client.get("/metrics")
    assert response.status_code == 401


def test_model_info_requires_api_key(api_client) -> None:
    response = api_client.get("/api/v1/model/info")
    assert response.status_code == 401


def test_401_response_includes_request_id_and_no_internal_detail(api_client) -> None:
    response = api_client.post("/api/v1/predict", json={})
    assert response.status_code == 401
    body = response.json()
    assert body["request_id"]
    text = response.text.lower()
    assert "traceback" not in text
    assert "psycopg" not in text


def test_response_carries_request_id_header(api_client, _example_payload) -> None:
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert "x-request-id" in response.headers
    assert response.json()["request_id"] == response.headers["x-request-id"]
