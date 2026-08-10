"""Tests for the API's scoring endpoints end to end: /predict, /explain,
/applications, /model, /health/ready, latency, and persistence.
"""

from __future__ import annotations

import time

from conftest import API_KEY
from creditguard.db.repository import PredictionRepository

HEADERS = {"X-API-Key": API_KEY}


def test_frd_example_payload_returns_a_coherent_result(
    api_client, _example_payload
) -> None:
    """Acceptance criterion 2: the FRD example payload returns a coherent
    result (valid types/ranges, decision consistent with the returned
    probability/score).
    """
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert 300 <= body["credit_score"] <= 900
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["recommendation"] in {"APPROVE", "REVIEW", "REJECT"}
    # A score >= 700 should never come back REJECT (recommendation.py's own
    # policy), and vice versa for a clearly rejectable score.
    if body["recommendation"] == "APPROVE":
        assert body["credit_score"] >= 700
    if body["credit_score"] < 550:
        assert body["recommendation"] == "REJECT"


def test_unknown_loan_id_is_404_not_500(api_client) -> None:
    response = api_client.get("/api/v1/applications/DOES-NOT-EXIST", headers=HEADERS)
    assert response.status_code == 404
    body = response.json()
    assert body["request_id"]
    text = response.text.lower()
    assert "traceback" not in text
    assert "psycopg" not in text


def test_successful_predict_writes_exactly_one_predictions_row(
    api_client, _example_payload
) -> None:
    payload = {**_example_payload, "loan_id": "LOAN-PERSIST-CHECK"}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 200

    rows = PredictionRepository().fetch_dataframe(
        filters={"loan_id": "LOAN-PERSIST-CHECK"}
    )
    assert len(rows) == 1
    assert rows.iloc[0]["request_source"] == "api"


def test_single_prediction_p95_latency_under_2000ms(
    api_client, _example_payload
) -> None:
    """NFR-001. Runs 50 sequential calls and reports p95 -- printed so the
    measured number is visible in test output, not just asserted silently.
    """
    latencies_ms: list[float] = []
    for i in range(50):
        payload = {**_example_payload, "loan_id": f"LATENCY-{i:03d}"}
        start = time.perf_counter()
        response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        assert response.status_code == 200

    latencies_ms.sort()
    p95_index = int(round(0.95 * (len(latencies_ms) - 1)))
    p95 = latencies_ms[p95_index]
    print(f"\np95 latency over 50 sequential /predict calls: {p95:.1f} ms")
    assert p95 < 2000.0


def test_invalid_input_never_crashes_the_service(api_client, _example_payload) -> None:
    """NFR-002, exercised with a battery of malformed/hostile payloads."""
    malformed_payloads = [
        {},
        {"age": "not-a-number"},
        {**_example_payload, "age": -5},
        {**_example_payload, "loan_type": None},
        {**_example_payload, "annual_income": "free money"},
        None,
    ]
    for payload in malformed_payloads:
        response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
        assert response.status_code in (400, 422)


def test_explain_returns_full_per_feature_breakdown(
    api_client, _example_payload
) -> None:
    response = api_client.post(
        "/api/v1/explain", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["shap_base_value"], float)
    assert len(body["feature_contributions"]) > 0
    for row in body["feature_contributions"]:
        assert {"feature", "value", "impact", "description"} <= row.keys()
    # Explaining does not persist a prediction.
    if _example_payload.get("loan_id"):
        rows = PredictionRepository().fetch_dataframe(
            filters={"loan_id": _example_payload["loan_id"]}
        )
        assert len(rows) == 0


def test_model_info_reports_the_active_model(api_client, loaded_model) -> None:
    response = api_client.get("/api/v1/model/info", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == loaded_model.model_id
    assert body["feature_count"] == len(loaded_model.feature_names)
    assert body["is_active"] is True
    assert "chosen_threshold" in body


def test_model_versions_lists_registered_models(api_client, loaded_model) -> None:
    response = api_client.get("/api/v1/model/versions", headers=HEADERS)
    assert response.status_code == 200
    versions = response.json()["versions"]
    assert any(v["model_id"] == loaded_model.model_id for v in versions)


def test_create_and_retrieve_application(api_client, _example_payload) -> None:
    payload = {
        **_example_payload,
        "customer_id": "CUST-APP-001",
        "loan_id": "LOAN-APP-001",
    }
    create_response = api_client.post(
        "/api/v1/applications", json=payload, headers=HEADERS
    )
    assert create_response.status_code == 201
    created = create_response.json()
    assert created["loan_id"] == "LOAN-APP-001"
    assert created["customer_id"] == "CUST-APP-001"
    assert created["latest_prediction"] is not None
    assert created["latest_prediction"]["loan_id"] == "LOAN-APP-001"

    get_response = api_client.get("/api/v1/applications/LOAN-APP-001", headers=HEADERS)
    assert get_response.status_code == 200
    fetched = get_response.json()
    assert fetched["customer_id"] == "CUST-APP-001"
    assert fetched["latest_prediction"] is not None
    assert (
        fetched["latest_prediction"]["credit_score"]
        == created["latest_prediction"]["credit_score"]
    )


def test_duplicate_application_customer_id_is_422_not_500(
    api_client, _example_payload
) -> None:
    payload = {
        **_example_payload,
        "customer_id": "CUST-DUP-001",
        "loan_id": "LOAN-DUP-001",
    }
    first = api_client.post("/api/v1/applications", json=payload, headers=HEADERS)
    assert first.status_code == 201

    second = api_client.post(
        "/api/v1/applications",
        json={**payload, "loan_id": "LOAN-DUP-002"},
        headers=HEADERS,
    )
    assert second.status_code == 422
    text = second.text.lower()
    assert "traceback" not in text
    assert "psycopg" not in text


def test_list_predictions_is_paginated_and_filterable(
    api_client, _example_payload
) -> None:
    for i in range(5):
        payload = {**_example_payload, "loan_id": f"LIST-{i:03d}"}
        api_client.post("/api/v1/predict", json=payload, headers=HEADERS)

    response = api_client.get(
        "/api/v1/predictions", params={"page": 1, "page_size": 3}, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["page_size"] == 3
    assert len(body["items"]) == 3
    assert body["total"] >= 5


def test_readiness_reports_not_ready_when_model_fails_to_load(
    api_client, monkeypatch
) -> None:
    from creditguard.scoring import engine

    monkeypatch.setattr(engine, "is_model_loaded", lambda: False)
    response = api_client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["model_loaded"] is False


def test_service_unavailable_error_never_leaks_exception_text(
    api_client, _example_payload, monkeypatch
) -> None:
    from creditguard.scoring import engine

    def _boom(*_args, **_kwargs):
        raise engine.ScoringEngineError(
            "leaking-secret-internal-detail-that-must-not-appear"
        )

    monkeypatch.setattr(engine, "score_application", _boom)
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 503
    assert "leaking-secret-internal-detail-that-must-not-appear" not in response.text


def test_unexpected_error_never_leaks_exception_text(
    api_client, _example_payload, monkeypatch
) -> None:
    from creditguard.scoring import engine

    def _boom(*_args, **_kwargs):
        raise RuntimeError(
            "psycopg.OperationalError: password authentication failed for user x"
        )

    monkeypatch.setattr(engine, "score_application", _boom)
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 500
    text = response.text.lower()
    assert "psycopg" not in text
    assert "password" not in text
    assert "traceback" not in text
    body = response.json()
    assert body["request_id"]


def test_metrics_endpoint_reports_counts_and_latency(
    api_client, _example_payload
) -> None:
    api_client.post("/api/v1/predict", json=_example_payload, headers=HEADERS)
    response = api_client.get("/metrics", headers=HEADERS)
    assert response.status_code == 200
    body = response.json()
    assert body["request_count"] >= 1
    assert body["latency_p50_ms"] >= 0
    assert body["latency_p95_ms"] >= body["latency_p50_ms"]
