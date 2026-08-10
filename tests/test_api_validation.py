"""Tests for the API's request validation: schema constraints, cross-field
validators, and batch size limits -- all mapped to 422, never a crash.
"""

from __future__ import annotations

from conftest import API_KEY

HEADERS = {"X-API-Key": API_KEY}


def test_valid_payload_returns_200_with_every_field_populated_and_typed(
    api_client, _example_payload
) -> None:
    response = api_client.post(
        "/api/v1/predict", json=_example_payload, headers=HEADERS
    )
    assert response.status_code == 200
    body = response.json()

    assert isinstance(body["request_id"], str) and body["request_id"]
    assert body["loan_id"] is None  # not supplied in _example_payload
    assert isinstance(body["credit_score"], int)
    assert 300 <= body["credit_score"] <= 900
    assert isinstance(body["default_probability"], float)
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_category"] in {
        "VERY_LOW",
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH",
    }
    assert body["recommendation"] in {"APPROVE", "REVIEW", "REJECT"}
    assert isinstance(body["triggered_rules"], list) and body["triggered_rules"]
    assert isinstance(body["top_risk_factors"], list)
    assert isinstance(body["top_positive_factors"], list)
    for row in body["top_risk_factors"] + body["top_positive_factors"]:
        assert {"feature", "value", "impact", "description"} <= row.keys()
        assert isinstance(row["impact"], float)
        assert isinstance(row["description"], str) and row["description"]
    assert isinstance(body["model_id"], str) and body["model_id"]
    assert isinstance(body["model_version"], str) and body["model_version"]
    assert isinstance(body["latency_ms"], int)
    assert isinstance(body["scored_at"], str)


def test_age_below_minimum_is_422_and_names_the_field(
    api_client, _example_payload
) -> None:
    payload = {**_example_payload, "age": 15}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any("age" in item["loc"] for item in detail)


def test_age_above_maximum_is_422(api_client, _example_payload) -> None:
    payload = {**_example_payload, "age": 150}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_negative_income_is_422(api_client, _example_payload) -> None:
    payload = {**_example_payload, "annual_income": -1000.0}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_annual_income_inconsistent_with_monthly_income_is_422(
    api_client, _example_payload
) -> None:
    payload = {**_example_payload, "monthly_income": 5000.0}  # far below annual/12
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_employment_years_exceeding_age_minus_16_is_422(
    api_client, _example_payload
) -> None:
    payload = {**_example_payload, "age": 20, "employment_years": 10.0}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_total_outstanding_exceeding_1_5x_limit_is_422(
    api_client, _example_payload
) -> None:
    payload = {
        **_example_payload,
        "total_credit_limit": 100000.0,
        "total_outstanding": 200000.0,
    }
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_expenses_plus_emi_exceeding_2x_monthly_income_is_422(
    api_client, _example_payload
) -> None:
    payload = {
        **_example_payload,
        "monthly_income": 20000.0,
        "monthly_expenses": 30000.0,
        "monthly_emi": 30000.0,
    }
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_invalid_enum_value_is_422(api_client, _example_payload) -> None:
    payload = {**_example_payload, "loan_type": "NOT_A_REAL_LOAN_TYPE"}
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_missing_required_field_is_422(api_client, _example_payload) -> None:
    payload = dict(_example_payload)
    del payload["annual_income"]
    response = api_client.post("/api/v1/predict", json=payload, headers=HEADERS)
    assert response.status_code == 422


def test_batch_of_100_returns_100_results_in_original_order(
    api_client, _example_payload
) -> None:
    batch = [{**_example_payload, "loan_id": f"BATCH-{i:03d}"} for i in range(100)]
    response = api_client.post("/api/v1/predict/batch", json=batch, headers=HEADERS)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 100
    assert [r["loan_id"] for r in results] == [f"BATCH-{i:03d}" for i in range(100)]


def test_batch_over_the_limit_is_422(api_client, _example_payload) -> None:
    batch = [_example_payload] * 1001
    response = api_client.post("/api/v1/predict/batch", json=batch, headers=HEADERS)
    assert response.status_code == 422


def test_batch_empty_is_422(api_client) -> None:
    response = api_client.post("/api/v1/predict/batch", json=[], headers=HEADERS)
    assert response.status_code == 422


def test_no_error_response_leaks_stack_trace_or_db_error_text(
    api_client, _example_payload
) -> None:
    cases = [
        api_client.post(
            "/api/v1/predict", json={**_example_payload, "age": 5}, headers=HEADERS
        ),
        api_client.post("/api/v1/predict", json={}, headers=HEADERS),
        api_client.post(
            "/api/v1/predict/batch", json=[_example_payload] * 1001, headers=HEADERS
        ),
    ]
    for response in cases:
        assert response.status_code == 422
        text = response.text.lower()
        assert "traceback" not in text
        assert "psycopg" not in text
        assert 'file "' not in text
