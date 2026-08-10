"""Tests for creditguard.db.repository: insert/read round-trips via the repository."""

from __future__ import annotations

from datetime import UTC, date, datetime

from creditguard.db.repository import (
    CustomerRepository,
    LoanApplicationRepository,
    PredictionRepository,
)
from creditguard.models import registry


def _customer_record(customer_id: str, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = dict(
        customer_id=customer_id,
        age=30,
        gender="F",
        marital_status="SINGLE",
        dependents=0,
        education="GRADUATE",
        employment_type="SALARIED",
        employment_years=5,
        annual_income=500000,
        monthly_income=40000,
        city_tier=2,
    )
    record.update(overrides)
    return record


def test_customer_insert_and_get_by_id_round_trip() -> None:
    """A customer inserted via the repository should read back with the same values."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-100", age=42, annual_income=1200000)])

    fetched = repo.get_by_id("CUST-100")

    assert fetched is not None
    assert fetched["customer_id"] == "CUST-100"
    assert fetched["age"] == 42
    assert float(fetched["annual_income"]) == 1200000


def test_fetch_dataframe_filters_by_column() -> None:
    """fetch_dataframe should return only rows matching the given equality filters."""
    repo = CustomerRepository()
    repo.insert_many(
        [
            _customer_record("CUST-200", city_tier=3),
            _customer_record("CUST-201", city_tier=1),
        ]
    )

    df = repo.fetch_dataframe(filters={"city_tier": 3})

    assert len(df) == 1
    assert df.iloc[0]["customer_id"] == "CUST-200"


def test_fetch_dataframe_with_raw_sql() -> None:
    """fetch_dataframe should also support executing raw SQL as a DataFrame."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-210")])

    df = repo.fetch_dataframe(sql="SELECT customer_id FROM customers")

    assert "CUST-210" in df["customer_id"].tolist()


def test_upsert_updates_existing_row() -> None:
    """Upserting a record with an existing primary key should update it in place."""
    repo = CustomerRepository()
    repo.insert_many([_customer_record("CUST-300", age=25)])

    repo.upsert([_customer_record("CUST-300", age=26)])

    fetched = repo.get_by_id("CUST-300")
    assert fetched is not None
    assert fetched["age"] == 26


def test_loan_application_round_trip_with_foreign_key() -> None:
    """A loan application referencing an existing customer should round-trip."""
    customer_repo = CustomerRepository()
    loan_repo = LoanApplicationRepository()

    customer_repo.insert_many([_customer_record("CUST-400")])
    loan_repo.insert_many(
        [
            {
                "loan_id": "LOAN-400",
                "customer_id": "CUST-400",
                "loan_type": "PERSONAL",
                "loan_amount": 200000,
                "loan_tenure_months": 24,
                "interest_rate": 12.5,
                "loan_purpose": "MEDICAL",
                "application_date": date(2026, 1, 15),
                "status": "PENDING",
            }
        ]
    )

    fetched = loan_repo.get_by_id("LOAN-400")

    assert fetched is not None
    assert fetched["customer_id"] == "CUST-400"
    assert fetched["status"] == "PENDING"


def _register_dummy_model(model_id: str) -> None:
    registry.register_model(
        model_id=model_id,
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="test-version",
        feature_list=["dummy"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.5},
        mlflow_run_id="test-run",
        artifact_path="unused-in-test",
    )


def test_query_predictions_filters_by_risk_category_and_recommendation() -> None:
    _register_dummy_model("model-query-1")
    repo = PredictionRepository()
    repo.insert_many(
        [
            {
                "loan_id": "LOAN-Q1",
                "customer_id": "CUST-Q1",
                "model_id": "model-query-1",
                "default_probability": 0.05,
                "credit_score": 800,
                "risk_category": "VERY_LOW",
                "recommendation": "APPROVE",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 10,
                "request_source": "test",
            },
            {
                "loan_id": "LOAN-Q2",
                "customer_id": "CUST-Q2",
                "model_id": "model-query-1",
                "default_probability": 0.9,
                "credit_score": 320,
                "risk_category": "VERY_HIGH",
                "recommendation": "REJECT",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 12,
                "request_source": "test",
            },
        ]
    )

    rows, total = repo.query_predictions(risk_category="VERY_HIGH")
    assert total == 1
    assert rows[0]["loan_id"] == "LOAN-Q2"

    rows, total = repo.query_predictions(recommendation="APPROVE")
    assert total == 1
    assert rows[0]["loan_id"] == "LOAN-Q1"

    rows, total = repo.query_predictions(model_id="model-query-1")
    assert total == 2


def test_query_predictions_is_paginated_newest_first() -> None:
    _register_dummy_model("model-query-2")
    repo = PredictionRepository()
    base_time = datetime(2026, 1, 1, tzinfo=UTC)
    repo.insert_many(
        [
            {
                "loan_id": f"LOAN-P{i}",
                "customer_id": f"CUST-P{i}",
                "model_id": "model-query-2",
                "default_probability": 0.1,
                "credit_score": 700,
                "risk_category": "LOW",
                "recommendation": "APPROVE",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 5,
                "request_source": "test",
                "created_at": base_time.replace(day=1 + i),
            }
            for i in range(5)
        ]
    )

    rows, total = repo.query_predictions(model_id="model-query-2", page=1, page_size=2)
    assert total == 5
    assert len(rows) == 2
    assert [r["loan_id"] for r in rows] == ["LOAN-P4", "LOAN-P3"]

    rows, total = repo.query_predictions(model_id="model-query-2", page=2, page_size=2)
    assert [r["loan_id"] for r in rows] == ["LOAN-P2", "LOAN-P1"]


def test_query_predictions_filters_by_date_range() -> None:
    _register_dummy_model("model-query-3")
    repo = PredictionRepository()
    repo.insert_many(
        [
            {
                "loan_id": "LOAN-D1",
                "customer_id": "CUST-D1",
                "model_id": "model-query-3",
                "default_probability": 0.1,
                "credit_score": 700,
                "risk_category": "LOW",
                "recommendation": "APPROVE",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 5,
                "request_source": "test",
                "created_at": datetime(2026, 1, 1, tzinfo=UTC),
            },
            {
                "loan_id": "LOAN-D2",
                "customer_id": "CUST-D2",
                "model_id": "model-query-3",
                "default_probability": 0.1,
                "credit_score": 700,
                "risk_category": "LOW",
                "recommendation": "APPROVE",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 5,
                "request_source": "test",
                "created_at": datetime(2026, 6, 1, tzinfo=UTC),
            },
        ]
    )

    rows, total = repo.query_predictions(
        date_from=datetime(2026, 3, 1, tzinfo=UTC), model_id="model-query-3"
    )
    assert total == 1
    assert rows[0]["loan_id"] == "LOAN-D2"
