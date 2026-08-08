"""Tests for creditguard.db.models: schema creation and CHECK constraint enforcement."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from creditguard.db.engine import get_engine, get_session
from creditguard.db.models import Customer, ModelRegistry, Prediction

EXPECTED_TABLES = {
    "customers",
    "loan_applications",
    "financial_profiles",
    "credit_history",
    "loan_outcomes",
    "model_registry",
    "predictions",
    "data_quality_issues",
    "monitoring_metrics",
    "drift_reports",
}


def test_init_db_creates_every_expected_table() -> None:
    """Every table declared in db/schema.sql should exist after schema application."""
    inspector = inspect(get_engine())
    actual_tables = set(inspector.get_table_names())
    assert EXPECTED_TABLES <= actual_tables


def _valid_customer(customer_id: str = "CUST-1", **overrides: object) -> Customer:
    defaults: dict[str, object] = dict(
        customer_id=customer_id,
        age=30,
        gender="F",
        marital_status="SINGLE",
        dependents=0,
        education="GRADUATE",
        employment_type="SALARIED",
        employment_years=5,
        annual_income=600000,
        monthly_income=50000,
        city_tier=1,
    )
    defaults.update(overrides)
    return Customer(**defaults)


def _valid_model_registry(model_id: str = "MODEL-1") -> ModelRegistry:
    return ModelRegistry(
        model_id=model_id,
        model_version="v1",
        algorithm="XGBOOST",
        training_date=datetime(2026, 1, 1),
        dataset_version="v1",
        feature_list=[],
        hyperparameters={},
        metrics={},
        mlflow_run_id="run-1",
        artifact_path="models/artifacts/model-1",
        is_active=True,
    )


def test_customer_age_check_constraint_rejects_bad_row() -> None:
    """A customer age outside 18-100 must be rejected by the CHECK constraint."""
    with pytest.raises(IntegrityError):
        with get_session() as session:
            session.add(_valid_customer(age=5))


def test_customer_negative_income_check_constraint_rejects_bad_row() -> None:
    """A negative annual income must be rejected by the CHECK constraint."""
    with pytest.raises(IntegrityError):
        with get_session() as session:
            session.add(_valid_customer(annual_income=-1))


def test_prediction_credit_score_check_constraint_rejects_bad_row() -> None:
    """A credit score outside 300-900 must be rejected by the CHECK constraint."""
    with get_session() as session:
        session.add(_valid_customer(customer_id="CUST-2"))
        session.add(_valid_model_registry())

    with pytest.raises(IntegrityError):
        with get_session() as session:
            session.add(
                Prediction(
                    loan_id="LOAN-1",
                    customer_id="CUST-2",
                    model_id="MODEL-1",
                    default_probability=0.1,
                    credit_score=1000,
                    risk_category="LOW",
                    recommendation="APPROVE",
                    top_risk_factors={},
                    top_positive_factors={},
                    latency_ms=10,
                    request_source="api",
                )
            )


def test_prediction_recommendation_check_constraint_rejects_bad_row() -> None:
    """An unrecognised recommendation value must be rejected by the CHECK constraint."""
    with get_session() as session:
        session.add(_valid_customer(customer_id="CUST-3"))
        session.add(_valid_model_registry(model_id="MODEL-2"))

    with pytest.raises(IntegrityError):
        with get_session() as session:
            session.add(
                Prediction(
                    loan_id="LOAN-2",
                    customer_id="CUST-3",
                    model_id="MODEL-2",
                    default_probability=0.1,
                    credit_score=650,
                    risk_category="MEDIUM",
                    recommendation="MAYBE",
                    top_risk_factors={},
                    top_positive_factors={},
                    latency_ms=10,
                    request_source="api",
                )
            )
