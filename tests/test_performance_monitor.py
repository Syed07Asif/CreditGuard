"""Tests for creditguard.monitoring.performance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.monitoring import performance


def _matured_frame(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.2, size=n)
    # Predicted probability correlated with the label so the metric suite
    # returns something meaningfully non-degenerate (not used for exact
    # value assertions, just "restricts to matured loans only" behaviour).
    p = np.clip(y * 0.6 + rng.normal(0, 0.1, size=n), 0.01, 0.99)
    return pd.DataFrame({"default_12m": y, "default_probability": p})


def test_compute_predictive_metrics_returns_none_below_minimum() -> None:
    matured = _matured_frame(10)
    assert performance.compute_predictive_metrics(matured, min_matured_loans=30) is None


def test_compute_predictive_metrics_computes_when_enough_matured_loans() -> None:
    matured = _matured_frame(200)
    result = performance.compute_predictive_metrics(matured, min_matured_loans=30)
    assert result is not None
    assert "roc_auc" in result
    assert "pr_auc" in result
    assert "calibration_slope" in result


def test_fetch_matured_predictions_restricts_to_matured_loans_only() -> None:
    """Only a `predictions` row whose `loan_id` has a matching, in-window
    `loan_outcomes` row counts as "matured" -- a prediction with no outcome
    at all (e.g. a stateless `/predict` call) must be excluded even though
    it's for the same model.
    """
    from datetime import UTC, date, datetime

    from creditguard.db.repository import (
        CustomerRepository,
        LoanApplicationRepository,
        LoanOutcomeRepository,
        PredictionRepository,
    )
    from creditguard.models import registry

    model_row = registry.register_model(
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="ds1",
        feature_list=["income"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.5},
        mlflow_run_id="run-1",
        artifact_path="unused",
    )
    model_id = model_row["model_id"]

    # loan_outcomes.loan_id is a FK to loan_applications, which in turn FKs
    # customers -- both need a real row before "L-MATURED" can get an outcome.
    CustomerRepository().insert_many(
        [
            {
                "customer_id": "C1",
                "age": 30,
                "gender": "MALE",
                "marital_status": "SINGLE",
                "dependents": 0,
                "education": "GRADUATE",
                "employment_type": "SALARIED",
                "employment_years": 5,
                "annual_income": 600000,
                "monthly_income": 50000,
                "city_tier": 1,
            }
        ]
    )
    LoanApplicationRepository().insert_many(
        [
            {
                "loan_id": "L-MATURED",
                "customer_id": "C1",
                "loan_type": "PERSONAL",
                "loan_amount": 100000,
                "loan_tenure_months": 24,
                "interest_rate": 12.0,
                "loan_purpose": "OTHER",
                "application_date": date(2023, 5, 1),
                "status": "APPROVED",
            }
        ]
    )
    PredictionRepository().insert_many(
        [
            {
                "loan_id": "L-MATURED",
                "customer_id": "C1",
                "model_id": model_id,
                "default_probability": 0.2,
                "credit_score": 700,
                "risk_category": "LOW",
                "recommendation": "APPROVE",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 50,
                "request_source": "test",
            },
            {
                "loan_id": "L-UNMATURED",
                "customer_id": "C2",
                "model_id": model_id,
                "default_probability": 0.3,
                "credit_score": 650,
                "risk_category": "MODERATE",
                "recommendation": "REVIEW",
                "top_risk_factors": [],
                "top_positive_factors": [],
                "latency_ms": 60,
                "request_source": "test",
            },
        ]
    )
    LoanOutcomeRepository().insert_many(
        [
            {
                "loan_id": "L-MATURED",
                "default_12m": 0,
                "outcome_observed_date": date(2024, 6, 15),
            }
        ]
    )

    matured = performance.fetch_matured_predictions(
        model_id, date(2024, 6, 1), date(2024, 6, 30)
    )
    assert list(matured["loan_id"]) == ["L-MATURED"]


def test_compute_operational_metrics_empty_frame() -> None:
    result = performance.compute_operational_metrics(pd.DataFrame())
    assert result["prediction_volume"] == 0.0
    assert math_isnan(result["approve_rate"])


def test_compute_operational_metrics_decision_mix_and_latency() -> None:
    frame = pd.DataFrame(
        {
            "recommendation": ["APPROVE", "APPROVE", "REVIEW", "REJECT"],
            "credit_score": [700, 720, 600, 400],
            "default_probability": [0.05, 0.04, 0.2, 0.6],
            "risk_category": ["LOW", "LOW", "MODERATE", "VERY_HIGH"],
            "latency_ms": [100, 120, 90, 110],
        }
    )
    result = performance.compute_operational_metrics(frame)
    assert result["prediction_volume"] == 4.0
    assert result["approve_rate"] == pytest.approx(0.5)
    assert result["review_rate"] == pytest.approx(0.25)
    assert result["reject_rate"] == pytest.approx(0.25)
    assert result["high_risk_share"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("metric_name", "training_value", "current_value", "expected"),
    [
        ("roc_auc", 0.80, 0.65, True),  # higher-is-better, dropped > 10%
        ("roc_auc", 0.80, 0.78, False),  # within tolerance
        ("brier_score", 0.10, 0.13, True),  # lower-is-better, rose > 10%
        ("brier_score", 0.10, 0.105, False),
        ("calibration_slope", 1.0, 1.5, True),  # distance from 1.0 grew a lot
        ("calibration_slope", 1.0, 1.02, False),
    ],
)
def test_is_degraded_directionality(
    metric_name: str, training_value: float, current_value: float, expected: bool
) -> None:
    assert (
        performance.is_degraded(
            metric_name, training_value, current_value, tolerance=0.10
        )
        is expected
    )


def test_is_degraded_ignores_nan_inputs() -> None:
    assert performance.is_degraded("roc_auc", float("nan"), 0.5, 0.10) is False
    assert performance.is_degraded("roc_auc", 0.8, float("nan"), 0.10) is False


def math_isnan(value: float) -> bool:
    return value != value
