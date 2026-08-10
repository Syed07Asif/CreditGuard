"""Tests for creditguard.scoring.engine: input validation, point-in-time
frame assembly, and score_application end-to-end (against a small,
in-memory-fitted model injected into the module cache -- not the real
Phase 6 model/dataset, which is exercised separately by the fixture-scoring
demo script for the real acceptance-criteria run).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from creditguard.db.repository import PredictionRepository
from creditguard.explain import shap_explainer
from creditguard.features.build import align_target, filter_tables, temporal_split
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.models import registry
from creditguard.scoring import categories, engine, recommendation, scorecard
from creditguard.scoring.engine import ScoringInputError
from creditguard.validation.engine import load_rule_config


def _features_config() -> dict:
    return {
        "behavioural": {"n_quantile_bins": 5},
        "preprocessing": {"min_frequency": 0.01},
        "feature_columns": {
            "numeric": [
                "age",
                "dependents",
                "employment_years",
                "annual_income",
                "city_tier",
                "loan_amount",
                "loan_tenure_months",
                "interest_rate",
                "monthly_income",
                "monthly_expenses",
                "existing_loan_count",
                "existing_loan_amount",
                "monthly_emi",
                "savings_balance",
                "total_assets",
                "total_liabilities",
                "credit_history_months",
                "num_credit_accounts",
                "total_credit_limit",
                "total_outstanding",
                "previous_defaults",
                "late_payments_12m",
                "missed_payments_12m",
                "active_loans",
                "closed_loans",
                "dti",
                "emi_to_income",
                "credit_utilization",
                "loan_to_income",
                "proposed_emi",
                "post_loan_dti",
                "savings_to_income",
                "net_worth",
                "leverage_ratio",
                "disposable_income",
                "months_of_runway",
                "delinquency_rate",
                "has_prior_default",
                "credit_history_years",
                "accounts_per_year",
                "active_loan_ratio",
                "employment_stability",
                "income_per_dependent",
            ],
            "categorical": [
                "gender",
                "marital_status",
                "education",
                "employment_type",
                "loan_type",
                "loan_purpose",
            ],
            "ordinal": ["utilization_band", "age_band", "tenure_band", "income_band"],
        },
    }


def _multitable_fixture(n_customers: int = 40) -> dict[str, pd.DataFrame]:
    ages = [22 + i for i in range(n_customers)]
    employment_years = [1.0 + 0.5 * i for i in range(n_customers)]
    annual_income = [300000.0 + 20000 * i for i in range(n_customers)]
    customer_ids = [f"C{i:03d}" for i in range(n_customers)]
    loan_ids = [f"L{i:03d}" for i in range(n_customers)]
    monthly_income = [a / 12 for a in annual_income]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": ["MALE", "FEMALE"] * (n_customers // 2),
            "marital_status": ["SINGLE", "MARRIED"] * (n_customers // 2),
            "dependents": [0, 1] * (n_customers // 2),
            "education": ["GRADUATE"] * n_customers,
            "employment_type": ["SALARIED"] * n_customers,
            "employment_years": employment_years,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": [1, 2, 3] * (n_customers // 3) + [1] * (n_customers % 3),
        }
    )
    application_dates = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=5 * i)
        for i in range(n_customers)
    ]
    loan_applications = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "customer_id": customer_ids,
            "loan_type": ["PERSONAL"] * n_customers,
            "loan_amount": [100000.0 + 5000 * i for i in range(n_customers)],
            "loan_tenure_months": [24] * n_customers,
            "interest_rate": [12.0] * n_customers,
            "loan_purpose": ["OTHER"] * n_customers,
            "application_date": application_dates,
            "decision_date": [d + pd.Timedelta(days=3) for d in application_dates],
            "status": ["APPROVED"] * n_customers,
        }
    )
    as_of_dates = [d - pd.Timedelta(days=5) for d in application_dates]
    financial_profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "monthly_income": monthly_income,
            "monthly_expenses": [m * 0.3 for m in monthly_income],
            "existing_loan_count": [0] * n_customers,
            "existing_loan_amount": [0.0] * n_customers,
            "monthly_emi": [m * 0.1 for m in monthly_income],
            "savings_balance": [m * 6 for m in monthly_income],
            "total_assets": [a * 0.5 for a in annual_income],
            "total_liabilities": [0.0] * n_customers,
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "credit_history_months": [24 + 3 * i for i in range(n_customers)],
            "num_credit_accounts": [2] * n_customers,
            "total_credit_limit": [200000.0] * n_customers,
            "total_outstanding": [50000.0] * n_customers,
            "credit_utilization": [0.25] * n_customers,
            "previous_defaults": [0] * n_customers,
            "late_payments_12m": [0] * n_customers,
            "missed_payments_12m": [0] * n_customers,
            "active_loans": [1] * n_customers,
            "closed_loans": [1] * n_customers,
        }
    )
    loan_outcomes = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "default_12m": [i % 2 for i in range(n_customers)],
            "outcome_observed_date": [
                d + pd.DateOffset(months=12) for d in application_dates
            ],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def _build_loaded_model(model_id: str) -> engine._LoadedModel:
    tables = _multitable_fixture()
    features_config = _features_config()
    cleaning_config = load_rule_config("config/validation_rules.yaml")
    with open("config/scoring.yaml", encoding="utf-8") as f:
        scoring_config = yaml.safe_load(f)

    pipeline = build_feature_pipeline(features_config, cleaning_config)
    train_ids, val_ids, _test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)

    merge_step = pipeline.named_steps["cleaning_and_merge"]
    train_merged = merge_step.fit_transform(train_tables)
    val_merged = merge_step.transform(val_tables)
    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    y_val = align_target(val_merged, val_tables["loan_outcomes"])
    pipeline[1:].fit(train_merged, y_train)

    preprocess = pipeline.named_steps["preprocess"]
    ratios_behavioural = pipeline[1:-1]
    feature_names = list(preprocess.get_feature_names_out())

    train_frame = ratios_behavioural.transform(train_merged)
    X_train = pd.DataFrame(preprocess.transform(train_frame), columns=feature_names)
    val_frame = ratios_behavioural.transform(val_merged)
    X_val = pd.DataFrame(preprocess.transform(val_frame), columns=feature_names)

    base = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    # cv=2 (not the production default of 5): this tiny fixture's validation
    # split doesn't have 5 members of each class. Calibration numerics are
    # covered by tests/test_models.py; this is just a stand-in model for
    # exercising the engine's plumbing.
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid", cv=2)
    calibrated.fit(X_val, y_val)

    background = X_train.sample(n=min(20, len(X_train)), random_state=42)
    benchmarks = shap_explainer.build_portfolio_benchmarks(
        train_frame.reset_index(drop=True),
        features_config["feature_columns"]["numeric"],
        features_config["feature_columns"]["categorical"],
        features_config["feature_columns"]["ordinal"],
    )
    explainer = shap_explainer.get_cached_explainer(
        model_id, "logistic_regression", base, background
    )

    scorecard_config = scorecard.ScorecardConfig.from_config(scoring_config)
    risk_bands = categories.load_risk_bands(scoring_config)
    policy = recommendation.RecommendationPolicy.from_config(
        scoring_config, chosen_threshold=0.3
    )
    columns = features_config["feature_columns"]

    return engine._LoadedModel(
        model_id=model_id,
        model_version="0.0.1-test",
        algorithm="logistic_regression",
        calibrated_model=calibrated,
        ratios_behavioural=ratios_behavioural,
        preprocess=preprocess,
        feature_names=feature_names,
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"],
        ordinal_columns=columns["ordinal"],
        scorecard_config=scorecard_config,
        risk_bands=risk_bands,
        recommendation_policy=policy,
        explainer=explainer,
        benchmarks=benchmarks,
        top_k=5,
    )


@pytest.fixture
def loaded_model():
    model_row = registry.register_model(
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="fixture-version",
        feature_list=["dummy"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.3},
        mlflow_run_id="test-run",
        artifact_path="unused-in-test",
    )
    built = _build_loaded_model(model_row["model_id"])
    engine._MODEL_CACHE = built
    yield built
    engine.reload_active_model()


def _sample_raw_input(**overrides) -> dict:
    base = {
        "customer_id": "CUST-TEST-001",
        "loan_id": "LOAN-TEST-001",
        "age": 35,
        "gender": "MALE",
        "marital_status": "MARRIED",
        "dependents": 1,
        "education": "GRADUATE",
        "employment_type": "SALARIED",
        "employment_years": 8.0,
        "annual_income": 600000.0,
        "city_tier": 1,
        "loan_type": "PERSONAL",
        "loan_amount": 200000.0,
        "loan_tenure_months": 36,
        "interest_rate": 12.0,
        "loan_purpose": "OTHER",
        "monthly_income": 50000.0,
        "monthly_expenses": 15000.0,
        "existing_loan_count": 0,
        "existing_loan_amount": 0.0,
        "monthly_emi": 0.0,
        "savings_balance": 100000.0,
        "total_assets": 500000.0,
        "total_liabilities": 0.0,
        "credit_history_months": 60,
        "num_credit_accounts": 3,
        "total_credit_limit": 300000.0,
        "total_outstanding": 50000.0,
        "previous_defaults": 0,
        "late_payments_12m": 0,
        "missed_payments_12m": 0,
        "active_loans": 1,
        "closed_loans": 2,
    }
    base.update(overrides)
    return base


# --- input validation --------------------------------------------------


def test_validate_input_accepts_a_valid_application() -> None:
    validated = engine._validate_input(_sample_raw_input())
    assert validated.customer_id == "CUST-TEST-001"
    assert validated.loan_type == "PERSONAL"


def test_validate_input_rejects_invalid_loan_type() -> None:
    with pytest.raises(ScoringInputError):
        engine._validate_input(_sample_raw_input(loan_type="NOT_A_TYPE"))


def test_validate_input_rejects_negative_income() -> None:
    with pytest.raises(ScoringInputError):
        engine._validate_input(_sample_raw_input(annual_income=-100.0))


def test_validate_input_rejects_zero_loan_amount() -> None:
    with pytest.raises(ScoringInputError):
        engine._validate_input(_sample_raw_input(loan_amount=0.0))


def test_validate_input_rejects_missing_required_field() -> None:
    payload = _sample_raw_input()
    del payload["annual_income"]
    with pytest.raises(ScoringInputError):
        engine._validate_input(payload)


def test_build_point_in_time_frame_has_expected_columns() -> None:
    validated = engine._validate_input(_sample_raw_input())
    frame = engine._build_point_in_time_frame(validated)
    assert len(frame) == 1
    assert set(frame.columns) == set(engine._FRAME_COLUMNS)
    assert frame.iloc[0]["annual_income"] == 600000.0


# --- score_application end to end --------------------------------------


def test_score_application_returns_complete_result(loaded_model) -> None:
    result = engine.score_application(_sample_raw_input(), persist=False)
    assert result.loan_id == "LOAN-TEST-001"
    assert result.customer_id == "CUST-TEST-001"
    assert result.model_id == loaded_model.model_id
    assert 0.0 <= result.default_probability <= 1.0
    assert 300 <= result.credit_score <= 900
    assert result.risk_category in {
        "VERY_LOW",
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH",
    }
    assert result.recommendation in {"APPROVE", "REVIEW", "REJECT"}
    assert result.triggered_rules
    assert isinstance(result.top_risk_factors, list)
    assert isinstance(result.top_positive_factors, list)
    for row in result.top_risk_factors + result.top_positive_factors:
        assert {"feature", "contribution", "reason"} <= row.keys()
    assert result.latency_ms >= 0
    assert result.latency_ms < 500


def test_score_application_generates_loan_id_when_missing(loaded_model) -> None:
    payload = _sample_raw_input()
    del payload["loan_id"]
    result = engine.score_application(payload, persist=False)
    assert result.loan_id.startswith("SIM-")


def test_score_application_persists_exactly_one_prediction_row(loaded_model) -> None:
    result = engine.score_application(_sample_raw_input(loan_id="LOAN-PERSIST-001"))
    rows = PredictionRepository().fetch_dataframe(
        filters={"loan_id": "LOAN-PERSIST-001"}
    )
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["customer_id"] == "CUST-TEST-001"
    assert row["model_id"] == result.model_id
    assert row["recommendation"] == result.recommendation
    assert int(row["credit_score"]) == result.credit_score


def test_score_application_persist_false_writes_nothing(loaded_model) -> None:
    engine.score_application(
        _sample_raw_input(loan_id="LOAN-NO-PERSIST"), persist=False
    )
    rows = PredictionRepository().fetch_dataframe(
        filters={"loan_id": "LOAN-NO-PERSIST"}
    )
    assert len(rows) == 0


def test_reload_active_model_clears_the_cache() -> None:
    engine._MODEL_CACHE = object()  # type: ignore[assignment]
    engine.reload_active_model()
    assert engine._MODEL_CACHE is None


def test_is_model_loaded_reflects_cache_state() -> None:
    engine._MODEL_CACHE = None
    assert engine.is_model_loaded() is False
    engine._MODEL_CACHE = object()  # type: ignore[assignment]
    assert engine.is_model_loaded() is True
    engine.reload_active_model()


def test_explain_application_returns_full_per_feature_breakdown(loaded_model) -> None:
    detailed = engine.explain_application(_sample_raw_input())
    assert isinstance(detailed, engine.DetailedExplanation)
    assert detailed.result.model_id == loaded_model.model_id
    assert isinstance(detailed.shap_base_value, float)
    # Every logical feature (numeric + categorical + ordinal) should be
    # represented, not just the top-k ScoringResult carries.
    n_expected = (
        len(loaded_model.numeric_columns)
        + len(loaded_model.categorical_columns)
        + len(loaded_model.ordinal_columns)
    )
    assert len(detailed.contributions_by_feature) == n_expected
    assert set(detailed.raw_features) >= set(loaded_model.numeric_columns)
    assert detailed.benchmarks == loaded_model.benchmarks


def test_explain_application_does_not_persist(loaded_model) -> None:
    payload = _sample_raw_input(loan_id="LOAN-EXPLAIN-NO-PERSIST")
    engine.explain_application(payload)
    rows = PredictionRepository().fetch_dataframe(
        filters={"loan_id": "LOAN-EXPLAIN-NO-PERSIST"}
    )
    assert len(rows) == 0
