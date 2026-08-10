"""Shared pytest fixtures: test environment defaults and database setup/teardown."""

from __future__ import annotations

import os

# Env vars must be set before creditguard modules are imported anywhere, since
# Settings validation happens on first use.
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "5432")
os.environ.setdefault("DB_NAME", "creditguard_test")
os.environ.setdefault("DB_USER", "creditguard")
os.environ.setdefault("DB_PASSWORD", "creditguard")
os.environ.setdefault("DB_SCHEMA", "public")
os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ENV", "test")
os.environ.setdefault("SEED", "42")

import uuid  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from datetime import UTC, datetime  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.frozen import FrozenEstimator  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sqlalchemy import text  # noqa: E402

from creditguard.api.dependencies import reset_rate_limiter  # noqa: E402
from creditguard.api.middleware import metrics_store  # noqa: E402
from creditguard.db.engine import get_engine  # noqa: E402
from creditguard.db.init_db import apply_schema  # noqa: E402
from creditguard.db.models import Base  # noqa: E402
from creditguard.explain import shap_explainer  # noqa: E402
from creditguard.features.build import (
    align_target,
    filter_tables,
    temporal_split,
)  # noqa: E402
from creditguard.features.pipeline import build_feature_pipeline  # noqa: E402
from creditguard.models import registry  # noqa: E402
from creditguard.scoring import (
    categories,
    engine,
    recommendation,
    scorecard,
)  # noqa: E402
from creditguard.validation.engine import load_rule_config  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database() -> None:
    """Apply the schema once per test session before any test runs."""
    apply_schema()


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    """Truncate all application tables before each test so tests stay isolated."""
    engine_ = get_engine()
    table_names = [table.name for table in Base.metadata.sorted_tables]
    with engine_.begin() as conn:
        quoted = ", ".join(f'"{name}"' for name in table_names)
        conn.execute(text(f"TRUNCATE {quoted} RESTART IDENTITY CASCADE"))
    yield


@pytest.fixture(autouse=True)
def _reset_api_state() -> None:
    """Reset the API's in-memory rate limiter and metrics store before
    every test, so unrelated tests sharing this process don't interfere
    with each other's request counts / latency samples.
    """
    reset_rate_limiter()
    metrics_store.reset()


# -- shared synthetic model fixture, for Phase 7/8 tests that need a fitted
# model without depending on the real ~67k-row dataset on disk --------------


def api_test_features_config() -> dict:
    """The same `config/features.yaml` shape used across Phase 7/8 tests
    that build a small synthetic model, kept here once rather than
    duplicated per test file.
    """
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


def api_test_multitable_fixture(n_customers: int = 40) -> dict[str, pd.DataFrame]:
    """A small, internally-consistent multi-table dataset (same shape as
    Phase 4's real tables) big enough to fit a real (if tiny) logistic
    regression and calibrate it, without touching the real ~67k-row
    dataset on disk.
    """
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


def build_synthetic_loaded_model(model_id: str) -> engine._LoadedModel:
    """A real (if tiny) fitted `creditguard.scoring.engine._LoadedModel` --
    fitted pipeline, calibrated logistic regression, SHAP explainer,
    scorecard/recommendation config -- for tests to inject directly into
    `engine._MODEL_CACHE`, bypassing the database/dataset-driven
    `_load_active_model`.
    """
    tables = api_test_multitable_fixture()
    features_config = api_test_features_config()
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
    # cv=2 (not the production default of 5): this tiny fixture's
    # validation split doesn't have 5 members of each class.
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
def loaded_model() -> Iterator[engine._LoadedModel]:
    """Build a synthetic fitted model, register a matching `model_registry`
    row (needed for the `predictions.model_id` foreign key, and so
    `GET /api/v1/model/info`'s `feature_count` reflects the real encoded
    feature width), and inject it directly into `engine._MODEL_CACHE` so
    `score_application`/`explain_application` work without needing the
    real dataset.
    """
    model_id = f"test-logistic-regression-{uuid.uuid4().hex[:12]}"
    built = build_synthetic_loaded_model(model_id)
    registry.register_model(
        model_id=model_id,
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="fixture-version",
        feature_list=built.feature_names,
        hyperparameters={},
        metrics={"chosen_threshold": 0.3},
        mlflow_run_id="test-run",
        artifact_path="unused-in-test",
    )
    engine._MODEL_CACHE = built
    yield built
    engine.reload_active_model()


@pytest.fixture
def api_client(loaded_model):
    """A `TestClient` for the FastAPI app, with `loaded_model` already
    injected before the app's `lifespan` runs -- `engine.warm_up()` sees
    `_MODEL_CACHE` already populated and is a no-op, so no test ever
    triggers the real database/dataset-driven model load.

    `raise_server_exceptions=False`: TestClient's default
    (`True`) re-raises any exception that reaches `ServerErrorMiddleware`
    in the *test process itself*, bypassing `creditguard.api.errors`'
    registered handlers entirely -- a debugging convenience for seeing a
    real traceback while developing route code, not what a real client
    (or uvicorn) sees. Tests that exercise the actual 500/503 response
    contract need it off.
    """
    from fastapi.testclient import TestClient

    from creditguard.api.main import app

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


API_KEY = "test-api-key"


@pytest.fixture
def _example_payload() -> dict:
    """A valid `PredictionRequest` payload -- the same example the OpenAPI
    schema shows in `/docs`, imported lazily so importing this module
    doesn't require the API package for tests that don't need it.
    """
    from creditguard.api.schemas import _EXAMPLE_PAYLOAD

    return dict(_EXAMPLE_PAYLOAD)
