"""Tests for creditguard.features.pipeline / encoders / behavioural: train-only
fitted statistics, deterministic transforms, and stable feature names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditguard.features.behavioural import BehaviouralFeatures
from creditguard.features.build import align_target, filter_tables, temporal_split
from creditguard.features.encoders import build_preprocessor
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.validation.engine import load_rule_config


def _rule_config() -> dict:
    return load_rule_config("config/validation_rules.yaml")


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


def _multitable_fixture(n_customers: int = 10) -> dict[str, pd.DataFrame]:
    """A small, internally-consistent multi-table dataset that satisfies
    every Phase 3 validation rule (so DataCleaner drops nothing), with
    distinct enough age/employment_years/annual_income values that quantile
    bin edges are easy to reason about by hand.
    """
    ages = [22, 25, 28, 31, 34, 37, 40, 45, 50, 60][:n_customers]
    employment_years = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0][
        :n_customers
    ]
    annual_income = [
        300000.0,
        350000.0,
        400000.0,
        500000.0,
        600000.0,
        700000.0,
        800000.0,
        900000.0,
        1000000.0,
        1500000.0,
    ][:n_customers]
    customer_ids = [f"C{i:03d}" for i in range(n_customers)]
    loan_ids = [f"L{i:03d}" for i in range(n_customers)]
    monthly_income = [a / 12 for a in annual_income]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": ["MALE", "FEMALE"] * (n_customers // 2)
            + ["MALE"] * (n_customers % 2),
            "marital_status": ["SINGLE", "MARRIED"] * (n_customers // 2)
            + ["SINGLE"] * (n_customers % 2),
            "dependents": [0, 1] * (n_customers // 2) + [0] * (n_customers % 2),
            "education": ["GRADUATE"] * n_customers,
            "employment_type": ["SALARIED"] * n_customers,
            "employment_years": employment_years,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": [1, 2, 3] * (n_customers // 3) + [1] * (n_customers % 3),
        }
    )

    application_dates = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=10 * i)
        for i in range(n_customers)
    ]
    loan_applications = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "customer_id": customer_ids,
            "loan_type": ["PERSONAL"] * n_customers,
            "loan_amount": [100000.0 + 10000 * i for i in range(n_customers)],
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
            "credit_history_months": [24 + 12 * i for i in range(n_customers)],
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


def test_behavioural_bin_edges_fit_on_train_only_match_manual_quantiles() -> None:
    train = pd.DataFrame(
        {
            "age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 65],
            "employment_years": [1, 2, 5, 8, 10, 12, 15, 18, 20, 25],
            "annual_income": [
                300000,
                350000,
                400000,
                500000,
                600000,
                700000,
                800000,
                900000,
                1000000,
                1200000,
            ],
        }
    )
    bf = BehaviouralFeatures(n_bins=5)
    bf.fit(train)

    expected_age_edges = np.quantile(train["age"].astype(float), np.linspace(0, 1, 6))
    expected_age_edges[0] = -np.inf
    expected_age_edges[-1] = np.inf
    np.testing.assert_allclose(bf.age_edges_, expected_age_edges)

    # A value outside the observed training range still bins into the
    # extreme bucket (via the -inf/+inf substitution), never NaN.
    unseen = pd.Series([10.0, 500.0])
    bands = bf._apply_band(unseen, bf.age_edges_)
    assert "nan" not in bands


def test_preprocessor_numeric_stats_fit_on_train_only() -> None:
    train = pd.DataFrame(
        {"num_a": [1.0, 2.0, 3.0, 4.0, 5.0], "cat_a": ["X", "Y", "X", "Y", "X"]}
    )
    preprocessor = build_preprocessor(
        numeric_columns=["num_a"], categorical_columns=["cat_a"], ordinal_columns=[]
    )
    preprocessor.fit(train)

    scaler = preprocessor.named_transformers_["numeric"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, [train["num_a"].mean()])
    np.testing.assert_allclose(scaler.scale_, [train["num_a"].std(ddof=0)])

    # Transforming different (val) data must not refit the scaler.
    val = pd.DataFrame({"num_a": [100.0, 200.0], "cat_a": ["X", "Y"]})
    preprocessor.transform(val)
    np.testing.assert_allclose(scaler.mean_, [train["num_a"].mean()])
    np.testing.assert_allclose(scaler.scale_, [train["num_a"].std(ddof=0)])


def test_transform_is_deterministic_given_fixed_fit() -> None:
    tables = _multitable_fixture()
    pipeline = build_feature_pipeline(_features_config(), _rule_config())
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    rest = pipeline[1:]

    train_ids, val_ids, _test_ids = temporal_split(
        tables["loan_applications"], train_fraction=0.7, val_fraction=0.15
    )
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)

    train_merged = merge_step.fit_transform(train_tables)
    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    rest.fit(train_merged, y_train)

    val_merged = merge_step.transform(val_tables)
    first = rest.transform(val_merged)
    second = rest.transform(val_merged)
    np.testing.assert_array_equal(first, second)


def test_feature_names_stable_across_independent_runs() -> None:
    tables = _multitable_fixture()
    features_config = _features_config()
    cleaning_config = _rule_config()

    names_per_run = []
    for _ in range(2):
        pipeline = build_feature_pipeline(features_config, cleaning_config)
        merge_step = pipeline.named_steps["cleaning_and_merge"]
        rest = pipeline[1:]
        merged = merge_step.fit_transform(tables)
        y = align_target(merged, tables["loan_outcomes"])
        rest.fit_transform(merged, y)
        names_per_run.append(
            list(pipeline.named_steps["preprocess"].get_feature_names_out())
        )

    assert names_per_run[0] == names_per_run[1]
    assert len(names_per_run[0]) == len(set(names_per_run[0]))  # no duplicate names


def test_pipeline_end_to_end_produces_finite_matrix_with_no_leakage() -> None:
    tables = _multitable_fixture()
    pipeline = build_feature_pipeline(_features_config(), _rule_config())
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    rest = pipeline[1:]

    train_ids, val_ids, test_ids = temporal_split(
        tables["loan_applications"], train_fraction=0.7, val_fraction=0.15
    )
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)
    test_tables = filter_tables(tables, test_ids)

    train_merged = merge_step.fit_transform(train_tables)
    val_merged = merge_step.transform(val_tables)
    test_merged = merge_step.transform(test_tables)

    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    X_train = rest.fit_transform(train_merged, y_train)
    X_val = rest.transform(val_merged)
    X_test = rest.transform(test_merged)

    for X in (X_train, X_val, X_test):
        assert np.all(np.isfinite(X))

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert "default_12m" not in feature_names
    assert "outcome_observed_date" not in feature_names
