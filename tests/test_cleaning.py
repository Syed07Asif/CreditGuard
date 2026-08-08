"""Tests for creditguard.validation.cleaning: idempotency, no-leakage, drop rules."""

from __future__ import annotations

import pandas as pd

from creditguard.validation.cleaning import DataCleaner
from creditguard.validation.engine import build_registry, load_rule_config, run


def _rule_config() -> dict:
    return load_rule_config("config/validation_rules.yaml")


def _clean_multitable_dataset() -> dict[str, pd.DataFrame]:
    """A tiny, internally-consistent dataset that satisfies every configured rule."""
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "age": [35, 40],
            "gender": ["MALE", "FEMALE"],
            "marital_status": ["MARRIED", "SINGLE"],
            "dependents": [1, 0],
            "education": ["GRADUATE", "POSTGRADUATE"],
            "employment_type": ["SALARIED", "SALARIED"],
            "employment_years": [10.0, 15.0],
            "annual_income": [600000.0, 900000.0],
            "monthly_income": [50000.0, 75000.0],
            "city_tier": [1, 2],
        }
    )
    loan_applications = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "customer_id": ["C1", "C2"],
            "loan_type": ["PERSONAL", "AUTO"],
            "loan_amount": [200000.0, 500000.0],
            "loan_tenure_months": [36, 48],
            "interest_rate": [12.5, 9.0],
            "loan_purpose": ["OTHER", "VEHICLE"],
            "application_date": ["2024-01-01", "2024-02-01"],
            "decision_date": ["2024-01-05", "2024-02-04"],
            "status": ["APPROVED", "APPROVED"],
        }
    )
    financial_profiles = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "as_of_date": ["2023-12-25", "2024-01-25"],
            "monthly_income": [50000.0, 75000.0],
            "monthly_expenses": [20000.0, 30000.0],
            "existing_loan_count": [1, 0],
            "existing_loan_amount": [50000.0, 0.0],
            "monthly_emi": [5000.0, 0.0],
            "savings_balance": [100000.0, 200000.0],
            "total_assets": [300000.0, 500000.0],
            "total_liabilities": [50000.0, 0.0],
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "as_of_date": ["2023-12-25", "2024-01-25"],
            "credit_history_months": [60, 100],
            "num_credit_accounts": [3, 5],
            "total_credit_limit": [200000.0, 400000.0],
            "total_outstanding": [80000.0, 100000.0],
            "credit_utilization": [0.4, 0.25],
            "previous_defaults": [0, 0],
            "late_payments_12m": [0, 1],
            "missed_payments_12m": [0, 0],
            "active_loans": [1, 0],
            "closed_loans": [2, 3],
        }
    )
    loan_outcomes = pd.DataFrame(
        {
            "loan_id": ["L1", "L2"],
            "default_12m": [0, 0],
            "outcome_observed_date": ["2025-01-01", "2025-02-01"],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def test_cleaning_is_idempotent_on_already_clean_data() -> None:
    tables = _clean_multitable_dataset()
    cleaner = DataCleaner(_rule_config())
    once = cleaner.fit_transform(tables)
    twice = cleaner.transform(once)

    for table in once:
        pd.testing.assert_frame_equal(
            once[table].reset_index(drop=True), twice[table].reset_index(drop=True)
        )


def test_imputer_uses_stored_train_median_not_recomputed_on_test() -> None:
    common_columns = {
        "monthly_income": 50000.0,
        "monthly_expenses": 20000.0,
        "existing_loan_count": 1,
        "existing_loan_amount": 50000.0,
        "monthly_emi": 5000.0,
        "savings_balance": 100000.0,
        "total_liabilities": 50000.0,
    }

    train = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "as_of_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "total_assets": [100.0, 200.0, None],
            **{k: [v] * 3 for k, v in common_columns.items()},
        }
    )
    test = pd.DataFrame(
        {
            "customer_id": ["C4", "C5"],
            "as_of_date": ["2024-02-01", "2024-02-02"],
            "total_assets": [1000.0, None],
            **{k: [v] * 2 for k, v in common_columns.items()},
        }
    )

    cleaner = DataCleaner(_rule_config())
    cleaner.fit({"financial_profiles": train})
    assert cleaner.medians_["financial_profiles"]["total_assets"] == 150.0

    cleaned_test = cleaner.transform({"financial_profiles": test})["financial_profiles"]
    imputed_row = cleaned_test.loc[cleaned_test["customer_id"] == "C5"].iloc[0]

    # Stored train median (150.0), NOT the test-only median (which would be 1000.0).
    assert imputed_row["total_assets"] == 150.0
    assert bool(imputed_row["total_assets_was_missing"]) is True


def test_clean_output_has_zero_remaining_error_violations() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C3"],
            "age": [200, 30, 35, 35],  # C1: impossible age
            "gender": ["MALE", "FEMALE", "MALE", "MALE"],
            "marital_status": ["MARRIED", "SINGLE", "SINGLE", "SINGLE"],
            "dependents": [1, 0, 2, 2],
            "education": ["GRADUATE", "GRADUATE", "GRADUATE", "GRADUATE"],
            "employment_type": ["SALARIED", "SALARIED", "SALARIED", "SALARIED"],
            "employment_years": [10.0, 5.0, 8.0, 8.0],
            "annual_income": [600000.0, -500000.0, 720000.0, 720000.0],  # C2: negative
            "monthly_income": [50000.0, 0.0, 60000.0, 60000.0],
            "city_tier": [1, 2, 3, 3],
        }
    )
    tables = {"customers": customers}

    cleaner = DataCleaner(_rule_config())
    cleaned = cleaner.fit_transform(tables)
    cleaned_customers = cleaned["customers"]

    # Duplicate C3 row deduplicated away.
    assert len(cleaned_customers) == 3
    # Impossible age clipped into bounds, not dropped.
    c1 = cleaned_customers.loc[cleaned_customers["customer_id"] == "C1"].iloc[0]
    assert c1["age"] == 100
    # Negative income repaired (clipped to >= 0, then possibly winsorised), not dropped.
    c2 = cleaned_customers.loc[cleaned_customers["customer_id"] == "C2"].iloc[0]
    assert c2["annual_income"] >= 0.0

    registry = build_registry(_rule_config())
    result = run(cleaned, registry, dataset_version="post-clean")
    error_violations = result.violations[result.violations["severity"] == "ERROR"]
    assert error_violations.empty, error_violations
