"""Unit tests for creditguard.features.leakage: forbidden-feature enforcement,
the point-in-time join, and the target-correlation screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import creditguard.features.leakage as leakage_module
from creditguard.features.leakage import (
    LeakageError,
    assert_no_leakage,
    check_target_correlation,
    point_in_time_join,
)


def test_assert_no_leakage_raises_on_named_forbidden_column() -> None:
    with pytest.raises(LeakageError, match="default_12m"):
        assert_no_leakage(["age", "income", "default_12m"])


@pytest.mark.parametrize(
    "column",
    [
        "future_income",
        "credit_limit_after_decision",
        "post_disbursement_fee",
        "outcome_status",
        "actual_default_date",
        "repayment_schedule",
        "charge_off_amount",
        "decision_date",
        "status",
        "days_past_due",
        "recovery_amount",
        "collection_status",
        "write_off_amount",
        "future_missed_payments",
    ],
)
def test_assert_no_leakage_raises_on_every_named_and_pattern_offender(
    column: str,
) -> None:
    with pytest.raises(LeakageError):
        assert_no_leakage(["age", column])


def test_assert_no_leakage_passes_clean_feature_names() -> None:
    # Includes post_loan_dti deliberately: it starts with "post_" but is not
    # leakage (see leakage.py's FORBIDDEN_PATTERNS comment for why the
    # pattern is `^post_disbursement_`, not a bare `^post_`).
    assert_no_leakage(
        ["age", "dti", "post_loan_dti", "credit_utilization", "income_band"]
    )


def test_leakage_guard_genuinely_fails_when_a_leaky_column_is_selected() -> None:
    """Acceptance-criterion demonstration: temporarily widen what
    point_in_time_join selects from loan_applications to include a forbidden
    column, and confirm the guard actually fires -- it isn't just present in
    the code and untested.
    """
    tables = _tiny_tables()
    tables["loan_applications"] = tables["loan_applications"].copy()
    tables["loan_applications"]["default_12m"] = 0

    original_loan_columns = leakage_module._LOAN_COLUMNS
    leakage_module._LOAN_COLUMNS = [*original_loan_columns, "default_12m"]
    try:
        with pytest.raises(LeakageError, match="default_12m"):
            point_in_time_join(
                tables["customers"],
                tables["loan_applications"],
                tables["financial_profiles"],
                tables["credit_history"],
            )
    finally:
        leakage_module._LOAN_COLUMNS = original_loan_columns


def test_point_in_time_join_never_selects_a_later_snapshot() -> None:
    """Fixture with a deliberately later record: it must never be picked."""
    customers = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "age": [30],
            "gender": ["MALE"],
            "marital_status": ["SINGLE"],
            "dependents": [0],
            "education": ["GRADUATE"],
            "employment_type": ["SALARIED"],
            "employment_years": [5.0],
            "annual_income": [600000.0],
            "city_tier": [1],
        }
    )
    loan_applications = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "loan_type": ["PERSONAL"],
            "loan_amount": [100000.0],
            "loan_tenure_months": [24],
            "interest_rate": [12.0],
            "loan_purpose": ["OTHER"],
            "application_date": ["2024-06-01"],
        }
    )
    # Three snapshots for the same customer: one correctly before the
    # application, one exactly on it, and one deliberately AFTER it.
    financial_profiles = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1"],
            "as_of_date": ["2024-05-01", "2024-06-01", "2024-09-01"],
            "monthly_income": [40000.0, 41000.0, 999999.0],
            "monthly_expenses": [10000.0, 10500.0, 1.0],
            "existing_loan_count": [0, 0, 0],
            "existing_loan_amount": [0.0, 0.0, 0.0],
            "monthly_emi": [0.0, 0.0, 0.0],
            "savings_balance": [50000.0, 51000.0, 999999.0],
            "total_assets": [200000.0, 201000.0, 999999.0],
            "total_liabilities": [0.0, 0.0, 0.0],
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C1"],
            "as_of_date": ["2024-05-01", "2024-06-01", "2024-09-01"],
            "credit_history_months": [24, 25, 999],
            "num_credit_accounts": [2, 2, 99],
            "total_credit_limit": [100000.0, 100000.0, 100000.0],
            "total_outstanding": [10000.0, 10000.0, 10000.0],
            "previous_defaults": [0, 0, 0],
            "late_payments_12m": [0, 0, 0],
            "missed_payments_12m": [0, 0, 0],
            "active_loans": [1, 1, 1],
            "closed_loans": [0, 0, 0],
        }
    )

    merged = point_in_time_join(
        customers, loan_applications, financial_profiles, credit_history
    )

    assert len(merged) == 1
    # The latest ON-OR-BEFORE application_date snapshot is the 2024-06-01 one,
    # never the 2024-09-01 one (whose absurd values -- 999999, 999 -- would be
    # an obvious tell if it leaked through).
    assert merged.loc[0, "financial_as_of_date"] == pd.Timestamp("2024-06-01")
    assert merged.loc[0, "credit_as_of_date"] == pd.Timestamp("2024-06-01")
    assert merged.loc[0, "monthly_income"] == 41000.0
    assert merged.loc[0, "credit_history_months"] == 25


def test_point_in_time_join_excludes_customers_with_no_eligible_snapshot() -> None:
    """A snapshot dated entirely after application_date leaves NaN, not a pick."""
    customers = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "age": [30],
            "gender": ["MALE"],
            "marital_status": ["SINGLE"],
            "dependents": [0],
            "education": ["GRADUATE"],
            "employment_type": ["SALARIED"],
            "employment_years": [5.0],
            "annual_income": [600000.0],
            "city_tier": [1],
        }
    )
    loan_applications = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "loan_type": ["PERSONAL"],
            "loan_amount": [100000.0],
            "loan_tenure_months": [24],
            "interest_rate": [12.0],
            "loan_purpose": ["OTHER"],
            "application_date": ["2024-01-01"],
        }
    )
    financial_profiles = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "as_of_date": ["2024-06-01"],  # after application_date
            "monthly_income": [40000.0],
            "monthly_expenses": [10000.0],
            "existing_loan_count": [0],
            "existing_loan_amount": [0.0],
            "monthly_emi": [0.0],
            "savings_balance": [50000.0],
            "total_assets": [200000.0],
            "total_liabilities": [0.0],
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "as_of_date": ["2024-06-01"],
            "credit_history_months": [24],
            "num_credit_accounts": [2],
            "total_credit_limit": [100000.0],
            "total_outstanding": [10000.0],
            "previous_defaults": [0],
            "late_payments_12m": [0],
            "missed_payments_12m": [0],
            "active_loans": [1],
            "closed_loans": [0],
        }
    )

    merged = point_in_time_join(
        customers, loan_applications, financial_profiles, credit_history
    )
    assert len(merged) == 1
    assert pd.isna(merged.loc[0, "monthly_income"])
    assert pd.isna(merged.loc[0, "financial_as_of_date"])


def test_check_target_correlation_flags_a_perfectly_correlated_feature() -> None:
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    X = pd.DataFrame(
        {
            "leaky": y.astype(float),
            "noise": [1.0, 5.0, 2.0, 9.0, 3.0, 1.0, 4.0, 8.0, 2.0, 6.0],
        }
    )
    with pytest.warns(UserWarning, match="leaky"):
        offenders = check_target_correlation(X, y, threshold=0.95)
    assert offenders == ["leaky"]


def test_check_target_correlation_does_not_flag_unrelated_features() -> None:
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=200))
    X = pd.DataFrame({"noise": rng.normal(size=200)})
    offenders = check_target_correlation(X, y, threshold=0.95)
    assert offenders == []


def _tiny_tables() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "age": [30],
            "gender": ["MALE"],
            "marital_status": ["SINGLE"],
            "dependents": [0],
            "education": ["GRADUATE"],
            "employment_type": ["SALARIED"],
            "employment_years": [5.0],
            "annual_income": [600000.0],
            "city_tier": [1],
        }
    )
    loan_applications = pd.DataFrame(
        {
            "loan_id": ["L1"],
            "customer_id": ["C1"],
            "loan_type": ["PERSONAL"],
            "loan_amount": [100000.0],
            "loan_tenure_months": [24],
            "interest_rate": [12.0],
            "loan_purpose": ["OTHER"],
            "application_date": ["2024-06-01"],
        }
    )
    financial_profiles = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "as_of_date": ["2024-05-01"],
            "monthly_income": [40000.0],
            "monthly_expenses": [10000.0],
            "existing_loan_count": [0],
            "existing_loan_amount": [0.0],
            "monthly_emi": [0.0],
            "savings_balance": [50000.0],
            "total_assets": [200000.0],
            "total_liabilities": [0.0],
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": ["C1"],
            "as_of_date": ["2024-05-01"],
            "credit_history_months": [24],
            "num_credit_accounts": [2],
            "total_credit_limit": [100000.0],
            "total_outstanding": [10000.0],
            "previous_defaults": [0],
            "late_payments_12m": [0],
            "missed_payments_12m": [0],
            "active_loans": [1],
            "closed_loans": [0],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
    }
