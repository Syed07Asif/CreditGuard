"""Unit tests for creditguard.features.ratios: hand-computed values and the
zero-denominator path for every ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.features.ratios import (
    CREDIT_UTILIZATION_CAP,
    DTI_CAP,
    EMI_TO_INCOME_CAP,
    LOAN_TO_INCOME_CAP,
    POST_LOAN_DTI_CAP,
    RATIO_OUTPUT_COLUMNS,
    SAVINGS_TO_INCOME_CAP,
    RatioFeatures,
    compute_credit_utilization,
    compute_disposable_income,
    compute_dti,
    compute_emi_to_income,
    compute_leverage_ratio,
    compute_loan_to_income,
    compute_months_of_runway,
    compute_net_worth,
    compute_post_loan_dti,
    compute_proposed_emi,
    compute_savings_to_income,
)


def test_dti_hand_computed() -> None:
    result = compute_dti(
        monthly_emi=[2000.0], monthly_expenses=[3000.0], monthly_income=[10000.0]
    )
    assert result[0] == 0.5


def test_dti_zero_income_maps_to_cap() -> None:
    result = compute_dti(
        monthly_emi=[2000.0], monthly_expenses=[3000.0], monthly_income=[0.0]
    )
    assert result[0] == DTI_CAP


def test_emi_to_income_hand_computed() -> None:
    result = compute_emi_to_income(monthly_emi=[2000.0], monthly_income=[10000.0])
    assert result[0] == 0.2


def test_emi_to_income_zero_income_maps_to_cap() -> None:
    result = compute_emi_to_income(monthly_emi=[2000.0], monthly_income=[0.0])
    assert result[0] == EMI_TO_INCOME_CAP


def test_credit_utilization_hand_computed() -> None:
    result = compute_credit_utilization(
        total_outstanding=[5000.0], total_credit_limit=[10000.0]
    )
    assert result[0] == 0.5


def test_credit_utilization_zero_limit_maps_to_cap() -> None:
    result = compute_credit_utilization(
        total_outstanding=[5000.0], total_credit_limit=[0.0]
    )
    assert result[0] == CREDIT_UTILIZATION_CAP


def test_loan_to_income_hand_computed() -> None:
    result = compute_loan_to_income(loan_amount=[200000.0], annual_income=[1000000.0])
    assert result[0] == 0.2


def test_loan_to_income_zero_income_maps_to_cap() -> None:
    result = compute_loan_to_income(loan_amount=[200000.0], annual_income=[0.0])
    assert result[0] == LOAN_TO_INCOME_CAP


def test_proposed_emi_hand_computed_standard_amortisation() -> None:
    principal, annual_rate, n = 100000.0, 12.0, 12
    r = annual_rate / 12 / 100
    expected = principal * r * (1 + r) ** n / ((1 + r) ** n - 1)
    result = compute_proposed_emi([principal], [annual_rate], [n])
    assert result[0] == pytest.approx(expected, rel=1e-9)


def test_proposed_emi_zero_rate_uses_equal_principal_limit() -> None:
    # r == 0 hits a 0/0 in the amortisation formula; the analytic limit is P/n.
    result = compute_proposed_emi([120000.0], [0.0], [12])
    assert result[0] == 10000.0


def test_post_loan_dti_hand_computed() -> None:
    result = compute_post_loan_dti(
        monthly_emi=[1000.0],
        proposed_emi=[500.0],
        monthly_expenses=[1500.0],
        monthly_income=[10000.0],
    )
    assert result[0] == 0.3


def test_post_loan_dti_zero_income_maps_to_cap() -> None:
    result = compute_post_loan_dti(
        monthly_emi=[1000.0],
        proposed_emi=[500.0],
        monthly_expenses=[1500.0],
        monthly_income=[0.0],
    )
    assert result[0] == POST_LOAN_DTI_CAP


def test_savings_to_income_hand_computed() -> None:
    result = compute_savings_to_income(
        savings_balance=[50000.0], monthly_income=[10000.0]
    )
    assert result[0] == 5.0


def test_savings_to_income_zero_income_maps_to_cap() -> None:
    result = compute_savings_to_income(savings_balance=[50000.0], monthly_income=[0.0])
    assert result[0] == SAVINGS_TO_INCOME_CAP


def test_net_worth_hand_computed() -> None:
    result = compute_net_worth(total_assets=[500000.0], total_liabilities=[200000.0])
    assert result[0] == 300000.0


def test_net_worth_can_be_negative() -> None:
    # No division involved -- net worth is allowed to go negative.
    result = compute_net_worth(total_assets=[100.0], total_liabilities=[500.0])
    assert result[0] == -400.0


def test_leverage_ratio_hand_computed() -> None:
    result = compute_leverage_ratio(
        total_liabilities=[50000.0], total_assets=[100000.0]
    )
    assert result[0] == 0.5


def test_leverage_ratio_zero_assets_uses_builtin_max_guard() -> None:
    # Formula is total_liabilities / max(total_assets, 1) -- the max(...,1) is
    # the spec's own guard, exercised here with total_assets == 0.
    result = compute_leverage_ratio(total_liabilities=[50000.0], total_assets=[0.0])
    assert result[0] == 50000.0


def test_disposable_income_hand_computed() -> None:
    result = compute_disposable_income(
        monthly_income=[10000.0], monthly_expenses=[3000.0], monthly_emi=[2000.0]
    )
    assert result[0] == 5000.0


def test_months_of_runway_hand_computed() -> None:
    result = compute_months_of_runway(
        savings_balance=[24000.0], monthly_expenses=[2000.0]
    )
    assert result[0] == 12.0


def test_months_of_runway_zero_expenses_uses_builtin_max_guard() -> None:
    # Formula is savings_balance / max(monthly_expenses, 1).
    result = compute_months_of_runway(savings_balance=[5000.0], monthly_expenses=[0.0])
    assert result[0] == 5000.0


def test_ratio_features_transformer_appends_all_11_columns() -> None:
    df = pd.DataFrame(
        {
            "loan_amount": [200000.0],
            "interest_rate": [12.0],
            "loan_tenure_months": [24],
            "monthly_emi": [2000.0],
            "monthly_expenses": [3000.0],
            "monthly_income": [10000.0],
            "total_outstanding": [5000.0],
            "total_credit_limit": [10000.0],
            "annual_income": [1000000.0],
            "savings_balance": [50000.0],
            "total_assets": [500000.0],
            "total_liabilities": [200000.0],
        }
    )
    result = RatioFeatures().fit_transform(df)
    for column in RATIO_OUTPUT_COLUMNS:
        assert column in result.columns
    assert np.isfinite(result[list(RATIO_OUTPUT_COLUMNS)].to_numpy()).all()


def test_ratio_features_get_feature_names_out_appends_to_input() -> None:
    names = RatioFeatures().get_feature_names_out(input_features=["age", "income"])
    assert list(names[:2]) == ["age", "income"]
    assert set(RATIO_OUTPUT_COLUMNS).issubset(set(names))
