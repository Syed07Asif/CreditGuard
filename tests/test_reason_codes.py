"""Tests for creditguard.explain.reason_codes: template coverage, sentence
rendering, and the top-k reason-code builder.
"""

from __future__ import annotations

import pytest

from creditguard.explain.reason_codes import (
    FEATURE_REASON_SPECS,
    ReasonCodeCoverageError,
    assert_full_coverage,
    generate_reason_codes,
    render_reason,
)

# The full logical feature set from config/features.yaml's feature_columns
# (43 numeric + 6 categorical + 4 ordinal = 53).
ALL_FEATURES = (
    [
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
    ]
    + [
        "gender",
        "marital_status",
        "education",
        "employment_type",
        "loan_type",
        "loan_purpose",
    ]
    + ["utilization_band", "age_band", "tenure_band", "income_band"]
)


def test_all_53_features_are_covered() -> None:
    assert len(ALL_FEATURES) == 53
    assert_full_coverage(ALL_FEATURES)  # must not raise


def test_missing_feature_raises_coverage_error() -> None:
    with pytest.raises(ReasonCodeCoverageError, match="made_up_feature"):
        assert_full_coverage([*ALL_FEATURES, "made_up_feature"])


def test_every_registered_feature_has_nonempty_both_direction_phrases() -> None:
    for name, spec in FEATURE_REASON_SPECS.items():
        assert spec.display, name
        assert spec.high_value_phrase, name
        assert spec.low_value_phrase, name


@pytest.mark.parametrize(
    "value,benchmark,expected_word",
    [
        (0.62, 0.38, "well above"),
        (0.40, 0.38, "above"),
        (0.38, 0.38, "in line with"),
        (0.37, 0.38, "in line with"),
        (0.20, 0.38, "below"),
        (0.10, 0.38, "well below"),
    ],
)
def test_comparison_word_thresholds(
    value: float, benchmark: float, expected_word: str
) -> None:
    from creditguard.explain.reason_codes import _comparison_word

    assert _comparison_word(value, benchmark) == expected_word


def test_render_reason_numeric_matches_documented_example_style() -> None:
    sentence = render_reason(
        "dti", 0.62, contribution=0.05, benchmark={"type": "numeric", "median": 0.38}
    )
    assert "Debt-to-income ratio" in sentence
    assert "62%" in sentence
    assert "38%" in sentence
    assert "well above" in sentence
    assert "increasing the estimated risk" in sentence


def test_render_reason_direction_follows_shap_sign_not_raw_value() -> None:
    # Same elevated value, but this time SHAP says it REDUCED risk for this
    # row -- the factual clause must stay the same, only the direction changes.
    sentence = render_reason(
        "dti", 0.62, contribution=-0.05, benchmark={"type": "numeric", "median": 0.38}
    )
    assert "well above" in sentence  # factual clause unchanged
    assert "reducing the estimated risk" in sentence


def test_render_reason_rate_percent_does_not_rescale() -> None:
    # interest_rate is stored in percentage-point units already (9.5 means
    # 9.5%), unlike dti/credit_utilization which are 0-1 fractions -- must
    # NOT be multiplied by 100 like "percent" format does.
    sentence = render_reason(
        "interest_rate",
        9.5,
        contribution=-0.01,
        benchmark={"type": "numeric", "median": 11.9},
    )
    assert "9.5%" in sentence
    assert "950%" not in sentence


def test_render_reason_count_zero_present_and_absent() -> None:
    present = render_reason(
        "previous_defaults",
        2,
        contribution=0.1,
        benchmark={"type": "numeric", "median": 0.0},
    )
    assert "history of previous defaults" in present
    absent = render_reason(
        "previous_defaults",
        0,
        contribution=-0.1,
        benchmark={"type": "numeric", "median": 0.0},
    )
    assert "No previous defaults on record" in absent


def test_render_reason_flag_format() -> None:
    present = render_reason(
        "has_prior_default",
        1,
        contribution=0.2,
        benchmark={"type": "numeric", "median": 0.0},
    )
    assert "prior default on record" in present
    absent = render_reason(
        "has_prior_default",
        0,
        contribution=-0.2,
        benchmark={"type": "numeric", "median": 0.0},
    )
    assert "no prior default" in absent.lower()


def test_render_reason_category_matches_and_differs() -> None:
    matches = render_reason(
        "employment_type",
        "SALARIED",
        contribution=-0.02,
        benchmark={"type": "categorical", "mode": "SALARIED"},
    )
    assert "matches" in matches

    differs = render_reason(
        "employment_type",
        "SELF_EMPLOYED",
        contribution=0.02,
        benchmark={"type": "categorical", "mode": "SALARIED"},
    )
    assert "differs from" in differs


def test_render_reason_unknown_feature_raises() -> None:
    with pytest.raises(ReasonCodeCoverageError):
        render_reason("not_a_real_feature", 1, contribution=0.1, benchmark={})


def test_generate_reason_codes_builds_risk_and_positive_lists() -> None:
    raw_features = {"dti": 0.62, "employment_years": 9.0, "previous_defaults": 0}
    benchmarks = {
        "dti": {"type": "numeric", "median": 0.38},
        "employment_years": {"type": "numeric", "median": 6.0},
        "previous_defaults": {"type": "numeric", "median": 0.0},
    }
    top_positive = [("dti", 0.09), ("previous_defaults", 0.01)]
    top_negative = [("employment_years", -0.04)]

    risk_factors, positive_factors = generate_reason_codes(
        top_positive, top_negative, raw_features, benchmarks
    )

    assert [row["feature"] for row in risk_factors] == ["dti", "previous_defaults"]
    assert all(row["contribution"] > 0 for row in risk_factors)
    assert all("reason" in row and row["reason"] for row in risk_factors)

    assert [row["feature"] for row in positive_factors] == ["employment_years"]
    assert positive_factors[0]["contribution"] < 0


def test_generate_reason_codes_respects_top_k_via_input_length() -> None:
    raw_features = {"dti": 0.5}
    benchmarks = {"dti": {"type": "numeric", "median": 0.4}}
    # Caller (creditguard.scoring.engine) is responsible for slicing to
    # top_k before calling; generate_reason_codes just renders what it's given.
    risk_factors, _ = generate_reason_codes(
        [("dti", 0.1)], [], raw_features, benchmarks
    )
    assert len(risk_factors) == 1
