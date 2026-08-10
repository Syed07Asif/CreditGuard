"""Tests for creditguard.scoring.recommendation: the APPROVE/REVIEW/REJECT
policy.
"""

from __future__ import annotations

from typing import Any

import pytest

from creditguard.scoring.recommendation import RecommendationPolicy, recommend

HARD_FAILS = {
    "previous_defaults_min": 2,
    "post_loan_dti_max": 0.60,
    "credit_utilization_max": 0.95,
    "thin_file_employment_years_max": 0.5,
    "thin_file_loan_to_income_min": 5.0,
}
SOFT_FLAGS = {
    "thin_credit_file_months": 12,
    "post_loan_dti_soft_min": 0.45,
    "post_loan_dti_soft_max": 0.60,
    "disposable_income_floor": 5000.0,
}


@pytest.fixture
def policy() -> RecommendationPolicy:
    return RecommendationPolicy(
        approve_score_min=700,
        reject_score_max=549,
        approve_probability_max=0.084,
        reject_probability_max=0.252,
        hard_fails=HARD_FAILS,
        soft_flags=SOFT_FLAGS,
    )


def safe_features(**overrides: Any) -> dict[str, Any]:
    """A feature row that triggers no hard fail and no soft flag."""
    base = {
        "previous_defaults": 0,
        "post_loan_dti": 0.30,
        "credit_utilization": 0.25,
        "employment_years": 8.0,
        "loan_to_income": 1.5,
        "credit_history_months": 96,
        "disposable_income": 20000.0,
    }
    base.update(overrides)
    return base


def test_from_config_resolves_thresholds_from_chosen_threshold() -> None:
    config = {
        "recommendation": {
            "approve_score_min": 700,
            "reject_score_max": 549,
            "reject_probability_multiplier": 3.0,
            "hard_fails": HARD_FAILS,
            "soft_flags": SOFT_FLAGS,
        }
    }
    policy = RecommendationPolicy.from_config(config, chosen_threshold=0.084)
    assert policy.approve_probability_max == pytest.approx(0.084)
    assert policy.reject_probability_max == pytest.approx(0.252)


def test_approve_when_probability_low_score_high_no_flags(
    policy: RecommendationPolicy,
) -> None:
    result = recommend(
        probability=0.02, credit_score=750, features=safe_features(), policy=policy
    )
    assert result.decision == "APPROVE"
    assert result.triggered_rules
    assert "Approved" in result.reason


def test_reject_when_probability_above_reject_threshold(
    policy: RecommendationPolicy,
) -> None:
    result = recommend(
        probability=0.30, credit_score=750, features=safe_features(), policy=policy
    )
    assert result.decision == "REJECT"
    assert any("probability" in rule for rule in result.triggered_rules)


def test_reject_when_score_at_or_below_reject_score_max(
    policy: RecommendationPolicy,
) -> None:
    result = recommend(
        probability=0.02, credit_score=549, features=safe_features(), policy=policy
    )
    assert result.decision == "REJECT"
    assert any("credit_score" in rule for rule in result.triggered_rules)


def test_score_550_is_not_rejected_by_score_rule_alone(
    policy: RecommendationPolicy,
) -> None:
    # 550 clears the reject-by-score bar; with a safe probability and no
    # flags this should NOT be REJECT.
    result = recommend(
        probability=0.02, credit_score=550, features=safe_features(), policy=policy
    )
    assert result.decision != "REJECT"


@pytest.mark.parametrize(
    "overrides,expected_rule_substring",
    [
        ({"previous_defaults": 2}, "previous_defaults"),
        ({"post_loan_dti": 0.61}, "post_loan_dti"),
        ({"credit_utilization": 0.96}, "credit_utilization"),
        (
            {"employment_years": 0.25, "loan_to_income": 6.0},
            "thin_file",
        ),
    ],
)
def test_each_hard_fail_rule_triggers_reject_on_a_targeted_fixture(
    policy: RecommendationPolicy,
    overrides: dict[str, Any],
    expected_rule_substring: str,
) -> None:
    # Otherwise-perfect probability/score -- only the hard fail should force REJECT.
    result = recommend(
        probability=0.01,
        credit_score=850,
        features=safe_features(**overrides),
        policy=policy,
    )
    assert result.decision == "REJECT"
    assert any(expected_rule_substring in rule for rule in result.triggered_rules)


def test_hard_fail_overrides_otherwise_approvable_application(
    policy: RecommendationPolicy,
) -> None:
    result = recommend(
        probability=0.01,
        credit_score=900,
        features=safe_features(previous_defaults=3),
        policy=policy,
    )
    assert result.decision == "REJECT"


@pytest.mark.parametrize(
    "overrides,expected_rule_substring",
    [
        ({"credit_history_months": 6}, "thin_credit_file"),
        ({"post_loan_dti": 0.50}, "elevated_post_loan_dti"),
        ({"disposable_income": 1000.0}, "low_disposable_income"),
    ],
)
def test_each_soft_flag_forces_review_instead_of_approve(
    policy: RecommendationPolicy,
    overrides: dict[str, Any],
    expected_rule_substring: str,
) -> None:
    # Would otherwise APPROVE (low probability, high score, no hard fail).
    result = recommend(
        probability=0.01,
        credit_score=850,
        features=safe_features(**overrides),
        policy=policy,
    )
    assert result.decision == "REVIEW"
    assert any(expected_rule_substring in rule for rule in result.triggered_rules)


def test_middle_band_with_no_flags_is_review(policy: RecommendationPolicy) -> None:
    # Probability between approve and reject thresholds, score between the
    # approve and reject bars -- no hard fail, no soft flag.
    result = recommend(
        probability=0.15, credit_score=620, features=safe_features(), policy=policy
    )
    assert result.decision == "REVIEW"
    assert any("middle_band" in rule for rule in result.triggered_rules)


def test_missing_required_feature_raises_key_error(
    policy: RecommendationPolicy,
) -> None:
    incomplete = safe_features()
    del incomplete["post_loan_dti"]
    with pytest.raises(KeyError):
        recommend(
            probability=0.02, credit_score=750, features=incomplete, policy=policy
        )


def test_every_decision_is_traceable_to_a_rule(policy: RecommendationPolicy) -> None:
    for probability, score, overrides in [
        (0.02, 750, {}),
        (0.30, 750, {}),
        (0.02, 549, {}),
        (0.01, 850, {"credit_history_months": 6}),
        (0.15, 620, {}),
    ]:
        result = recommend(
            probability=probability,
            credit_score=score,
            features=safe_features(**overrides),
            policy=policy,
        )
        assert result.triggered_rules, "every decision must carry at least one rule"
        assert result.reason
