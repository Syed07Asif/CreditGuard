"""Lending recommendation policy: turns a calibrated default probability, a
credit score and a handful of raw point-in-time feature values into an
APPROVE / REVIEW / REJECT decision.

Every input to the policy other than the probability/score themselves comes
from `config/scoring.yaml`'s `recommendation` section -- nothing is
hard-coded here. The probability thresholds are anchored to the *active
model's own* Phase 6 cost-optimal operating point
(`model_registry.metrics.chosen_threshold`, see
`creditguard.models.threshold`) rather than a second, independently chosen
number: `approve_probability_max` is that threshold itself (approve only
when the model wouldn't flag the applicant as risky at all), and
`reject_probability_max` is a configurable multiple of it.

Every `Recommendation` carries the exact rule(s) that produced it
(`triggered_rules`) so a decision is always traceable, never a bare label.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

Decision = str  # "APPROVE" | "REVIEW" | "REJECT"


@dataclass(frozen=True)
class Recommendation:
    """The outcome of `recommend`: a decision, the rule(s) that produced
    it, and a human-readable reason string combining them.
    """

    decision: Decision
    triggered_rules: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class RecommendationPolicy:
    """Every tunable `recommend` needs, resolved once per active model
    (the probability thresholds depend on that model's own registered
    `chosen_threshold`) and reused across requests.
    """

    approve_score_min: int
    reject_score_max: int
    approve_probability_max: float
    reject_probability_max: float
    hard_fails: dict[str, float]
    soft_flags: dict[str, float]

    @classmethod
    def from_config(
        cls, config: dict[str, Any], chosen_threshold: float
    ) -> RecommendationPolicy:
        """Build from `config/scoring.yaml`'s `recommendation` section and
        the active model's registered `chosen_threshold` (Phase 6's
        expected-cost-minimising decision threshold).
        """
        section = config["recommendation"]
        multiplier = float(section["reject_probability_multiplier"])
        return cls(
            approve_score_min=int(section["approve_score_min"]),
            reject_score_max=int(section["reject_score_max"]),
            approve_probability_max=float(chosen_threshold),
            reject_probability_max=float(chosen_threshold) * multiplier,
            hard_fails=dict(section["hard_fails"]),
            soft_flags=dict(section["soft_flags"]),
        )


def _as_float(features: Mapping[str, Any], key: str) -> float:
    value = features.get(key)
    if value is None:
        raise KeyError(f"recommend() requires feature {key!r}, which was missing")
    return float(value)


def _check_hard_fails(
    features: Mapping[str, Any], hard_fails: Mapping[str, float]
) -> list[str]:
    """Every hard-fail rule that fires for this application. Any one alone
    forces REJECT, regardless of probability or score.
    """
    triggered: list[str] = []

    previous_defaults = _as_float(features, "previous_defaults")
    threshold = hard_fails["previous_defaults_min"]
    if previous_defaults >= threshold:
        triggered.append(
            f"hard_fail:previous_defaults ({previous_defaults:g}) >= {threshold:g}"
        )

    post_loan_dti = _as_float(features, "post_loan_dti")
    threshold = hard_fails["post_loan_dti_max"]
    if post_loan_dti > threshold:
        triggered.append(
            f"hard_fail:post_loan_dti ({post_loan_dti:.3f}) > {threshold:g}"
        )

    credit_utilization = _as_float(features, "credit_utilization")
    threshold = hard_fails["credit_utilization_max"]
    if credit_utilization > threshold:
        triggered.append(
            f"hard_fail:credit_utilization ({credit_utilization:.3f}) > {threshold:g}"
        )

    employment_years = _as_float(features, "employment_years")
    loan_to_income = _as_float(features, "loan_to_income")
    years_threshold = hard_fails["thin_file_employment_years_max"]
    lti_threshold = hard_fails["thin_file_loan_to_income_min"]
    if employment_years < years_threshold and loan_to_income > lti_threshold:
        triggered.append(
            f"hard_fail:thin_file (employment_years={employment_years:.2f} < "
            f"{years_threshold:g} AND loan_to_income={loan_to_income:.2f} > "
            f"{lti_threshold:g})"
        )

    return triggered


def _check_soft_flags(
    features: Mapping[str, Any], soft_flags: Mapping[str, float]
) -> list[str]:
    """Every soft-policy flag that fires. Any one alone forces REVIEW
    instead of APPROVE, even when probability/score would otherwise clear
    the approve bar.
    """
    triggered: list[str] = []

    credit_history_months = _as_float(features, "credit_history_months")
    threshold = soft_flags["thin_credit_file_months"]
    if credit_history_months < threshold:
        triggered.append(
            f"soft_flag:thin_credit_file (credit_history_months="
            f"{credit_history_months:g} < {threshold:g})"
        )

    post_loan_dti = _as_float(features, "post_loan_dti")
    low, high = (
        soft_flags["post_loan_dti_soft_min"],
        soft_flags["post_loan_dti_soft_max"],
    )
    if low <= post_loan_dti <= high:
        triggered.append(
            f"soft_flag:elevated_post_loan_dti (post_loan_dti="
            f"{post_loan_dti:.3f} in [{low:g}, {high:g}])"
        )

    disposable_income = _as_float(features, "disposable_income")
    floor = soft_flags["disposable_income_floor"]
    if disposable_income < floor:
        triggered.append(
            f"soft_flag:low_disposable_income (disposable_income="
            f"{disposable_income:.2f} < {floor:g})"
        )

    return triggered


def recommend(
    *,
    probability: float,
    credit_score: int,
    features: Mapping[str, Any],
    policy: RecommendationPolicy,
) -> Recommendation:
    """Decide APPROVE / REVIEW / REJECT.

    `features` must carry (at minimum) `previous_defaults`, `post_loan_dti`,
    `credit_utilization`, `employment_years`, `loan_to_income`,
    `credit_history_months` and `disposable_income` -- the raw, human-unit
    values from the point-in-time feature row (not the scaled/encoded model
    input).

    Evaluation order: REJECT (probability above the reject threshold, OR
    score at/below `reject_score_max`, OR any hard fail) takes precedence
    over everything else. Failing that, any soft flag forces REVIEW even if
    the applicant would otherwise clear the approve bar. Failing that,
    APPROVE requires both probability below the approve threshold and score
    at/above `approve_score_min`. Anything left over -- the genuine middle
    band -- is REVIEW.
    """
    hard_fail_rules = _check_hard_fails(features, policy.hard_fails)
    soft_flag_rules = _check_soft_flags(features, policy.soft_flags)

    reject_rules = list(hard_fail_rules)
    if probability > policy.reject_probability_max:
        reject_rules.append(
            f"probability ({probability:.4f}) > reject threshold "
            f"({policy.reject_probability_max:.4f})"
        )
    if credit_score <= policy.reject_score_max:
        reject_rules.append(
            f"credit_score ({credit_score}) <= reject_score_max "
            f"({policy.reject_score_max})"
        )
    if reject_rules:
        return Recommendation(
            decision="REJECT",
            triggered_rules=reject_rules,
            reason="Rejected: " + "; ".join(reject_rules),
        )

    if soft_flag_rules:
        return Recommendation(
            decision="REVIEW",
            triggered_rules=soft_flag_rules,
            reason="Referred for review: " + "; ".join(soft_flag_rules),
        )

    if (
        probability < policy.approve_probability_max
        and credit_score >= policy.approve_score_min
    ):
        rule = (
            f"probability ({probability:.4f}) < approve threshold "
            f"({policy.approve_probability_max:.4f}) AND credit_score "
            f"({credit_score}) >= approve_score_min ({policy.approve_score_min})"
        )
        return Recommendation(
            decision="APPROVE", triggered_rules=[rule], reason=f"Approved: {rule}"
        )

    rule = (
        f"middle_band: probability ({probability:.4f}) and credit_score "
        f"({credit_score}) meet neither the APPROVE nor REJECT criteria"
    )
    return Recommendation(
        decision="REVIEW", triggered_rules=[rule], reason=f"Referred for review: {rule}"
    )
