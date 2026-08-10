"""Translate SHAP attributions into sentences a loan officer can read.

Every feature `creditguard.scoring.engine` can hand to `generate_reason_codes`
(the full logical/source feature set from `config/features.yaml`'s
`feature_columns`) has a registered `FeatureReasonSpec`, covering *both*
directions -- what a high/present raw value means, and what a low/typical/
absent one means -- keyed by feature name (`FEATURE_REASON_SPECS`,
checked for completeness by `assert_full_coverage`).

A sentence has two independent parts, deliberately kept separate:

  1. A **factual clause** describing the applicant's actual raw value
     against the training-data portfolio benchmark (median for numeric
     features, mode for categorical/ordinal ones) -- always computed from
     the real value, never from which SHAP bucket the feature landed in.
  2. A **direction clause** ("increasing"/"reducing the estimated risk for
     this application") -- always computed from the real SHAP sign for
     this row, never assumed from the feature's usual direction.

For a linear model (the type actually registered in this project) these two
almost always agree, since a single global coefficient means sign follows
value deterministically; keeping them decoupled is still the more correct
design in general (e.g. if a tree-based model with real interaction effects
were ever promoted), and it means every sentence is honest about what's
literally true of the applicant even in a surprising case.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

FormatKind = str  # "percent" | "rate_percent" | "ratio" | "currency" | "years" |
#                    "months" | "months_of_income" | "count" | "count_zero" |
#                    "flag" | "category"
# "percent" multiplies a [0, 1]-ish fraction by 100 before appending "%"
# (e.g. dti=0.62 -> "62%"). "rate_percent" is for values already stored in
# percentage-point units (e.g. interest_rate=9.5 meaning 9.5%) -- appends
# "%" without rescaling.


class ReasonCodeCoverageError(RuntimeError):
    """Raised when a feature is missing a registered `FeatureReasonSpec`."""


@dataclass(frozen=True)
class FeatureReasonSpec:
    """One feature's display label, value formatting, and the business
    meaning of a high/present value vs. a low/typical/absent one --
    independent of which direction any particular application's SHAP
    contribution points.
    """

    display: str
    format: FormatKind
    high_value_phrase: str
    low_value_phrase: str


def _spec(display: str, format: FormatKind, high: str, low: str) -> FeatureReasonSpec:
    return FeatureReasonSpec(
        display=display, format=format, high_value_phrase=high, low_value_phrase=low
    )


FEATURE_REASON_SPECS: dict[str, FeatureReasonSpec] = {
    # -- demographic / employment (source: customers) --
    "age": _spec(
        "Applicant age",
        "years",
        "an older age, which correlates with more established finances",
        "a younger age, which correlates with a shorter financial track record",
    ),
    "dependents": _spec(
        "Number of dependents",
        "count",
        "more financial dependents, adding competing claims on income",
        "fewer financial dependents, leaving more income available",
    ),
    "employment_years": _spec(
        "Employment tenure",
        "years",
        "longer job tenure, a stronger stability signal",
        "shorter job tenure, a weaker stability signal",
    ),
    "annual_income": _spec(
        "Annual income",
        "currency",
        "higher annual income relative to the portfolio, increasing debt-servicing "
        "capacity",
        "lower annual income relative to the portfolio, reducing debt-servicing "
        "capacity",
    ),
    "city_tier": _spec(
        "City tier",
        "count",
        "a city tier associated with higher observed risk in this portfolio",
        "a city tier associated with lower observed risk in this portfolio",
    ),
    "gender": _spec(
        "Gender",
        "category",
        "not used by design in this project's risk assessment",
        "not used by design in this project's risk assessment",
    ),
    "marital_status": _spec(
        "Marital status",
        "category",
        "a household structure associated with higher observed risk in this portfolio",
        "a household structure associated with lower observed risk in this portfolio",
    ),
    "education": _spec(
        "Education level",
        "category",
        "an education level associated with higher observed risk in this portfolio",
        "an education level associated with lower observed risk in this portfolio",
    ),
    "employment_type": _spec(
        "Employment type",
        "category",
        "an employment type associated with higher observed income-stability risk",
        "an employment type associated with lower observed income-stability risk",
    ),
    # -- loan terms (source: loan_applications) --
    "loan_amount": _spec(
        "Requested loan amount",
        "currency",
        "a larger requested loan amount relative to the portfolio",
        "a smaller requested loan amount relative to the portfolio",
    ),
    "loan_tenure_months": _spec(
        "Loan tenure",
        "months",
        "a repayment tenure associated with higher observed risk in this portfolio",
        "a repayment tenure associated with lower observed risk in this portfolio",
    ),
    "interest_rate": _spec(
        "Interest rate",
        "rate_percent",
        "a higher-priced interest rate, typically reflecting bureau-assessed risk",
        "a lower-priced interest rate, typically reflecting bureau-assessed risk",
    ),
    "loan_type": _spec(
        "Loan type",
        "category",
        "a loan product type associated with higher observed risk in this portfolio",
        "a loan product type associated with lower observed risk in this portfolio",
    ),
    "loan_purpose": _spec(
        "Loan purpose",
        "category",
        "a stated purpose associated with higher observed risk in this portfolio",
        "a stated purpose associated with lower observed risk in this portfolio",
    ),
    # -- financial snapshot (source: financial_profiles) --
    "monthly_income": _spec(
        "Monthly income",
        "currency",
        "higher monthly income relative to the portfolio",
        "lower monthly income relative to the portfolio",
    ),
    "monthly_expenses": _spec(
        "Monthly expenses",
        "currency",
        "higher recurring monthly expenses, reducing disposable income",
        "lower recurring monthly expenses, preserving disposable income",
    ),
    "existing_loan_count": _spec(
        "Existing loan count",
        "count_zero",
        "one or more existing loans, adding to the overall debt burden",
        "no existing loans on record",
    ),
    "existing_loan_amount": _spec(
        "Existing loan balance",
        "currency",
        "a larger outstanding balance on existing loans",
        "a smaller outstanding balance on existing loans",
    ),
    "monthly_emi": _spec(
        "Existing monthly EMI",
        "currency",
        "a heavier existing EMI obligation",
        "a lighter existing EMI obligation",
    ),
    "savings_balance": _spec(
        "Savings balance",
        "currency",
        "a larger liquid savings buffer",
        "a smaller liquid savings buffer",
    ),
    "total_assets": _spec(
        "Total assets",
        "currency",
        "a larger overall asset base",
        "a smaller overall asset base",
    ),
    "total_liabilities": _spec(
        "Total liabilities",
        "currency",
        "a larger overall liability load",
        "a smaller overall liability load",
    ),
    # -- credit bureau snapshot (source: credit_history) --
    "credit_history_months": _spec(
        "Credit history length",
        "months",
        "a longer bureau credit history, more track record to assess",
        "a shorter bureau credit history, less track record to assess",
    ),
    "num_credit_accounts": _spec(
        "Number of credit accounts",
        "count",
        "a number of bureau-tracked accounts associated with higher observed risk",
        "a number of bureau-tracked accounts associated with lower observed risk",
    ),
    "total_credit_limit": _spec(
        "Total credit limit",
        "currency",
        "a larger aggregate credit limit",
        "a smaller aggregate credit limit",
    ),
    "total_outstanding": _spec(
        "Total outstanding balance",
        "currency",
        "a larger aggregate outstanding balance",
        "a smaller aggregate outstanding balance",
    ),
    "previous_defaults": _spec(
        "Previous defaults",
        "count_zero",
        "a history of previous defaults on record",
        "no previous defaults on record",
    ),
    "late_payments_12m": _spec(
        "Late payments (trailing 12 months)",
        "count_zero",
        "recent late payments on record",
        "no late payments in the trailing 12 months",
    ),
    "missed_payments_12m": _spec(
        "Missed payments (trailing 12 months)",
        "count_zero",
        "recent missed payments on record",
        "no missed payments in the trailing 12 months",
    ),
    "active_loans": _spec(
        "Active loans",
        "count",
        "a number of active loan accounts associated with higher observed risk",
        "a number of active loan accounts associated with lower observed risk",
    ),
    "closed_loans": _spec(
        "Closed loans",
        "count",
        "more closed/paid-off loan accounts, a stronger completed track record",
        "fewer closed/paid-off loan accounts, a thinner completed track record",
    ),
    # -- ratio features (creditguard.features.ratios) --
    "dti": _spec(
        "Debt-to-income ratio",
        "percent",
        "a heavier overall debt burden relative to income",
        "a manageable overall debt burden relative to income",
    ),
    "emi_to_income": _spec(
        "EMI-to-income ratio",
        "percent",
        "a heavier existing EMI burden relative to income",
        "a lighter existing EMI burden relative to income",
    ),
    "credit_utilization": _spec(
        "Credit utilisation",
        "percent",
        "heavy reliance on revolving credit",
        "light reliance on revolving credit",
    ),
    "loan_to_income": _spec(
        "Loan-to-income ratio",
        "ratio",
        "a requested loan size that is large relative to income",
        "a requested loan size that is modest relative to income",
    ),
    "proposed_emi": _spec(
        "Proposed EMI",
        "currency",
        "a larger EMI the new loan would add",
        "a smaller EMI the new loan would add",
    ),
    "post_loan_dti": _spec(
        "Projected post-loan DTI",
        "percent",
        "a projected debt-to-income ratio, if this loan is approved, that runs high",
        "a projected debt-to-income ratio, if this loan is approved, that stays "
        "manageable",
    ),
    "savings_to_income": _spec(
        "Savings-to-income ratio",
        "months_of_income",
        "more months of income held in savings",
        "fewer months of income held in savings",
    ),
    "net_worth": _spec(
        "Net worth",
        "currency",
        "a larger overall financial cushion",
        "a smaller overall financial cushion",
    ),
    "leverage_ratio": _spec(
        "Leverage ratio",
        "ratio",
        "higher debt relative to assets",
        "lower debt relative to assets",
    ),
    "disposable_income": _spec(
        "Disposable income",
        "currency",
        "more cash left after fixed obligations",
        "less cash left after fixed obligations",
    ),
    "months_of_runway": _spec(
        "Savings runway",
        "months",
        "more months the applicant could cover expenses from savings alone",
        "fewer months the applicant could cover expenses from savings alone",
    ),
    # -- behavioural features (creditguard.features.behavioural) --
    "delinquency_rate": _spec(
        "Delinquency rate",
        "ratio",
        "more delinquency incidents per account held",
        "fewer delinquency incidents per account held",
    ),
    "has_prior_default": _spec(
        "Prior default flag",
        "flag",
        "a prior default on record",
        "no prior default on record",
    ),
    "credit_history_years": _spec(
        "Credit history length",
        "years",
        "a longer bureau history in years",
        "a shorter bureau history in years",
    ),
    "accounts_per_year": _spec(
        "Account-opening pace",
        "ratio",
        "a faster account-opening pace, which can signal credit-seeking stress",
        "a slower, steadier account-opening pace",
    ),
    "active_loan_ratio": _spec(
        "Active loan ratio",
        "percent",
        "a larger share of credit history still open",
        "a larger share of credit history already closed out",
    ),
    "employment_stability": _spec(
        "Employment stability",
        "percent",
        "a larger fraction of working-age life spent in the current job",
        "a smaller fraction of working-age life spent in the current job",
    ),
    "income_per_dependent": _spec(
        "Income per dependent",
        "currency",
        "more income available per household dependent",
        "less income available per household dependent",
    ),
    # -- band features (fixed or quantile-binned, ordinal-encoded) --
    "utilization_band": _spec(
        "Credit utilisation band",
        "category",
        "a coarse utilisation tier associated with higher risk",
        "a coarse utilisation tier associated with lower risk",
    ),
    "age_band": _spec(
        "Age band",
        "category",
        "an age tier associated with higher observed risk in this portfolio",
        "an age tier associated with lower observed risk in this portfolio",
    ),
    "tenure_band": _spec(
        "Employment tenure band",
        "category",
        "a shorter-tenure band, a weaker stability signal",
        "a longer-tenure band, a stronger stability signal",
    ),
    "income_band": _spec(
        "Income band",
        "category",
        "a lower income tier",
        "a higher income tier",
    ),
}


def assert_full_coverage(feature_names: list[str]) -> None:
    """Raise `ReasonCodeCoverageError` listing every feature in
    `feature_names` (normally `config/features.yaml`'s full
    `feature_columns` list) that has no registered `FeatureReasonSpec`.
    """
    missing = [name for name in feature_names if name not in FEATURE_REASON_SPECS]
    if missing:
        raise ReasonCodeCoverageError(
            f"No reason-code template registered for feature(s): {missing}"
        )


def _format_value(value: Any, format_kind: FormatKind) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if format_kind == "percent":
        return f"{numeric * 100:.0f}%"
    if format_kind == "rate_percent":
        return f"{numeric:.1f}%"
    if format_kind == "currency":
        return f"{numeric:,.0f}"
    if format_kind == "years":
        return f"{numeric:.1f} years"
    if format_kind == "months":
        return f"{numeric:.0f} months"
    if format_kind == "months_of_income":
        return f"{numeric:.1f} months of income"
    if format_kind == "count":
        return f"{numeric:.0f}"
    if format_kind == "ratio":
        return f"{numeric:.2f}"
    return str(value)


def _comparison_word(value: float, benchmark: float) -> str:
    """A human comparison word for `value` against `benchmark`, matching
    the style of the worked examples ("well above the portfolio median").
    """
    if benchmark == 0:
        if value > 0:
            return "above"
        if value == 0:
            return "in line with"
        return "below"
    ratio = value / benchmark
    if ratio >= 1.5:
        return "well above"
    if ratio > 1.05:
        return "above"
    if ratio >= 0.95:
        return "in line with"
    if ratio > 0.5:
        return "below"
    return "well below"


def _direction_clause(contribution: float) -> str:
    return (
        "increasing the estimated risk for this application"
        if contribution > 0
        else "reducing the estimated risk for this application"
    )


def render_reason(
    feature: str,
    value: Any,
    contribution: float,
    benchmark: Mapping[str, Any],
) -> str:
    """One human-readable sentence for `feature`'s SHAP contribution on a
    single application: a factual clause (the applicant's raw value vs. the
    training-data portfolio benchmark) plus a direction clause (this row's
    actual SHAP sign).

    Raises:
        ReasonCodeCoverageError: if `feature` has no registered spec.
    """
    if feature not in FEATURE_REASON_SPECS:
        raise ReasonCodeCoverageError(
            f"No reason-code template registered for {feature!r}"
        )
    spec = FEATURE_REASON_SPECS[feature]
    direction_clause = _direction_clause(contribution)

    if spec.format in ("count_zero", "flag"):
        is_present = bool(float(value) > 0) if value is not None else False
        clause = spec.high_value_phrase if is_present else spec.low_value_phrase
        return f"{clause[0].upper()}{clause[1:]}, {direction_clause}."

    if spec.format == "category":
        mode = benchmark.get("mode")
        matches_common = str(value) == str(mode)
        clause = spec.low_value_phrase if matches_common else spec.high_value_phrase
        relation = "matches" if matches_common else "differs from"
        return (
            f"{spec.display} of {value} {relation} the portfolio's most common "
            f"value ({mode}), indicating {clause}, {direction_clause}."
        )

    median = float(benchmark.get("median", 0.0))
    numeric_value = float(value)
    comparison = _comparison_word(numeric_value, median)
    is_elevated = comparison in ("well above", "above")
    clause = spec.high_value_phrase if is_elevated else spec.low_value_phrase
    return (
        f"{spec.display} of {_format_value(numeric_value, spec.format)} is "
        f"{comparison} the portfolio median of {_format_value(median, spec.format)}, "
        f"indicating {clause}, {direction_clause}."
    )


def generate_reason_codes(
    top_positive_factors: list[tuple[str, float]],
    top_negative_factors: list[tuple[str, float]],
    raw_features: Mapping[str, Any],
    benchmarks: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the reason-code lists for a `ScoringResult`: `(risk_factors,
    positive_factors)`, each a list of `{"feature", "contribution",
    "reason"}` dicts, ordered as given (callers pass
    `ShapExplanation.top_positive_factors`/`top_negative_factors`, already
    ranked by absolute contribution).
    """

    def _build(factors: list[tuple[str, float]]) -> list[dict[str, Any]]:
        rows = []
        for feature, contribution in factors:
            value = raw_features.get(feature)
            benchmark = benchmarks.get(feature, {})
            reason = render_reason(feature, value, contribution, benchmark)
            rows.append(
                {"feature": feature, "contribution": contribution, "reason": reason}
            )
        return rows

    risk_factors = _build(top_positive_factors)
    positive_factors = _build(top_negative_factors)
    return risk_factors, positive_factors
