"""Leakage prevention: forbidden-feature enforcement, the point-in-time join,
and a target-correlation screen.

Two failure modes are made structurally impossible here, not just avoided by
care:

  (a) **Target leakage** -- using information that only exists after the
      decision. `FORBIDDEN_FEATURES`/`FORBIDDEN_PATTERNS` name every
      post-decision/outcome field, and `assert_no_leakage` is called from
      inside `creditguard.features.pipeline`'s fit *and* transform, not only
      from tests -- a forbidden column reaching feature code raises, it
      doesn't get quietly dropped.
  (b) **Train/test leakage** -- fitting any statistic on data outside the
      training fold. This module doesn't fit anything itself (the join below
      is a deterministic point-in-time lookup, not a statistic); see
      `creditguard.features.behavioural.BehaviouralFeatures` and
      `creditguard.features.encoders.build_preprocessor` for where
      train-only fitting actually happens.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

# Fields that only exist once a loan has already been decided and/or its
# outcome observed. Named explicitly per the project's post-decision/target
# leakage list, plus two fields not literally on that list but excluded for
# the same reason: `decision_date` and `status` only exist once the decision
# this model is meant to *inform* has already been made (CLAUDE.md hard rule
# #5: "No feature may use data dated after the loan application / decision
# date").
FORBIDDEN_FEATURES: frozenset[str] = frozenset(
    {
        "default_12m",
        "outcome_observed_date",
        "days_past_due",
        "recovery_amount",
        "collection_status",
        "write_off_amount",
        "future_missed_payments",
        "decision_date",
        "status",
    }
)

# `post_disbursement_*`, `repayment_*`, `charge_off_*` from the spec are
# prefix patterns, folded into the same compiled-regex list as the other
# patterns. `^post_disbursement_` is used rather than a bare `^post_`: a bare
# `^post_` collides with `ratios.compute_post_loan_dti`'s `post_loan_dti`
# feature, which is entirely pre-decision (it's the DTI *if* the loan being
# applied for is approved, computed from the application's own terms -- "post"
# here means "after the loan is added", not "after the decision"). Caught by
# testing this module against the real engineered feature set, not by
# inspection -- a reminder that an overly broad leakage pattern is a real bug
# (a false positive), not just extra caution.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"^future_",
        r"_after_decision$",
        r"^post_disbursement_",
        r"^outcome_",
        r"^actual_",
        r"^repayment_",
        r"^charge_off_",
    )
)


class LeakageError(ValueError):
    """Raised when a forbidden post-decision/target column reaches feature code."""


def assert_no_leakage(feature_names: Iterable[str]) -> None:
    """Raise `LeakageError` listing every forbidden column name found.

    Call this from both `fit` and `transform` of any pipeline stage that
    could plausibly see a raw source column -- catching it once in `fit` is
    not enough, since `transform` can be called independently (e.g. at
    serving time) with a differently-shaped frame.
    """
    offenders = sorted(
        {
            name
            for name in feature_names
            if name in FORBIDDEN_FEATURES
            or any(pattern.search(name) for pattern in FORBIDDEN_PATTERNS)
        }
    )
    if offenders:
        raise LeakageError(
            f"Forbidden post-decision/target column(s) reached feature code: "
            f"{offenders}"
        )


# The fixed, deterministic set of columns `point_in_time_join` produces.
# Deterministic (not data-dependent) because the column selection below is
# hard-coded, which lets `pipeline.CleaningAndMergeStep.get_feature_names_out`
# report it without needing to run on real data.
MERGED_FRAME_COLUMNS: tuple[str, ...] = (
    "loan_id",
    "customer_id",
    "application_date",
    "age",
    "gender",
    "marital_status",
    "dependents",
    "education",
    "employment_type",
    "employment_years",
    "annual_income",
    "city_tier",
    "loan_type",
    "loan_amount",
    "loan_tenure_months",
    "interest_rate",
    "loan_purpose",
    "financial_as_of_date",
    "monthly_income",
    "monthly_expenses",
    "existing_loan_count",
    "existing_loan_amount",
    "monthly_emi",
    "savings_balance",
    "total_assets",
    "total_liabilities",
    "credit_as_of_date",
    "credit_history_months",
    "num_credit_accounts",
    "total_credit_limit",
    "total_outstanding",
    "previous_defaults",
    "late_payments_12m",
    "missed_payments_12m",
    "active_loans",
    "closed_loans",
)

_CUSTOMER_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "marital_status",
    "dependents",
    "education",
    "employment_type",
    "employment_years",
    "annual_income",
    "city_tier",
]
_LOAN_COLUMNS = [
    "loan_id",
    "customer_id",
    "loan_type",
    "loan_amount",
    "loan_tenure_months",
    "interest_rate",
    "loan_purpose",
    "application_date",
]
_FINANCIAL_COLUMNS = [
    "customer_id",
    "as_of_date",
    "monthly_income",
    "monthly_expenses",
    "existing_loan_count",
    "existing_loan_amount",
    "monthly_emi",
    "savings_balance",
    "total_assets",
    "total_liabilities",
]
_CREDIT_COLUMNS = [
    "customer_id",
    "as_of_date",
    "credit_history_months",
    "num_credit_accounts",
    "total_credit_limit",
    "total_outstanding",
    "previous_defaults",
    "late_payments_12m",
    "missed_payments_12m",
    "active_loans",
    "closed_loans",
]
# Deliberately NOT carried over from credit_history: `credit_utilization`.
# `ratios.RatioFeatures` recomputes it as total_outstanding/total_credit_limit
# through `safe_divide` (guarded against a zero/negative credit limit), so
# there is exactly one, safely-guarded source of truth for this ratio rather
# than a raw bureau-reported column that could disagree with it.


def point_in_time_join(
    customers: pd.DataFrame,
    loan_applications: pd.DataFrame,
    financial_profiles: pd.DataFrame,
    credit_history: pd.DataFrame,
) -> pd.DataFrame:
    """One row per loan: customer + loan terms + that customer's *latest*
    financial_profiles/credit_history snapshot with `as_of_date <=
    application_date`.

    Never joins on `customer_id` alone: uses `pd.merge_asof(...,
    direction="backward")`, which additionally requires the snapshot's
    `as_of_date` to be on or before the loan's `application_date` for that
    exact customer. A snapshot dated after `application_date` -- however
    recent -- can never be selected.
    """
    loans = loan_applications[_LOAN_COLUMNS].merge(
        customers[_CUSTOMER_COLUMNS], on="customer_id", how="left"
    )
    loans = loans.assign(application_date=pd.to_datetime(loans["application_date"]))
    loans = loans.sort_values("application_date").reset_index(drop=True)

    financial = financial_profiles[_FINANCIAL_COLUMNS].copy()
    financial["as_of_date"] = pd.to_datetime(financial["as_of_date"])
    financial = financial.sort_values("as_of_date").reset_index(drop=True)

    credit = credit_history[_CREDIT_COLUMNS].copy()
    credit["as_of_date"] = pd.to_datetime(credit["as_of_date"])
    credit = credit.sort_values("as_of_date").reset_index(drop=True)

    merged = pd.merge_asof(
        loans,
        financial,
        left_on="application_date",
        right_on="as_of_date",
        by="customer_id",
        direction="backward",
    ).rename(columns={"as_of_date": "financial_as_of_date"})

    merged = merged.sort_values("application_date")
    merged = pd.merge_asof(
        merged,
        credit,
        left_on="application_date",
        right_on="as_of_date",
        by="customer_id",
        direction="backward",
    ).rename(columns={"as_of_date": "credit_as_of_date"})

    merged = merged.reset_index(drop=True)
    assert_no_leakage(merged.columns)
    return merged


def check_target_correlation(
    X: pd.DataFrame, y: pd.Series, threshold: float = 0.95
) -> list[str]:
    """Warn (loudly) about any numeric feature with |corr| > `threshold`
    against the target -- almost always leakage, not a great feature.

    Returns the list of offending column names so callers can act on it
    programmatically as well as seeing the warning.
    """
    numeric = X.select_dtypes(include=[np.number])
    y_float = y.astype(float)
    offenders: list[str] = []
    for column in numeric.columns:
        corr = numeric[column].corr(y_float)
        if pd.notna(corr) and abs(corr) > threshold:
            offenders.append(column)
            warnings.warn(
                f"Feature '{column}' has |corr|={abs(corr):.3f} with the "
                f"target (> {threshold}) -- almost always leakage, not a "
                "great feature. Review before using it in modelling.",
                stacklevel=2,
            )
    return offenders
