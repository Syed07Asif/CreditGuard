"""Ratio features: point-in-time financial ratios derived from the merged
per-loan frame produced by `creditguard.features.leakage.point_in_time_join`.

Every formula divides by a real-world quantity that can legitimately be zero
(and, pre-cleaning, negative): `monthly_income`, `total_credit_limit`,
`annual_income`, `monthly_expenses`. Rather than let a division produce
inf/NaN -- which would silently corrupt the imputer/scaler downstream --
every ratio goes through `safe_divide`, which maps a non-positive denominator
to a documented, ratio-specific cap (its worst-case sentinel) and clips every
result, including ordinary divisions, into a fixed finite range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Per-ratio (floor, cap) bounds. A non-positive denominator maps to `cap`;
# every result is clipped to [floor, cap] so a single extreme row can never
# distort StandardScaler's fitted mean/std downstream.
DTI_CAP = 5.0
EMI_TO_INCOME_CAP = 2.0
CREDIT_UTILIZATION_CAP = 2.0
LOAN_TO_INCOME_CAP = 10.0
POST_LOAN_DTI_CAP = 5.0
SAVINGS_TO_INCOME_CAP = 50.0

RATIO_OUTPUT_COLUMNS: tuple[str, ...] = (
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
)


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    *,
    cap: float,
    floor: float = 0.0,
) -> np.ndarray:
    """`numerator / denominator`, guarded against zero/negative denominators.

    A denominator <= 0 maps to `cap` (the ratio's documented worst-case
    sentinel) instead of producing inf/NaN. The final result -- including
    ordinary in-range divisions -- is always clipped to `[floor, cap]`.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(denominator > 0, numerator / denominator, cap)
    raw = np.nan_to_num(raw, nan=cap, posinf=cap, neginf=floor)
    return np.clip(raw, floor, cap)


def compute_proposed_emi(
    loan_amount: pd.Series | np.ndarray,
    interest_rate: pd.Series | np.ndarray,
    loan_tenure_months: pd.Series | np.ndarray,
) -> np.ndarray:
    """Standard amortised EMI: `P*r*(1+r)^n / ((1+r)^n - 1)`, `r = annual_rate/12/100`.

    `r == 0` (a zero-interest loan) is handled by its analytic limit, `P / n`,
    rather than a 0/0 division. `n <= 0` (should never happen --
    `loan_tenure_months` is DB-constrained `> 0` -- but guarded anyway) maps
    to `P`: pay it all now. A pathological negative `interest_rate` (also
    DB-constrained `>= 0`) is clipped to 0 before use.
    """
    principal = np.asarray(loan_amount, dtype=float)
    annual_rate = np.clip(np.asarray(interest_rate, dtype=float), 0.0, None)
    n = np.asarray(loan_tenure_months, dtype=float)
    r = annual_rate / 12.0 / 100.0

    n_safe = np.where(n > 0, n, 1.0)
    zero_rate_emi = principal / n_safe

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.power(1.0 + r, n_safe)
        amortised = principal * r * factor / (factor - 1.0)

    emi = np.where(np.isclose(r, 0.0), zero_rate_emi, amortised)
    emi = np.where(n > 0, emi, principal)
    return np.nan_to_num(emi, nan=0.0, posinf=0.0, neginf=0.0)


def compute_dti(monthly_emi, monthly_expenses, monthly_income) -> np.ndarray:
    """`(monthly_emi + monthly_expenses) / monthly_income`."""
    numerator = np.asarray(monthly_emi, dtype=float) + np.asarray(
        monthly_expenses, dtype=float
    )
    return safe_divide(numerator, monthly_income, cap=DTI_CAP)


def compute_emi_to_income(monthly_emi, monthly_income) -> np.ndarray:
    """`monthly_emi / monthly_income`."""
    return safe_divide(monthly_emi, monthly_income, cap=EMI_TO_INCOME_CAP)


def compute_credit_utilization(total_outstanding, total_credit_limit) -> np.ndarray:
    """`total_outstanding / total_credit_limit`."""
    return safe_divide(
        total_outstanding, total_credit_limit, cap=CREDIT_UTILIZATION_CAP
    )


def compute_loan_to_income(loan_amount, annual_income) -> np.ndarray:
    """`loan_amount / annual_income`."""
    return safe_divide(loan_amount, annual_income, cap=LOAN_TO_INCOME_CAP)


def compute_post_loan_dti(
    monthly_emi, proposed_emi, monthly_expenses, monthly_income
) -> np.ndarray:
    """`(monthly_emi + proposed_emi + monthly_expenses) / monthly_income`."""
    numerator = (
        np.asarray(monthly_emi, dtype=float)
        + np.asarray(proposed_emi, dtype=float)
        + np.asarray(monthly_expenses, dtype=float)
    )
    return safe_divide(numerator, monthly_income, cap=POST_LOAN_DTI_CAP)


def compute_savings_to_income(savings_balance, monthly_income) -> np.ndarray:
    """`savings_balance / monthly_income`."""
    return safe_divide(savings_balance, monthly_income, cap=SAVINGS_TO_INCOME_CAP)


def compute_net_worth(total_assets, total_liabilities) -> np.ndarray:
    """`total_assets - total_liabilities`. No division; NaNs pass through for
    the downstream imputer."""
    return np.asarray(total_assets, dtype=float) - np.asarray(
        total_liabilities, dtype=float
    )


def compute_leverage_ratio(total_liabilities, total_assets) -> np.ndarray:
    """`total_liabilities / max(total_assets, 1)`."""
    denominator = np.maximum(np.asarray(total_assets, dtype=float), 1.0)
    return np.asarray(total_liabilities, dtype=float) / denominator


def compute_disposable_income(
    monthly_income, monthly_expenses, monthly_emi
) -> np.ndarray:
    """`monthly_income - monthly_expenses - monthly_emi`. No division."""
    return (
        np.asarray(monthly_income, dtype=float)
        - np.asarray(monthly_expenses, dtype=float)
        - np.asarray(monthly_emi, dtype=float)
    )


def compute_months_of_runway(savings_balance, monthly_expenses) -> np.ndarray:
    """`savings_balance / max(monthly_expenses, 1)`."""
    denominator = np.maximum(np.asarray(monthly_expenses, dtype=float), 1.0)
    return np.asarray(savings_balance, dtype=float) / denominator


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Appends the 11 point-in-time financial ratio features. Stateless."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> RatioFeatures:
        """No-op: every ratio here is a pure function of the current row."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of `X` with the 11 ratio columns appended."""
        df = X.copy()
        df["proposed_emi"] = compute_proposed_emi(
            df["loan_amount"], df["interest_rate"], df["loan_tenure_months"]
        )
        df["dti"] = compute_dti(
            df["monthly_emi"], df["monthly_expenses"], df["monthly_income"]
        )
        df["emi_to_income"] = compute_emi_to_income(
            df["monthly_emi"], df["monthly_income"]
        )
        df["credit_utilization"] = compute_credit_utilization(
            df["total_outstanding"], df["total_credit_limit"]
        )
        df["loan_to_income"] = compute_loan_to_income(
            df["loan_amount"], df["annual_income"]
        )
        df["post_loan_dti"] = compute_post_loan_dti(
            df["monthly_emi"],
            df["proposed_emi"],
            df["monthly_expenses"],
            df["monthly_income"],
        )
        df["savings_to_income"] = compute_savings_to_income(
            df["savings_balance"], df["monthly_income"]
        )
        df["net_worth"] = compute_net_worth(df["total_assets"], df["total_liabilities"])
        df["leverage_ratio"] = compute_leverage_ratio(
            df["total_liabilities"], df["total_assets"]
        )
        df["disposable_income"] = compute_disposable_income(
            df["monthly_income"], df["monthly_expenses"], df["monthly_emi"]
        )
        df["months_of_runway"] = compute_months_of_runway(
            df["savings_balance"], df["monthly_expenses"]
        )
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Input feature names plus the 11 ratio columns this stage adds."""
        base = list(input_features) if input_features is not None else []
        new = [name for name in RATIO_OUTPUT_COLUMNS if name not in base]
        return np.asarray(base + new, dtype=object)
