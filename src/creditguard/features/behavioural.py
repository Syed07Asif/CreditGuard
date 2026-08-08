"""Behavioural credit-bureau features, including four banded columns.

`utilization_band` uses fixed business cutoffs (0-30/30-50/50-70/70-90/90+),
computed the same way regardless of what data it sees. `age_band`,
`tenure_band` and `income_band` are quantile bins whose edges are learned
from training data only (`BehaviouralFeatures.fit`) and applied unchanged in
`transform` -- this is this phase's primary train/test-leakage guard for
binning, alongside `creditguard.features.encoders.build_preprocessor`'s
imputer/scaler/encoder statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

N_QUANTILE_BINS = 5
QUANTILE_LABELS: tuple[str, ...] = tuple(f"Q{i}" for i in range(1, N_QUANTILE_BINS + 1))

UTILIZATION_BAND_EDGES: tuple[float, ...] = (-np.inf, 0.30, 0.50, 0.70, 0.90, np.inf)
UTILIZATION_BAND_LABELS: tuple[str, ...] = ("0-30", "30-50", "50-70", "70-90", "90+")

BEHAVIOURAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "delinquency_rate",
    "has_prior_default",
    "credit_history_years",
    "accounts_per_year",
    "active_loan_ratio",
    "employment_stability",
    "income_per_dependent",
    "utilization_band",
    "age_band",
    "tenure_band",
    "income_band",
)


def compute_delinquency_rate(
    late_payments_12m, missed_payments_12m, num_credit_accounts
) -> np.ndarray:
    """`(late_payments_12m + missed_payments_12m) / max(num_credit_accounts, 1)`."""
    denominator = np.maximum(np.asarray(num_credit_accounts, dtype=float), 1.0)
    numerator = np.asarray(late_payments_12m, dtype=float) + np.asarray(
        missed_payments_12m, dtype=float
    )
    return numerator / denominator


def compute_has_prior_default(previous_defaults) -> np.ndarray:
    """`previous_defaults > 0`, as an int 0/1 flag."""
    return (np.asarray(previous_defaults, dtype=float) > 0).astype(int)


def compute_credit_history_years(credit_history_months) -> np.ndarray:
    """`credit_history_months / 12`."""
    return np.asarray(credit_history_months, dtype=float) / 12.0


def compute_accounts_per_year(num_credit_accounts, credit_history_years) -> np.ndarray:
    """`num_credit_accounts / max(credit_history_years, 0.5)`."""
    denominator = np.maximum(np.asarray(credit_history_years, dtype=float), 0.5)
    return np.asarray(num_credit_accounts, dtype=float) / denominator


def compute_active_loan_ratio(active_loans, closed_loans) -> np.ndarray:
    """`active_loans / max(active_loans + closed_loans, 1)`."""
    active = np.asarray(active_loans, dtype=float)
    denominator = np.maximum(active + np.asarray(closed_loans, dtype=float), 1.0)
    return active / denominator


def compute_employment_stability(employment_years, age) -> np.ndarray:
    """`employment_years / max(age - 18, 1)`."""
    denominator = np.maximum(np.asarray(age, dtype=float) - 18.0, 1.0)
    return np.asarray(employment_years, dtype=float) / denominator


def compute_income_per_dependent(monthly_income, dependents) -> np.ndarray:
    """`monthly_income / (dependents + 1)`."""
    return np.asarray(monthly_income, dtype=float) / (
        np.asarray(dependents, dtype=float) + 1.0
    )


def compute_utilization_band(credit_utilization) -> np.ndarray:
    """Fixed-cutoff band: 0-30/30-50/50-70/70-90/90+. Not fit from data."""
    values = np.asarray(credit_utilization, dtype=float)
    # pd.cut on a bare ndarray returns a Categorical (not a Series), whose
    # .astype(str) already yields a plain ndarray -- no .to_numpy() to call.
    labels = pd.cut(values, bins=UTILIZATION_BAND_EDGES, labels=UTILIZATION_BAND_LABELS)
    return np.asarray(labels.astype(str))


class BehaviouralFeatures(BaseEstimator, TransformerMixin):
    """Appends behavioural features plus 4 banded columns.

    `age_band`/`tenure_band`/`income_band` quantile edges are learned in
    `fit` from whatever data it's given -- callers MUST call `fit` on the
    training split only. `-inf`/`+inf` are substituted for the outermost
    edges so that a val/test/serving value outside the observed training
    range still lands in the extreme bin instead of becoming NaN.
    """

    def __init__(self, n_bins: int = N_QUANTILE_BINS) -> None:
        self.n_bins = n_bins

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> BehaviouralFeatures:
        quantiles = np.linspace(0, 1, self.n_bins + 1)
        self.age_edges_ = self._fit_edges(X["age"], quantiles)
        self.tenure_edges_ = self._fit_edges(X["employment_years"], quantiles)
        self.income_edges_ = self._fit_edges(X["annual_income"], quantiles)
        return self

    @staticmethod
    def _fit_edges(series: pd.Series, quantiles: np.ndarray) -> np.ndarray:
        clean = series.dropna().astype(float)
        edges = np.unique(np.quantile(clean, quantiles))
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @staticmethod
    def _apply_band(series: pd.Series, edges: np.ndarray) -> np.ndarray:
        n_labels = len(edges) - 1
        labels = list(QUANTILE_LABELS[:n_labels])
        binned = pd.cut(
            series.astype(float), bins=edges, labels=labels, duplicates="drop"
        )
        return binned.astype(str).to_numpy()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["credit_history_years"] = compute_credit_history_years(
            df["credit_history_months"]
        )
        df["delinquency_rate"] = compute_delinquency_rate(
            df["late_payments_12m"],
            df["missed_payments_12m"],
            df["num_credit_accounts"],
        )
        df["has_prior_default"] = compute_has_prior_default(df["previous_defaults"])
        df["accounts_per_year"] = compute_accounts_per_year(
            df["num_credit_accounts"], df["credit_history_years"]
        )
        df["active_loan_ratio"] = compute_active_loan_ratio(
            df["active_loans"], df["closed_loans"]
        )
        df["employment_stability"] = compute_employment_stability(
            df["employment_years"], df["age"]
        )
        df["income_per_dependent"] = compute_income_per_dependent(
            df["monthly_income"], df["dependents"]
        )

        utilization_source = (
            df["credit_utilization"]
            if "credit_utilization" in df.columns
            else pd.Series(np.nan, index=df.index)
        )
        df["utilization_band"] = compute_utilization_band(utilization_source)

        df["age_band"] = self._apply_band(df["age"], self.age_edges_)
        df["tenure_band"] = self._apply_band(df["employment_years"], self.tenure_edges_)
        df["income_band"] = self._apply_band(df["annual_income"], self.income_edges_)
        return df

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Input feature names plus the behavioural/band columns this stage adds."""
        base = list(input_features) if input_features is not None else []
        new = [name for name in BEHAVIOURAL_OUTPUT_COLUMNS if name not in base]
        return np.asarray(base + new, dtype=object)
