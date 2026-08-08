"""Information Value / Weight of Evidence, and the temporal check that
justifies (or challenges) Phase 4's temporal train/val/test split.

WOE/IV convention follows Siddiqi's standard definition:

    WOE_bin = ln( %non-events in bin / %events in bin )
    IV      = sum_bins[ (%non-events - %events) * WOE_bin ]

"Events" means `default_12m == 1`. A bin with a higher-than-average default
rate has more of its share of events than non-events, so `%non-events /
%events` < 1 and WOE is negative -- higher risk bins get negative WOE, lower
risk bins get positive WOE, which is the standard reading. IV is always
>= 0 by construction (it's a sum of two non-negative terms' asymmetry).

A bin with zero events or zero non-events would make the ratio undefined;
`_woe_iv_from_bin_counts` applies the standard "add half a case" Laplace
correction, but only to bins that would otherwise divide by zero, so it
never perturbs a bin where both classes are already present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditguard.eda.bivariate import default_rate_by_decile

DEFAULT_TARGET_COLUMN = "default_12m"

IV_BANDS: tuple[tuple[float, str], ...] = (
    (0.02, "useless"),
    (0.1, "weak"),
    (0.3, "medium"),
    (0.5, "strong"),
    (float("inf"), "suspiciously strong - investigate for leakage"),
)


def interpret_iv(iv: float) -> str:
    """Map an IV value to its standard credit-scoring interpretation band."""
    for upper, label in IV_BANDS:
        if iv < upper:
            return label
    return IV_BANDS[-1][1]  # pragma: no cover - IV_BANDS' last bound is inf


def _woe_iv_from_bin_counts(
    bin_events: np.ndarray, bin_totals: np.ndarray
) -> tuple[np.ndarray, float]:
    """WOE per bin and total IV from raw (n_default, n_total) bin counts."""
    bin_events = np.asarray(bin_events, dtype=float)
    bin_totals = np.asarray(bin_totals, dtype=float)
    bin_non_events = bin_totals - bin_events

    bin_events = np.where(bin_events == 0, 0.5, bin_events)
    bin_non_events = np.where(bin_non_events == 0, 0.5, bin_non_events)

    total_events = bin_events.sum()
    total_non_events = bin_non_events.sum()
    dist_event = bin_events / total_events
    dist_non_event = bin_non_events / total_non_events

    woe = np.log(dist_non_event / dist_event)
    iv = float(((dist_non_event - dist_event) * woe).sum())
    return woe, iv


def compute_woe_iv_numeric(
    df: pd.DataFrame,
    column: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """WOE per decile bin (via `bivariate.default_rate_by_decile`) plus IV
    for a numeric feature.
    """
    decile_df = default_rate_by_decile(df, column, target_col, n_bins=n_bins)
    woe, iv = _woe_iv_from_bin_counts(
        decile_df["n_default"].to_numpy(), decile_df["n"].to_numpy()
    )
    decile_df = decile_df.copy()
    decile_df["woe"] = woe
    return decile_df, iv


def compute_woe_iv_categorical(
    df: pd.DataFrame, column: str, target_col: str = DEFAULT_TARGET_COLUMN
) -> tuple[pd.DataFrame, float]:
    """WOE per category plus IV for a categorical/ordinal feature (each
    distinct value is its own bin).
    """
    grouped = df.groupby(column, observed=True)[target_col]
    bin_df = grouped.agg(n="count", n_default="sum").reset_index()
    woe, iv = _woe_iv_from_bin_counts(
        bin_df["n_default"].to_numpy(), bin_df["n"].to_numpy()
    )
    bin_df["woe"] = woe
    bin_df["default_rate"] = bin_df["n_default"] / bin_df["n"]
    bin_df["feature"] = column
    return bin_df, iv


# Numeric columns with at most this many distinct values are treated as
# low-cardinality counts for IV purposes (exact-value bins), not continuous
# drivers (quantile deciles). On the real Phase 4 dataset there's a clean gap
# between count-like fields (has_prior_default=2 .. closed_loans=32 distinct
# values) and genuinely continuous ones (age=52 distinct values and up), so
# 40 separates them without needing per-column tuning.
LOW_CARDINALITY_THRESHOLD = 40


def iv_table(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_col: str = DEFAULT_TARGET_COLUMN,
    n_bins: int = 10,
) -> pd.DataFrame:
    """IV per feature across both numeric (decile-binned) and
    categorical/ordinal (native-category-binned) columns, sorted descending.

    A numeric column with `nunique() <= LOW_CARDINALITY_THRESHOLD` (e.g.
    `previous_defaults`, `dependents`, `has_prior_default`) is routed through
    the *categorical* WOE/IV path -- exact-value bins -- instead of `n_bins`
    quantile deciles. Quantile cut points on a zero-inflated, low-cardinality
    count column collapse onto the same value (usually 0) and
    `duplicates="drop"` then merges them into one bin covering the whole
    population, which silently reports IV == 0 for a feature that can be
    genuinely predictive once binned by its actual distinct values. This was
    caught by running this function against the real Phase 4 dataset:
    `previous_defaults` -- documented as the single strongest bureau risk
    signal -- came back IV == 0.0 via decile binning, but IV == 0.11
    ("medium") via per-value binning, with a clean monotone default-rate
    gradient from 10% to 100% across its 11 distinct values.
    """
    rows = []
    for column in numeric_columns:
        if df[column].nunique() <= LOW_CARDINALITY_THRESHOLD:
            _, iv = compute_woe_iv_categorical(df, column, target_col)
        else:
            _, iv = compute_woe_iv_numeric(df, column, target_col, n_bins=n_bins)
        rows.append({"feature": column, "type": "numeric", "iv": iv})
    for column in categorical_columns:
        _, iv = compute_woe_iv_categorical(df, column, target_col)
        rows.append({"feature": column, "type": "categorical", "iv": iv})
    result = pd.DataFrame(rows)
    result["interpretation"] = result["iv"].apply(interpret_iv)
    return result.sort_values("iv", ascending=False).reset_index(drop=True)


def monthly_volume_and_default_rate(
    df: pd.DataFrame,
    date_col: str = "application_date",
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Application volume and default rate by calendar month."""
    frame = df[[date_col, target_col]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame["month"] = frame[date_col].dt.to_period("M").astype(str)
    grouped = frame.groupby("month")[target_col].agg(n="count", n_default="sum")
    result = grouped.reset_index()
    result["default_rate"] = result["n_default"] / result["n"]
    return result.sort_values("month").reset_index(drop=True)


def detect_regime_shift(
    monthly_df: pd.DataFrame, z_threshold: float = 2.0
) -> dict[str, object]:
    """Flag months whose default rate is more than `z_threshold` standard
    deviations from the mean monthly default rate -- a simple, transparent
    signal for whether the temporal split (Phase 4) crosses a regime shift.
    """
    rates = monthly_df["default_rate"]
    mean = float(rates.mean())
    std = float(rates.std()) if len(rates) > 1 else 0.0
    if std == 0 or pd.isna(std):
        flagged: list[str] = []
    else:
        z = (rates - mean) / std
        flagged = monthly_df.loc[z.abs() > z_threshold, "month"].tolist()
    return {
        "mean_default_rate": mean,
        "std_default_rate": std,
        "flagged_months": flagged,
        "regime_shift_detected": len(flagged) > 0,
    }
