"""Bivariate / risk EDA: default rate by decile or band, correlation
structure among numeric features, and point-biserial correlation with the
target.

`default_rate_by_decile` is deliberately the most reusable function in this
module -- a decile-vs-default-rate table is, per the task brief, "the single
most useful chart in credit risk," and every numeric driver gets one.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats

DEFAULT_TARGET_COLUMN = "default_12m"


def default_rate_by_decile(
    df: pd.DataFrame,
    column: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin `column` into up to `n_bins` quantile buckets (fewer if there
    aren't `n_bins` distinct values) and compute the default rate per bucket.

    Returns one row per bucket, ordered from lowest to highest value, with
    `n`/`n_default` counts that sum to `len(df)` and monotonically
    non-decreasing bin edges.
    """
    values = df[column].astype(float)
    y = df[target_col].astype(int)
    valid = values.notna()
    vals, yv = values[valid], y[valid]

    if vals.nunique() < 2:
        # A zero-variance feature can't be split into quantile bins -- one
        # bin spanning the whole population is the correct (and honest)
        # description, and IV from it is 0 by construction (no discriminatory
        # power), which is the right answer for a constant feature.
        n = int(len(vals))
        n_default = int(yv.sum())
        return pd.DataFrame(
            {
                "feature": [column],
                "decile": [1],
                "bin": (
                    [pd.Interval(vals.min(), vals.max(), closed="both")]
                    if n
                    else [None]
                ),
                "min_value": [vals.min() if n else float("nan")],
                "max_value": [vals.max() if n else float("nan")],
                "n": [n],
                "n_default": [n_default],
                "default_rate": [n_default / n if n else float("nan")],
            }
        )

    bins = pd.qcut(vals, q=n_bins, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "y": yv, "value": vals})
    grouped = frame.groupby("bin", observed=True)
    result = grouped.agg(
        n=("y", "size"),
        n_default=("y", "sum"),
        min_value=("value", "min"),
        max_value=("value", "max"),
    ).reset_index()
    result = result.sort_values("min_value").reset_index(drop=True)
    result["decile"] = result.index + 1
    result["default_rate"] = result["n_default"] / result["n"]
    result["feature"] = column
    return result[
        [
            "feature",
            "decile",
            "bin",
            "min_value",
            "max_value",
            "n",
            "n_default",
            "default_rate",
        ]
    ]


def default_rate_by_band(
    df: pd.DataFrame,
    band_col: str,
    target_col: str = DEFAULT_TARGET_COLUMN,
    category_order: list[str] | None = None,
) -> pd.DataFrame:
    """Default rate for each distinct value of a categorical/band column."""
    grouped = df.groupby(band_col, observed=True)[target_col]
    result = grouped.agg(n="count", n_default="sum").reset_index()
    result["default_rate"] = result["n_default"] / result["n"]
    if category_order is not None:
        result[band_col] = pd.Categorical(
            result[band_col], categories=category_order, ordered=True
        )
        result = result.sort_values(band_col)
    else:
        result = result.sort_values(band_col, key=lambda s: s.astype(str))
    return result.reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation matrix among `columns`."""
    return df[columns].astype(float).corr()


def high_correlation_pairs(
    corr: pd.DataFrame, threshold: float = 0.8
) -> list[dict[str, float | str]]:
    """Every off-diagonal pair with |r| > `threshold`, sorted by |r| descending."""
    columns = list(corr.columns)
    pairs: list[dict[str, float | str]] = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > threshold:
                pairs.append(
                    {"feature_a": columns[i], "feature_b": columns[j], "r": float(r)}
                )
    return sorted(pairs, key=lambda p: -abs(float(p["r"])))


def point_biserial_correlations(
    df: pd.DataFrame, columns: list[str], target_col: str = DEFAULT_TARGET_COLUMN
) -> pd.DataFrame:
    """Point-biserial correlation of each numeric feature with the binary target."""
    y = df[target_col].astype(int)
    rows = []
    for column in columns:
        x = df[column].astype(float)
        mask = x.notna()
        if mask.sum() < 3 or x[mask].nunique() < 2:
            r, p = float("nan"), float("nan")
        else:
            r, p = stats.pointbiserialr(y[mask], x[mask])
        rows.append(
            {"feature": column, "point_biserial_r": float(r), "p_value": float(p)}
        )
    result = pd.DataFrame(rows)
    return result.reindex(
        result["point_biserial_r"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
