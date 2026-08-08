"""Univariate EDA: class balance, per-feature distribution stats, and
categorical frequency tables.

Nothing here fits a statistic used later by the modelling pipeline -- this
module only describes data that has already been produced by
`creditguard.features` (ratios/behavioural features included), so there is
no train/test-leakage concern the way there is in `creditguard.features`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_SKEW_THRESHOLD = 1.0


def default_rate_summary(y: pd.Series) -> dict[str, float | int]:
    """Overall class balance for a binary target: exact counts and rate."""
    y = y.astype(int)
    n = int(len(y))
    n_default = int(y.sum())
    n_non_default = n - n_default
    return {
        "n": n,
        "n_default": n_default,
        "n_non_default": n_non_default,
        "default_rate": n_default / n if n else float("nan"),
    }


def numeric_summary(
    df: pd.DataFrame,
    columns: list[str],
    skew_threshold: float = DEFAULT_SKEW_THRESHOLD,
) -> pd.DataFrame:
    """Distribution + summary stats for each numeric column, with a
    `log_transform_flag` for any feature whose |skewness| exceeds
    `skew_threshold`.

    Skewness and kurtosis use scipy's Fisher convention (kurtosis of a
    normal distribution is 0, not 3), computed on non-null values only.
    """
    rows = []
    for column in columns:
        series = df[column]
        clean = series.dropna().astype(float)
        skew = float(stats.skew(clean)) if len(clean) > 2 else float("nan")
        kurtosis = float(stats.kurtosis(clean)) if len(clean) > 2 else float("nan")
        rows.append(
            {
                "feature": column,
                "n": int(len(clean)),
                "n_missing": int(series.isna().sum()),
                "mean": float(clean.mean()) if len(clean) else float("nan"),
                "std": float(clean.std()) if len(clean) else float("nan"),
                "min": float(clean.min()) if len(clean) else float("nan"),
                "p25": float(clean.quantile(0.25)) if len(clean) else float("nan"),
                "median": float(clean.median()) if len(clean) else float("nan"),
                "p75": float(clean.quantile(0.75)) if len(clean) else float("nan"),
                "max": float(clean.max()) if len(clean) else float("nan"),
                "skew": skew,
                "kurtosis": kurtosis,
                "log_transform_flag": (
                    bool(np.abs(skew) > skew_threshold) if not np.isnan(skew) else False
                ),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def categorical_frequency(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, pd.DataFrame]:
    """Frequency table (count + pct of `n`) per categorical column, one
    DataFrame per column, sorted by count descending.
    """
    tables: dict[str, pd.DataFrame] = {}
    n = len(df)
    for column in columns:
        counts = df[column].value_counts(dropna=False)
        table = pd.DataFrame(
            {
                "count": counts,
                "pct": (counts / n * 100).round(2) if n else counts,
            }
        )
        table.index.name = column
        tables[column] = table
    return tables
