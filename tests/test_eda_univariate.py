"""Tests for creditguard.eda.univariate: default rate summary, numeric
summary stats (skew/kurtosis/log-transform flag), and categorical frequency
tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.eda.univariate import (
    categorical_frequency,
    default_rate_summary,
    numeric_summary,
)


def test_default_rate_summary_exact_counts() -> None:
    y = pd.Series([0, 0, 0, 1, 1])
    result = default_rate_summary(y)
    assert result == {"n": 5, "n_default": 2, "n_non_default": 3, "default_rate": 0.4}


def test_numeric_summary_flags_high_skew_for_log_transform() -> None:
    rng = np.random.default_rng(42)
    skewed = pd.Series(rng.exponential(scale=1.0, size=1000))
    symmetric = pd.Series(rng.normal(loc=0, scale=1, size=1000))
    df = pd.DataFrame({"skewed": skewed, "symmetric": symmetric})

    result = numeric_summary(df, ["skewed", "symmetric"])
    assert result.loc["skewed", "log_transform_flag"]
    assert not result.loc["symmetric", "log_transform_flag"]
    assert result.loc["skewed", "n"] == 1000


def test_numeric_summary_counts_missing_values_separately() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    result = numeric_summary(df, ["x"])
    assert result.loc["x", "n"] == 4
    assert result.loc["x", "n_missing"] == 1


def test_categorical_frequency_counts_and_pct_sum_to_100() -> None:
    df = pd.DataFrame({"gender": ["MALE", "MALE", "FEMALE", "MALE", "FEMALE"]})
    tables = categorical_frequency(df, ["gender"])
    table = tables["gender"]
    assert table["count"].sum() == 5
    assert table.loc["MALE", "count"] == 3
    assert table["pct"].sum() == pytest.approx(100.0)
