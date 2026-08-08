"""Tests for creditguard.eda.bivariate: the decile default-rate function
(returns 10 bins with monotone edges and full counts), band breakdowns,
correlation pairs, and point-biserial correlation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.eda.bivariate import (
    correlation_matrix,
    default_rate_by_band,
    default_rate_by_decile,
    high_correlation_pairs,
    point_biserial_correlations,
)


def _decile_fixture(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    score = np.arange(n, dtype=float)
    p_default = np.clip(0.9 - score / n, 0.02, 0.95)
    y = (rng.random(n) < p_default).astype(int)
    return pd.DataFrame({"score": score, "default_12m": y})


def test_default_rate_by_decile_returns_10_bins_with_monotone_edges() -> None:
    df = _decile_fixture(100)
    result = default_rate_by_decile(df, "score", n_bins=10)

    assert len(result) == 10
    assert result["n"].sum() == len(df)
    assert list(result["decile"]) == list(range(1, 11))
    assert (result["min_value"].diff().dropna() >= 0).all()
    assert (result["max_value"].diff().dropna() >= 0).all()


def test_default_rate_by_decile_handles_constant_column() -> None:
    df = pd.DataFrame({"const": [5.0] * 20, "default_12m": [0, 1] * 10})
    result = default_rate_by_decile(df, "const", n_bins=10)
    assert len(result) == 1
    assert result["n"].iloc[0] == 20


def test_default_rate_by_band_orders_by_category_order() -> None:
    df = pd.DataFrame(
        {
            "band": ["Q2", "Q1", "Q1", "Q3", "Q2"],
            "default_12m": [1, 0, 1, 0, 0],
        }
    )
    result = default_rate_by_band(df, "band", category_order=["Q1", "Q2", "Q3"])
    assert list(result["band"]) == ["Q1", "Q2", "Q3"]
    assert result["n"].sum() == len(df)


def test_correlation_and_high_correlation_pairs() -> None:
    df = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10], "c": [5, 3, 1, 4, 2]}
    )
    corr = correlation_matrix(df, ["a", "b", "c"])
    pairs = high_correlation_pairs(corr, threshold=0.8)
    assert any(
        p["feature_a"] == "a" and p["feature_b"] == "b" and p["r"] == pytest.approx(1.0)
        for p in pairs
    )


def test_point_biserial_correlation_direction() -> None:
    df = pd.DataFrame(
        {
            "risk_score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "default_12m": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )
    result = point_biserial_correlations(df, ["risk_score"], "default_12m")
    assert result.loc[0, "point_biserial_r"] < 0
