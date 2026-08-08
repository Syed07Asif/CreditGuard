"""Tests for creditguard.eda.risk_analysis: WOE/IV against a hand-computed
value, IV interpretation bands, and the temporal regime-shift check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from creditguard.eda.risk_analysis import (
    compute_woe_iv_categorical,
    detect_regime_shift,
    interpret_iv,
    iv_table,
    monthly_volume_and_default_rate,
)


def _two_bin_fixture() -> pd.DataFrame:
    # bin "A": n=5, n_default=3; bin "B": n=5, n_default=1.
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "default_12m": [1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
        }
    )


def test_woe_iv_categorical_matches_hand_computed_value() -> None:
    # Hand computation (Siddiqi WOE = ln(%non-event / %event)):
    #   dist_event = [3/4, 1/4], dist_non_event = [2/6, 4/6]
    #   woe_A = ln((2/6)/(3/4)) = ln(4/9) = -0.8109302162
    #   woe_B = ln((4/6)/(1/4)) = ln(8/3)  =  0.9808292530
    #   IV = (2/6-3/4)*woe_A + (4/6-1/4)*woe_B = 0.7465664455
    df = _two_bin_fixture()
    bin_df, iv = compute_woe_iv_categorical(df, "group", "default_12m")

    assert iv == pytest.approx(0.7465664455, rel=1e-6)
    woe_a = bin_df.loc[bin_df["group"] == "A", "woe"].iloc[0]
    woe_b = bin_df.loc[bin_df["group"] == "B", "woe"].iloc[0]
    assert woe_a == pytest.approx(-0.8109302162, rel=1e-6)
    assert woe_b == pytest.approx(0.9808292530, rel=1e-6)


def test_interpret_iv_bands() -> None:
    assert interpret_iv(0.01) == "useless"
    assert interpret_iv(0.05) == "weak"
    assert interpret_iv(0.2) == "medium"
    assert interpret_iv(0.4) == "strong"
    assert interpret_iv(0.6) == "suspiciously strong - investigate for leakage"


def test_woe_iv_zero_count_bin_uses_laplace_correction_not_a_crash() -> None:
    # Group "A" has zero defaults, group "B" is all defaults -- both would
    # divide by zero without the empty-cell correction.
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "default_12m": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    bin_df, iv = compute_woe_iv_categorical(df, "group", "default_12m")
    assert iv > 0
    assert bin_df["woe"].notna().all()


def test_iv_table_routes_low_cardinality_numeric_through_categorical_binning() -> None:
    # A zero-inflated count column with few distinct values: quantile deciles
    # would collapse every cut point onto 0 and report IV == 0, hiding real
    # signal. iv_table must route it through per-value (categorical) binning
    # instead, matching what happened with `previous_defaults` on real data.
    df = pd.DataFrame(
        {
            "prior_defaults": [0] * 80 + [1] * 15 + [2] * 5,
            "default_12m": [0] * 76 + [1] * 4 + [0] * 10 + [1] * 5 + [1] * 5,
        }
    )
    result = iv_table(
        df, numeric_columns=["prior_defaults"], categorical_columns=[], n_bins=10
    )
    assert result.loc[0, "iv"] > 0.02


def test_iv_table_sorted_descending_and_covers_all_columns() -> None:
    df = _two_bin_fixture()
    df["score"] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = iv_table(
        df,
        numeric_columns=["score"],
        categorical_columns=["group"],
        target_col="default_12m",
        n_bins=2,
    )
    assert set(result["feature"]) == {"score", "group"}
    assert list(result["iv"]) == sorted(result["iv"], reverse=True)


def test_monthly_volume_and_default_rate_and_regime_shift() -> None:
    dates = pd.to_datetime(
        ["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-15", "2024-03-01"]
    )
    df = pd.DataFrame({"application_date": dates, "default_12m": [0, 0, 0, 0, 1]})
    monthly = monthly_volume_and_default_rate(df)

    assert monthly["n"].sum() == len(df)
    assert list(monthly["month"]) == ["2024-01", "2024-02", "2024-03"]

    regime = detect_regime_shift(monthly, z_threshold=0.1)
    assert isinstance(regime["regime_shift_detected"], bool)
