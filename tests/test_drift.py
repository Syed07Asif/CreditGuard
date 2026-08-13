"""Tests for creditguard.monitoring.drift -- PSI/KS/chi-square math and
status-band verdicts on constructed fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from creditguard.monitoring import drift

EPSILON = 1e-6


def test_psi_from_percentages_matches_hand_computed_value() -> None:
    # PSI = sum((actual - expected) * ln(actual / expected))
    # expected = [0.9, 0.1], actual = [0.5, 0.5]
    # (0.5-0.9)*ln(0.5/0.9) + (0.5-0.1)*ln(0.5/0.1)
    expected = np.array([0.9, 0.1])
    actual = np.array([0.5, 0.5])
    hand_computed = (-0.4) * math.log(0.5 / 0.9) + 0.4 * math.log(0.5 / 0.1)

    result = drift.psi_from_percentages(expected, actual, EPSILON)
    assert result == pytest.approx(hand_computed, rel=1e-9)
    assert result == pytest.approx(0.8789, abs=1e-3)


def test_psi_from_percentages_identical_distributions_is_near_zero() -> None:
    expected = np.array([0.25, 0.25, 0.25, 0.25])
    actual = np.array([0.25, 0.25, 0.25, 0.25])
    assert drift.psi_from_percentages(expected, actual, EPSILON) == pytest.approx(0.0)


def test_psi_from_percentages_handles_empty_bin_with_epsilon() -> None:
    # An expected bin of exactly 0 would otherwise divide by zero / ln(0).
    expected = np.array([1.0, 0.0])
    actual = np.array([0.5, 0.5])
    result = drift.psi_from_percentages(expected, actual, EPSILON)
    assert math.isfinite(result)
    assert result > 0


def test_psi_status_bands() -> None:
    assert drift.psi_status(0.05, warning=0.10, drift=0.25) == drift.STATUS_OK
    assert drift.psi_status(0.15, warning=0.10, drift=0.25) == drift.STATUS_WARNING
    assert drift.psi_status(0.30, warning=0.10, drift=0.25) == drift.STATUS_DRIFT


def test_numeric_psi_identical_distribution_gives_psi_near_zero() -> None:
    edges = [-math.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, math.inf]
    current = pd.Series(range(0, 100))  # uniform 10% per bin, matching baseline
    psi_value = drift.numeric_psi(edges, current, EPSILON)
    assert psi_value < 0.02


def test_numeric_psi_shifted_distribution_crosses_drift_threshold() -> None:
    edges = [-math.inf, 10, 20, 30, 40, 50, 60, 70, 80, 90, math.inf]
    current = pd.Series([95] * 100)  # entirely in the top bin
    psi_value = drift.numeric_psi(edges, current, EPSILON)
    assert drift.psi_status(psi_value, 0.10, 0.25) == drift.STATUS_DRIFT


def test_categorical_psi_and_new_category_detection() -> None:
    baseline_frequencies = {"A": 0.5, "B": 0.5}
    same = pd.Series(["A"] * 50 + ["B"] * 50)
    psi_value, new_categories = drift.categorical_psi(
        baseline_frequencies, same, EPSILON
    )
    assert psi_value == pytest.approx(0.0, abs=1e-6)
    assert new_categories == []

    with_new_category = pd.Series(["A"] * 40 + ["B"] * 40 + ["C"] * 20)
    _, new_categories = drift.categorical_psi(
        baseline_frequencies, with_new_category, EPSILON
    )
    assert new_categories == ["C"]


def test_ks_test_feature_same_distribution_is_not_significant() -> None:
    rng = np.random.default_rng(42)
    baseline_values = pd.Series(rng.normal(0, 1, 500))
    current_values = pd.Series(rng.normal(0, 1, 500))
    statistic, p_value = drift.ks_test_feature(baseline_values, current_values)
    assert p_value > 0.05


def test_ks_test_feature_shifted_distribution_is_significant() -> None:
    rng = np.random.default_rng(42)
    baseline_values = pd.Series(rng.normal(0, 1, 500))
    current_values = pd.Series(rng.normal(5, 1, 500))
    statistic, p_value = drift.ks_test_feature(baseline_values, current_values)
    assert p_value < 0.05
    assert statistic > 0.5


def test_chi_square_test_feature_same_distribution_is_not_significant() -> None:
    baseline_frequencies = {"A": 0.5, "B": 0.5}
    current = pd.Series(["A"] * 250 + ["B"] * 250)
    statistic, p_value, new_categories = drift.chi_square_test_feature(
        baseline_frequencies, current
    )
    assert p_value > 0.05
    assert new_categories == []


def test_chi_square_test_feature_shifted_distribution_is_significant() -> None:
    baseline_frequencies = {"A": 0.5, "B": 0.5}
    current = pd.Series(["A"] * 480 + ["B"] * 20)
    statistic, p_value, _ = drift.chi_square_test_feature(baseline_frequencies, current)
    assert p_value < 0.05


def test_concept_drift_proxy_matching_rate_is_ok() -> None:
    rng = np.random.default_rng(42)
    outcomes = pd.Series(rng.binomial(1, 0.10, size=2000))
    result = drift.concept_drift_proxy(0.10, outcomes)
    assert result.status == drift.STATUS_OK
    assert result.ci_low <= 0.10 <= result.ci_high


def test_concept_drift_proxy_shifted_rate_is_warning() -> None:
    outcomes = pd.Series([1] * 80 + [0] * 20)  # 80% default rate, tight n=100
    result = drift.concept_drift_proxy(0.10, outcomes)
    assert result.status == drift.STATUS_WARNING
    assert not (result.ci_low <= 0.10 <= result.ci_high)


def test_concept_drift_proxy_empty_matured_loans_is_ok() -> None:
    result = drift.concept_drift_proxy(0.10, pd.Series(dtype=int))
    assert result.n == 0
    assert result.status == drift.STATUS_OK
