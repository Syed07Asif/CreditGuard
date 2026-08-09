"""Tests for creditguard.models.evaluate: hand-computed metrics, the FR-010
accuracy-selection guard, and cross-validation/per-segment breakdowns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.models.evaluate import (
    calibration_slope_intercept,
    classification_metrics_at_threshold,
    compute_gini,
    compute_ks_statistic,
    confusion_counts_at_threshold,
    full_metric_suite,
    lift_gains_table,
    per_segment_metrics,
    recall_at_precision,
    select_best_model,
    stratified_cv_metrics,
)


def test_classification_metrics_at_threshold_hand_computed() -> None:
    # threshold=0.5 -> pred=[0,1,0,1]: tp=1(idx3), fp=1(idx1), fn=1(idx2), tn=1(idx0)
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.6, 0.4, 0.9]
    m = classification_metrics_at_threshold(y_true, y_prob, 0.5)
    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_confusion_counts_at_threshold_hand_computed() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.6, 0.4, 0.9]
    counts = confusion_counts_at_threshold(y_true, y_prob, 0.5)
    assert counts == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_compute_gini_hand_computed() -> None:
    assert compute_gini(0.75) == pytest.approx(0.5)
    assert compute_gini(0.5) == pytest.approx(0.0)
    assert compute_gini(1.0) == pytest.approx(1.0)


def test_compute_ks_statistic_perfect_separation() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    assert compute_ks_statistic(y_true, y_prob) == pytest.approx(1.0)


def test_compute_ks_statistic_no_separation() -> None:
    y_true = [0, 1, 0, 1]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    assert compute_ks_statistic(y_true, y_prob) == pytest.approx(0.0)


def test_calibration_slope_intercept_close_to_identity_when_well_calibrated() -> None:
    # y generated exactly from p via the logistic relationship -- with enough
    # samples the Cox regression should recover slope~=1, intercept~=0.
    rng = np.random.default_rng(42)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.random(5000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_recall_at_precision_bounded() -> None:
    y_true = [0, 0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.9, 0.8, 0.95]
    r = recall_at_precision(y_true, y_prob, 0.5)
    assert 0.0 <= r <= 1.0


def test_recall_at_precision_zero_when_unreachable() -> None:
    y_true = [0, 1, 0, 1]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    assert recall_at_precision(y_true, y_prob, 0.99) == 0.0


def test_lift_gains_table_decile_1_is_highest_risk() -> None:
    rng = np.random.default_rng(42)
    n = 200
    p = np.linspace(0, 1, n)
    y = (rng.random(n) < p).astype(int)
    table = lift_gains_table(y, p, n_bins=10)
    assert len(table) == 10
    assert int(table["n"].sum()) == n
    rate_decile_1 = table.loc[table["decile"] == 1, "default_rate"].iloc[0]
    rate_decile_10 = table.loc[table["decile"] == 10, "default_rate"].iloc[0]
    assert rate_decile_1 > rate_decile_10


def test_full_metric_suite_reproducible_with_fixed_seed() -> None:
    def _generate() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=500)
        p = rng.uniform(0, 1, size=500)
        return y, p

    y1, p1 = _generate()
    y2, p2 = _generate()
    assert full_metric_suite(y1, p1) == full_metric_suite(y2, p2)


def test_select_best_model_raises_on_accuracy() -> None:
    results = [{"name": "a", "metrics": {"accuracy": 0.9, "average_precision": 0.2}}]
    with pytest.raises(ValueError, match="accuracy"):
        select_best_model(results, "accuracy")


def test_select_best_model_picks_highest_metric() -> None:
    results = [
        {"name": "a", "metrics": {"average_precision": 0.3}},
        {"name": "b", "metrics": {"average_precision": 0.6}},
    ]
    assert select_best_model(results, "average_precision")["name"] == "b"


def test_stratified_cv_metrics_returns_mean_and_std_rows() -> None:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "b", "c"])
    y = pd.Series((rng.random(200) < 0.3).astype(int))

    class _DummyModel:
        def fit(self, X: pd.DataFrame, y: pd.Series) -> _DummyModel:
            self.p_ = float(y.mean())
            return self

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            return np.full(len(X), self.p_)

    result = stratified_cv_metrics(_DummyModel, X, y, n_splits=5, seed=42)
    assert list(result.index) == ["mean", "std"]
    assert "pr_auc" in result.columns


def test_per_segment_metrics_breaks_out_by_column() -> None:
    segment_frame = pd.DataFrame({"loan_type": ["A", "A", "B", "B"]})
    y_true = [0, 1, 0, 1]
    y_prob = [0.2, 0.8, 0.3, 0.7]
    result = per_segment_metrics(segment_frame, y_true, y_prob, ["loan_type"])
    assert set(result["loan_type"]["loan_type"]) == {"A", "B"}
    assert result["loan_type"]["n"].sum() == 4
