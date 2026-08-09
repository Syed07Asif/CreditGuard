"""Tests for creditguard.models.{base,logistic,random_forest,xgboost_model,
imbalance,calibration,threshold}: each model family trains and returns valid
probabilities, imbalance strategies never leak validation rows into a
resampler, and calibration never makes Brier score worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.models.calibration import calibrate_model
from creditguard.models.imbalance import (
    STRATEGIES,
    build_estimator_pipeline,
    class_weight_params,
    run_imbalance_experiment,
)
from creditguard.models.logistic import LogisticRegressionModel
from creditguard.models.random_forest import RandomForestModel
from creditguard.models.threshold import CostMatrix, find_min_cost_threshold
from creditguard.models.xgboost_model import XGBoostModel

MODEL_CLASSES_UNDER_TEST = [
    (LogisticRegressionModel, {"solver": "lbfgs", "max_iter": 1000}),
    (RandomForestModel, {"n_estimators": 50, "random_state": 42}),
    (XGBoostModel, {"n_estimators": 50, "random_state": 42, "device": "cpu"}),
]


def _fixture(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.normal(size=n) for i in range(5)})
    logits = 1.5 * X["f0"] - 1.0 * X["f1"]
    p = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.random(n) < p).astype(int))
    return X, y


@pytest.mark.parametrize("model_cls,params", MODEL_CLASSES_UNDER_TEST)
def test_model_trains_and_returns_probabilities_in_unit_interval(
    model_cls, params
) -> None:
    X, y = _fixture()
    model = model_cls(params)
    model.fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.parametrize("model_cls,params", MODEL_CLASSES_UNDER_TEST)
def test_feature_importance_indexed_by_feature_name(model_cls, params) -> None:
    X, y = _fixture()
    model = model_cls(params)
    model.fit(X, y)
    importances = model.feature_importance()
    assert set(importances.index) == set(X.columns)


def test_save_and_load_round_trip(tmp_path) -> None:
    X, y = _fixture()
    model = LogisticRegressionModel({"solver": "lbfgs", "max_iter": 1000})
    model.fit(X, y)
    path = tmp_path / "model.joblib"
    model.save(path)
    loaded = LogisticRegressionModel.load(path)
    np.testing.assert_allclose(model.predict_proba(X), loaded.predict_proba(X))


def test_xgboost_early_stopping_requires_and_uses_eval_set() -> None:
    X, y = _fixture(n=400)
    X_train, X_val = X.iloc[:300], X.iloc[300:]
    y_train, y_val = y.iloc[:300], y.iloc[300:]
    model = XGBoostModel(
        {
            "n_estimators": 200,
            "early_stopping_rounds": 5,
            "random_state": 42,
            "device": "cpu",
        }
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    assert model.model_.best_iteration is not None


def test_same_seed_same_data_reproduces_identical_metrics() -> None:
    X, y = _fixture()
    first = RandomForestModel({"n_estimators": 50, "random_state": 42})
    first.fit(X, y)
    second = RandomForestModel({"n_estimators": 50, "random_state": 42})
    second.fit(X, y)
    np.testing.assert_array_equal(first.predict_proba(X), second.predict_proba(X))


def test_class_weight_params_xgboost_computes_scale_pos_weight() -> None:
    y_train = pd.Series([0] * 80 + [1] * 20)
    params = class_weight_params("xgboost", {}, y_train, "class_weight")
    assert params["scale_pos_weight"] == pytest.approx(4.0)


def test_class_weight_params_logistic_uses_balanced_string() -> None:
    y_train = pd.Series([0] * 80 + [1] * 20)
    params = class_weight_params("logistic_regression", {}, y_train, "class_weight")
    assert params["class_weight"] == "balanced"


def test_class_weight_params_noop_for_non_class_weight_strategy() -> None:
    params = class_weight_params(
        "logistic_regression", {"C": 1.0}, pd.Series([0, 1]), "none"
    )
    assert "class_weight" not in params
    assert params == {"C": 1.0}


def test_smote_resampler_never_sees_validation_rows(monkeypatch) -> None:
    """SMOTE is applied inside CV/train-fit only: the resampler must never
    be given the held-out validation rows.
    """
    X, y = _fixture(n=300)
    X_train = X.iloc[:200].reset_index(drop=True)
    X_val = X.iloc[200:].reset_index(drop=True)
    y_train = y.iloc[:200].reset_index(drop=True)

    seen_row_counts: list[int] = []
    from imblearn.over_sampling import SMOTE as RealSMOTE

    class RecordingSMOTE(RealSMOTE):
        def fit_resample(self, X_in, y_in):
            seen_row_counts.append(len(X_in))
            return super().fit_resample(X_in, y_in)

    monkeypatch.setattr("creditguard.models.imbalance.SMOTE", RecordingSMOTE)

    estimator = build_estimator_pipeline(
        "logistic_regression",
        {"solver": "lbfgs", "max_iter": 1000},
        y_train,
        "smote",
        seed=42,
    )
    estimator.fit(X_train, y_train)
    estimator.predict_proba(X_val)  # must not trigger another resample call

    assert seen_row_counts == [len(X_train)]


def test_imbalance_experiment_runs_every_strategy() -> None:
    X, y = _fixture(n=300)
    X_train, X_val = X.iloc[:200].reset_index(drop=True), X.iloc[200:].reset_index(
        drop=True
    )
    y_train, y_val = y.iloc[:200].reset_index(drop=True), y.iloc[200:].reset_index(
        drop=True
    )

    results = run_imbalance_experiment(
        "logistic_regression",
        {"solver": "lbfgs", "max_iter": 1000},
        X_train,
        y_train,
        X_val,
        y_val,
        seed=42,
    )
    assert set(results.keys()) == set(STRATEGIES)
    for metrics in results.values():
        assert 0.0 <= metrics["brier_score"] <= 1.0


def test_calibration_improves_or_preserves_brier_score() -> None:
    X, y = _fixture(n=600)
    X_train = X.iloc[:400].reset_index(drop=True)
    X_val = X.iloc[400:].reset_index(drop=True)
    y_train = y.iloc[:400].reset_index(drop=True)
    y_val = y.iloc[400:].reset_index(drop=True)

    base = LogisticRegressionModel({"solver": "lbfgs", "max_iter": 1000, "C": 1000.0})
    base.fit(X_train, y_train)

    result = calibrate_model(base.model_, X_val, y_val)
    assert result["brier_after"] <= result["brier_before"] + 1e-9


def test_min_cost_threshold_decreases_as_false_negative_cost_increases() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=200)
    y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.2, size=200) + 0.2, 0, 1)

    low_fn_cost = CostMatrix(cost_false_negative=1.0, cost_false_positive=1.0)
    high_fn_cost = CostMatrix(cost_false_negative=50.0, cost_false_positive=1.0)

    t_low, _ = find_min_cost_threshold(y_true, y_prob, low_fn_cost)
    t_high, _ = find_min_cost_threshold(y_true, y_prob, high_fn_cost)
    assert t_high <= t_low
