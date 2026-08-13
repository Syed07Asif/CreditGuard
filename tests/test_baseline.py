"""Tests for creditguard.monitoring.baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from creditguard.monitoring import baseline


def test_decile_edges_widen_outer_bounds_to_infinity() -> None:
    values = pd.Series(range(0, 100))
    edges = baseline._decile_edges(values, n_deciles=10)
    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert len(edges) >= 2


def test_decile_edges_handles_constant_feature() -> None:
    values = pd.Series([5.0] * 20)
    edges = baseline._decile_edges(values, n_deciles=10)
    assert edges[0] == float("-inf")
    assert edges[-1] == float("inf")
    assert len(edges) == 2


def test_build_baseline_profile_numeric_and_categorical() -> None:
    frame = pd.DataFrame(
        {
            "income": np.arange(100, dtype=float),
            "loan_type": ["PERSONAL"] * 60 + ["AUTO"] * 40,
        }
    )
    p_train = np.linspace(0.01, 0.5, 100)
    y_train = pd.Series([1 if i % 10 == 0 else 0 for i in range(100)])

    profile, sample = baseline.build_baseline_profile(
        model_id="m1",
        dataset_version="ds1",
        train_frame=frame,
        p_train=p_train,
        y_train=y_train,
        numeric_columns=["income"],
        categorical_columns=["loan_type"],
        training_window_start="2024-01-01",
        training_window_end="2024-06-01",
        n_deciles=5,
    )

    assert profile.model_id == "m1"
    assert profile.n_rows == 100
    assert profile.observed_default_rate == pytest.approx(0.1)
    assert "income" in profile.numeric_features
    assert profile.numeric_features["income"].mean == pytest.approx(49.5)
    assert "loan_type" in profile.categorical_features
    assert profile.categorical_features["loan_type"].frequencies[
        "PERSONAL"
    ] == pytest.approx(0.6)
    assert len(profile.prediction_probability_deciles) >= 2
    assert "income" in sample.columns
    assert "__p_default__" in sample.columns


def test_save_and_load_baseline_round_trip(tmp_path) -> None:
    frame = pd.DataFrame({"income": np.arange(50, dtype=float)})
    p_train = np.linspace(0.01, 0.3, 50)
    y_train = pd.Series([0] * 45 + [1] * 5)

    profile, sample = baseline.build_baseline_profile(
        model_id="m-roundtrip",
        dataset_version="ds1",
        train_frame=frame,
        p_train=p_train,
        y_train=y_train,
        numeric_columns=["income"],
        categorical_columns=[],
        training_window_start="2024-01-01",
        training_window_end="2024-06-01",
    )

    from creditguard.models import registry

    registry.register_model(
        model_id="m-roundtrip",
        algorithm="logistic_regression",
        training_date=pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime(),
        dataset_version="ds1",
        feature_list=["income"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.3},
        mlflow_run_id="run-1",
        artifact_path="unused",
    )

    saved_path = baseline.save_baseline(profile, sample, model_dir=tmp_path)
    assert saved_path.exists()

    loaded = baseline.load_baseline("m-roundtrip", model_dir=tmp_path)
    assert loaded.model_id == "m-roundtrip"
    assert loaded.numeric_features["income"].mean == pytest.approx(
        profile.numeric_features["income"].mean
    )

    loaded_sample = baseline.load_baseline_sample(loaded)
    assert len(loaded_sample) == len(sample)

    registered = registry.get_active_model()
    assert registered is not None
    assert registered["metrics"]["baseline"]["n_rows"] == profile.n_rows


def test_load_baseline_raises_when_missing(tmp_path) -> None:
    with pytest.raises(baseline.BaselineNotFoundError):
        baseline.load_baseline("no-such-model", model_dir=tmp_path)
