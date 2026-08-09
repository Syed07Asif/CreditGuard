"""Tests for creditguard.models.registry: model_registry promotion is
atomic (exactly one active row at a time), model_version is never reused,
and a deactivated model's artifact still loads.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from creditguard.db.repository import ModelRegistryRepository
from creditguard.models.base import BaseCreditModel
from creditguard.models.logistic import LogisticRegressionModel
from creditguard.models.registry import get_active_model, register_model


def _register(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "algorithm": "logistic_regression",
        "training_date": datetime.now(UTC),
        "dataset_version": "ds_test",
        "feature_list": ["a", "b"],
        "hyperparameters": {"C": 1.0},
        "metrics": {"pr_auc": 0.5},
        "mlflow_run_id": "run-000",
        "artifact_path": "models/artifacts/test.joblib",
    }
    defaults.update(overrides)
    return register_model(**defaults)


def test_register_model_sets_active_and_first_version_is_1_0_0() -> None:
    row = _register()
    assert row["is_active"] is True
    assert row["model_version"] == "1.0.0"


def test_registering_second_model_deactivates_first() -> None:
    first = _register(mlflow_run_id="run-a")
    second = _register(mlflow_run_id="run-b")

    assert second["is_active"] is True
    active = get_active_model()
    assert active is not None
    assert active["model_id"] == second["model_id"]

    repo = ModelRegistryRepository()
    first_row = repo.get_by_id(first["model_id"])
    assert first_row is not None
    assert first_row["is_active"] is False


def test_model_versions_are_never_reused() -> None:
    first = _register()
    second = _register()
    third = _register()
    versions = [first["model_version"], second["model_version"], third["model_version"]]
    assert versions == sorted(
        set(versions), key=lambda v: tuple(int(p) for p in v.split("."))
    )
    assert len(set(versions)) == 3


def test_get_active_model_returns_none_when_registry_empty() -> None:
    assert get_active_model() is None


def test_first_artifact_still_loads_after_second_registration(tmp_path) -> None:
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = pd.Series([0, 1, 0, 1])
    model = LogisticRegressionModel({"solver": "lbfgs", "max_iter": 1000})
    model.fit(X, y)
    artifact_path = tmp_path / "first_model.joblib"
    model.save(artifact_path)

    _register(artifact_path=str(artifact_path), mlflow_run_id="run-first")
    _register(mlflow_run_id="run-second")  # deactivates the first

    loaded = BaseCreditModel.load(artifact_path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X))
