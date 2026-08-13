"""Tests for creditguard.monitoring.retraining."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from creditguard.data.versioning import TABLE_FILENAMES
from creditguard.models import registry
from creditguard.models.train import load_training_data
from creditguard.monitoring import drift, retraining
from creditguard.validation.engine import load_rule_config

# -- shared fixture: a tiny, perfectly-separable-by-income dataset ----------
# default_12m alternates 0/1 by row index and is tied to a large income gap,
# so a model fit on the real label separates near-perfectly while a model
# fit on a *shuffled* label performs close to chance -- a deterministic way
# to build a "weak champion" / "strong challenger" pair without flakiness.


def _tiny_tables(
    n_customers: int = 140, shuffle_seed: int | None = None
) -> dict[str, pd.DataFrame]:
    customer_ids = [f"C{i:04d}" for i in range(n_customers)]
    loan_ids = [f"L{i:04d}" for i in range(n_customers)]
    label = np.array([i % 2 for i in range(n_customers)])
    monthly_income = np.where(label == 0, 100000.0, 20000.0)
    annual_income = monthly_income * 12

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": [25 + (i % 40) for i in range(n_customers)],
            "gender": ["MALE", "FEMALE"] * (n_customers // 2),
            "marital_status": ["SINGLE", "MARRIED"] * (n_customers // 2),
            "dependents": [0, 1] * (n_customers // 2),
            "education": ["GRADUATE"] * n_customers,
            "employment_type": ["SALARIED"] * n_customers,
            "employment_years": [1.0 + (i % 20) for i in range(n_customers)],
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": ([1, 2, 3] * (n_customers // 3 + 1))[:n_customers],
        }
    )
    application_dates = [
        pd.Timestamp("2023-01-01") + pd.Timedelta(days=2 * i)
        for i in range(n_customers)
    ]
    loan_applications = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "customer_id": customer_ids,
            "loan_type": ["PERSONAL"] * n_customers,
            "loan_amount": [100000.0 + 1000 * i for i in range(n_customers)],
            "loan_tenure_months": [24] * n_customers,
            "interest_rate": [12.0] * n_customers,
            "loan_purpose": ["OTHER"] * n_customers,
            "application_date": application_dates,
            "decision_date": [d + pd.Timedelta(days=3) for d in application_dates],
            "status": ["APPROVED"] * n_customers,
        }
    )
    as_of_dates = [d - pd.Timedelta(days=5) for d in application_dates]
    financial_profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "monthly_income": monthly_income,
            "monthly_expenses": monthly_income * 0.3,
            "existing_loan_count": [0] * n_customers,
            "existing_loan_amount": [0.0] * n_customers,
            "monthly_emi": monthly_income * 0.1,
            "savings_balance": monthly_income * 6,
            "total_assets": annual_income * 0.5,
            "total_liabilities": [0.0] * n_customers,
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "credit_history_months": [24 + i for i in range(n_customers)],
            "num_credit_accounts": [2] * n_customers,
            "total_credit_limit": [200000.0] * n_customers,
            "total_outstanding": [50000.0] * n_customers,
            "credit_utilization": [0.25] * n_customers,
            "previous_defaults": label.tolist(),
            "late_payments_12m": [0] * n_customers,
            "missed_payments_12m": [0] * n_customers,
            "active_loans": [1] * n_customers,
            "closed_loans": [1] * n_customers,
        }
    )
    final_label = label
    if shuffle_seed is not None:
        final_label = np.random.default_rng(shuffle_seed).permutation(label)
    loan_outcomes = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "default_12m": final_label.tolist(),
            "outcome_observed_date": [
                d + pd.DateOffset(months=12) for d in application_dates
            ],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def _write_dataset(
    data_dir: Path, dataset_version: str, tables: dict[str, pd.DataFrame]
) -> None:
    out_dir = data_dir / dataset_version
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in TABLE_FILENAMES.items():
        tables[name].to_parquet(out_dir / f"{filename}.parquet", index=False)


def _features_config() -> dict:
    return {
        "behavioural": {"n_quantile_bins": 5},
        "preprocessing": {"min_frequency": 0.01},
        "feature_columns": {
            "numeric": [
                "age",
                "dependents",
                "employment_years",
                "annual_income",
                "city_tier",
                "loan_amount",
                "loan_tenure_months",
                "interest_rate",
                "monthly_income",
                "monthly_expenses",
                "existing_loan_count",
                "existing_loan_amount",
                "monthly_emi",
                "savings_balance",
                "total_assets",
                "total_liabilities",
                "credit_history_months",
                "num_credit_accounts",
                "total_credit_limit",
                "total_outstanding",
                "previous_defaults",
                "late_payments_12m",
                "missed_payments_12m",
                "active_loans",
                "closed_loans",
                "dti",
                "emi_to_income",
                "credit_utilization",
                "loan_to_income",
                "proposed_emi",
                "post_loan_dti",
                "savings_to_income",
                "net_worth",
                "leverage_ratio",
                "disposable_income",
                "months_of_runway",
                "delinquency_rate",
                "has_prior_default",
                "credit_history_years",
                "accounts_per_year",
                "active_loan_ratio",
                "employment_stability",
                "income_per_dependent",
            ],
            "categorical": [
                "gender",
                "marital_status",
                "education",
                "employment_type",
                "loan_type",
                "loan_purpose",
            ],
            "ordinal": ["utilization_band", "age_band", "tenure_band", "income_band"],
        },
    }


_TINY_MODEL_CONFIG = {
    "random_seed": 42,
    "search": {
        "strategy": "random",
        "n_iter": 2,
        "cv_folds": 2,
        "scoring": "average_precision",
    },
    "models": {
        "logistic_regression": {
            "fixed_params": {"solver": "saga", "max_iter": 200, "random_state": 42},
            "param_grid": {"C": [0.1, 1.0], "penalty": ["l2"], "l1_ratio": [0.0]},
        },
    },
    "imbalance": {"strategies": ["none"], "recall_at_precision": 0.5},
    "cost_matrix": {
        "cost_false_negative": 10.0,
        "cost_false_positive": 1.0,
        "cost_true_positive": 0.0,
        "cost_true_negative": 0.0,
    },
    "calibration": {"methods": ["sigmoid"]},
    "evaluation": {
        "segment_columns": ["loan_type"],
        "lift_gains_bins": 5,
        "cv_folds": 2,
    },
    "mlflow": {"experiment_name": "creditguard-test"},
}


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def retraining_fixture(tmp_path):
    """Writes a tiny, real dataset version to disk and registers a weak
    champion (fit on shuffled labels) trained through the exact same
    pipeline/config the challenger will use -- so their feature columns
    line up when the champion is later scored on the challenger's test split.
    """
    dataset_version = "ds_retrain_test"
    _write_dataset(tmp_path, dataset_version, _tiny_tables())

    features_config = _features_config()
    features_config_path = _write_yaml(tmp_path / "features.yaml", features_config)
    cleaning_config_path = Path("config/validation_rules.yaml")
    cleaning_config = load_rule_config(cleaning_config_path)

    model_config_path = _write_yaml(tmp_path / "model_config.yaml", _TINY_MODEL_CONFIG)

    data = load_training_data(
        dataset_version, features_config, cleaning_config, tmp_path
    )

    weak_tables = _tiny_tables(shuffle_seed=7)
    weak_version = "ds_retrain_test_weak_champion"
    _write_dataset(tmp_path, weak_version, weak_tables)
    weak_data = load_training_data(
        weak_version, features_config, cleaning_config, tmp_path
    )

    base = LogisticRegression(max_iter=200).fit(weak_data.X_train, weak_data.y_train)
    calibrated_champion = CalibratedClassifierCV(
        FrozenEstimator(base), method="sigmoid", cv=2
    )
    calibrated_champion.fit(weak_data.X_val, weak_data.y_val)

    champion_path = tmp_path / "champion.joblib"
    joblib.dump(calibrated_champion, champion_path)

    champion_row = registry.register_model(
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version=weak_version,
        feature_list=weak_data.feature_names,
        hyperparameters={},
        metrics={"chosen_threshold": 0.5, "pr_auc": 0.5},
        mlflow_run_id="champion-run",
        artifact_path=str(champion_path),
    )

    return {
        "tmp_path": tmp_path,
        "dataset_version": dataset_version,
        "features_config_path": str(features_config_path),
        "cleaning_config_path": str(cleaning_config_path),
        "model_config_path": str(model_config_path),
        "champion_row": champion_row,
        "data": data,
    }


# -- should_retrain: each FR-026 trigger in isolation ------------------------


def _clean_drift_result(model_id: str) -> drift.DriftRunResult:
    return drift.DriftRunResult(
        model_id=model_id,
        window_start="2024-01-01",
        window_end="2024-01-31",
        findings=[],
        concept_drift=drift.ConceptDriftResult(
            n=100,
            current_rate=0.1,
            ci_low=0.05,
            ci_high=0.15,
            baseline_rate=0.1,
            status="OK",
        ),
        n_rows_scored=0,
    )


def _drifted_drift_result(model_id: str) -> drift.DriftRunResult:
    result = _clean_drift_result(model_id)
    finding = drift.DriftFinding(
        feature_name="income",
        method="psi",
        baseline_stat=0.0,
        current_stat=0.5,
        drift_score=0.5,
        status=drift.STATUS_DRIFT,
    )
    return drift.DriftRunResult(**{**result.__dict__, "findings": [finding]})


def _register_dummy_model() -> str:
    row = registry.register_model(
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="ds1",
        feature_list=["income"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.5},
        mlflow_run_id="run-1",
        artifact_path="unused",
    )
    return str(row["model_id"])


def test_should_retrain_false_when_nothing_triggers() -> None:
    model_id = _register_dummy_model()
    should, reasons = retraining.should_retrain(
        model_id=model_id,
        performance_summary={"degraded_metrics": []},
        drift_result=_clean_drift_result(model_id),
        n_new_labeled_loans=0,
    )
    assert should is False
    assert reasons == []


def test_should_retrain_true_on_performance_degradation_alone() -> None:
    model_id = _register_dummy_model()
    should, reasons = retraining.should_retrain(
        model_id=model_id,
        performance_summary={
            "degraded_metrics": [
                {
                    "metric": "roc_auc",
                    "training_value": 0.8,
                    "current_value": 0.5,
                    "tolerance": 0.1,
                }
            ]
        },
        drift_result=_clean_drift_result(model_id),
        n_new_labeled_loans=0,
    )
    assert should is True
    assert any("performance" in r for r in reasons)


def test_should_retrain_true_on_significant_drift_alone() -> None:
    model_id = _register_dummy_model()
    should, reasons = retraining.should_retrain(
        model_id=model_id,
        performance_summary={"degraded_metrics": []},
        drift_result=_drifted_drift_result(model_id),
        n_new_labeled_loans=0,
    )
    assert should is True
    assert any("drift" in r for r in reasons)


def test_should_retrain_true_on_enough_new_labeled_data_alone() -> None:
    model_id = _register_dummy_model()
    should, reasons = retraining.should_retrain(
        model_id=model_id,
        performance_summary={"degraded_metrics": []},
        drift_result=_clean_drift_result(model_id),
        n_new_labeled_loans=10_000,  # far above monitoring.yaml's default minimum
    )
    assert should is True
    assert any("labelled" in r for r in reasons)


# -- trigger_retraining / rollback --------------------------------------


def _monitoring_config_with_retraining_overrides(retraining_overrides: dict) -> dict:
    """The real config/monitoring.yaml with just its `retraining` section
    overridden -- `trigger_retraining` also reads the `baseline` section
    (to build the challenger's baseline on promotion) and the `alerts`
    section (to build the not-promoted-challenger alert dispatcher), so a
    test-only config needs every section, not just `retraining`.
    """
    with open("config/monitoring.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["retraining"] = {**config["retraining"], **retraining_overrides}
    return config


def test_trigger_retraining_creates_new_version_and_old_artifact_stays_loadable(
    retraining_fixture,
) -> None:
    fx = retraining_fixture
    monitoring_config = _monitoring_config_with_retraining_overrides(
        {
            "promotion_pr_auc_margin": 0.01,
            # Generous on purpose: this test exercises the PR-AUC-driven
            # promotion mechanics and version/artifact bookkeeping, not
            # calibration precision on a 140-row toy dataset (that's
            # Phase 6's own, already-covered territory).
            "calibration_slope_tolerance": 5.0,
        }
    )
    monitoring_config_path = _write_yaml(
        fx["tmp_path"] / "monitoring.yaml", monitoring_config
    )

    result = retraining.trigger_retraining(
        dataset_version=fx["dataset_version"],
        model_config_path=fx["model_config_path"],
        features_config_path=fx["features_config_path"],
        cleaning_config_path=fx["cleaning_config_path"],
        monitoring_config_path=str(monitoring_config_path),
        model_families=("logistic_regression",),
        data_dir=fx["tmp_path"],
        reports_dir=fx["tmp_path"] / "reports",
    )

    assert result.champion_model_id == fx["champion_row"]["model_id"]
    assert result.challenger_model_id != fx["champion_row"]["model_id"]
    assert result.promoted is True
    assert result.comparison["challenger_pr_auc"] > result.comparison["champion_pr_auc"]

    active = registry.get_active_model()
    assert active["model_id"] == result.challenger_model_id

    old_champion = joblib.load(fx["champion_row"]["artifact_path"])
    assert old_champion.predict_proba(fx["data"].X_test)[:, 1] is not None


def test_trigger_retraining_does_not_promote_a_losing_challenger(
    retraining_fixture,
) -> None:
    fx = retraining_fixture
    monitoring_config = _monitoring_config_with_retraining_overrides(
        {
            # Impossibly high: PR-AUC is bounded by 1.0, so no challenger
            # can ever satisfy this margin -- a deterministic "loses" case.
            "promotion_pr_auc_margin": 0.99,
            "calibration_slope_tolerance": 5.0,
        }
    )
    monitoring_config_path = _write_yaml(
        fx["tmp_path"] / "monitoring.yaml", monitoring_config
    )

    result = retraining.trigger_retraining(
        dataset_version=fx["dataset_version"],
        model_config_path=fx["model_config_path"],
        features_config_path=fx["features_config_path"],
        cleaning_config_path=fx["cleaning_config_path"],
        monitoring_config_path=str(monitoring_config_path),
        model_families=("logistic_regression",),
        data_dir=fx["tmp_path"],
        reports_dir=fx["tmp_path"] / "reports",
    )

    assert result.promoted is False
    active = registry.get_active_model()
    assert active["model_id"] == fx["champion_row"]["model_id"]

    from creditguard.db.repository import ModelRegistryRepository

    challenger_row = ModelRegistryRepository().get_by_id(result.challenger_model_id)
    assert challenger_row is not None
    assert challenger_row["is_active"] is False


def test_rollback_to_version_restores_previous_active_model() -> None:
    first_id = _register_dummy_model()
    second_row = registry.register_model(
        algorithm="logistic_regression",
        training_date=datetime.now(UTC),
        dataset_version="ds2",
        feature_list=["income"],
        hyperparameters={},
        metrics={"chosen_threshold": 0.5},
        mlflow_run_id="run-2",
        artifact_path="unused",
    )
    assert registry.get_active_model()["model_id"] == second_row["model_id"]

    restored = retraining.rollback_to_version(first_id)
    assert restored["model_id"] == first_id
    assert registry.get_active_model()["model_id"] == first_id

    from creditguard.db.repository import ModelRegistryRepository

    second_after = ModelRegistryRepository().get_by_id(second_row["model_id"])
    assert second_after["is_active"] is False
