"""The project's regression net (Phase 10): generate -> validate -> clean ->
features -> train -> register -> score via the API -> monitor, exercised
through the same functions each phase's own CLI already uses.

Deliberately writes into the real, git-ignored `data/`/`models/artifacts/`/
`reports/` directories (a uniquely-suffixed dataset_version keeps it from
colliding with anything real) rather than an isolated tmp_path: several
Phase 7-10 functions (`scoring.engine`'s cached model loader,
`monitoring.baseline`'s default artifact location) resolve their own paths
from `creditguard.config.get_settings()` without a full override surface,
the same way they do for a developer running the real pipeline by hand --
sandboxing this one test would mean testing a code path production never
takes. This matches the project's own established preference (see this
repo's MEMORY.md) for verifying against real behaviour over a narrower proxy.

Kept small (300 synthetic customers, one model family, a tiny search grid)
so it stays fast -- this is the reliability/speed the Phase 10 brief asks
for, not a full-scale training run (that's already covered, at real scale,
by Phase 6's own test suite).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import joblib
import pytest
from fastapi.testclient import TestClient

from creditguard.config import get_settings
from creditguard.data.generator import generate_dataset, load_config
from creditguard.data.versioning import (
    GeneratedDataset,
    make_dataset_version,
    read_dataset_tables,
    write_dataset,
)
from creditguard.db.repository import DriftReportRepository, PredictionRepository
from creditguard.features.build import build_features
from creditguard.models import registry
from creditguard.models.evaluate import select_best_model
from creditguard.models.train import (
    build_leaderboard,
    finalise_best_model,
    load_features_config,
    load_training_data,
    log_leaderboard_entry,
)
from creditguard.monitoring import baseline
from creditguard.monitoring.scheduler import run_monitoring_cycle
from creditguard.scoring import engine
from creditguard.validation.cleaning import DataCleaner
from creditguard.validation.engine import build_registry, load_rule_config
from creditguard.validation.engine import run as run_validation

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


@pytest.mark.slow
def test_full_pipeline_generate_to_monitor() -> None:
    settings = get_settings()
    run_id = uuid.uuid4().hex[:8]

    # -- generate -------------------------------------------------------
    generation_config = load_config("config/data_generation.yaml")
    dataset = generate_dataset(generation_config, seed=42, n_customers=300)
    dataset_version = (
        f"e2e{run_id}_{make_dataset_version(dataset.metadata['config'])[3:]}"
    )
    write_dataset(dataset, dataset_version, settings.data_dir / "raw")

    # -- validate ---------------------------------------------------------
    raw_tables = read_dataset_tables(dataset_version, settings.data_dir / "raw")
    rule_config = load_rule_config("config/validation_rules.yaml")
    validation_result = run_validation(
        raw_tables, build_registry(rule_config), dataset_version
    )
    # >= not == : Phase 2's generator deliberately injects a small number of
    # duplicate customer rows (see docs/data_generation.md), so the raw row
    # count can exceed n_customers slightly.
    assert validation_result.row_counts["customers"] >= 300

    # -- clean --------------------------------------------------------------
    cleaner = DataCleaner(rule_config)
    cleaned_tables = cleaner.fit_transform(raw_tables)
    clean_version = f"{dataset_version}_clean"
    cleaned_dataset = GeneratedDataset(
        customers=cleaned_tables["customers"],
        loan_applications=cleaned_tables["loan_applications"],
        financial_profiles=cleaned_tables["financial_profiles"],
        credit_history=cleaned_tables["credit_history"],
        loan_outcomes=cleaned_tables["loan_outcomes"],
        metadata={},
        injection_manifest={},
    )
    write_dataset(cleaned_dataset, clean_version, settings.data_dir / "processed")

    # -- features -------------------------------------------------------
    features_config = load_features_config("config/features.yaml")
    features_output_dir = settings.data_dir / "features" / clean_version
    build_result = build_features(
        dataset_version=clean_version,
        split_strategy="temporal",
        output_dir=features_output_dir,
        features_config=features_config,
        cleaning_config=rule_config,
    )
    assert build_result["metadata"]["n_features"] > 0

    # -- train (tiny/fast config -- see module docstring) ------------------
    data = load_training_data(clean_version, features_config, rule_config)
    model_config = _TINY_MODEL_CONFIG
    leaderboard = build_leaderboard(
        ["logistic_regression"], model_config, data, seed=42
    )
    for entry in leaderboard:
        log_leaderboard_entry(entry, clean_version, 42, data)
    best_entry = select_best_model(leaderboard, model_config["search"]["scoring"])
    final = finalise_best_model(
        best_entry,
        model_config,
        data,
        clean_version,
        42,
        settings.reports_dir / "figures" / "models",
    )

    model_path = settings.model_dir / f"e2e_model_{run_id}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["calibrated_model"], model_path)

    # -- register ------------------------------------------------------
    model_row = registry.register_model(
        algorithm=best_entry["family"],
        training_date=datetime.now(UTC),
        dataset_version=clean_version,
        feature_list=data.feature_names,
        hyperparameters=best_entry["params"],
        metrics={
            **final["test_metrics"],
            "chosen_threshold": final["threshold"]["chosen_threshold"],
        },
        mlflow_run_id=final["mlflow_run_id"],
        artifact_path=str(model_path),
    )
    model_id = model_row["model_id"]
    engine.reload_active_model()

    baseline_path = baseline.build_and_save_baseline_for_model(
        model_id,
        features_config_path="config/features.yaml",
        cleaning_config_path="config/validation_rules.yaml",
    )
    assert baseline_path.exists()

    # -- score via the API ---------------------------------------------
    # POST /applications (not the stateless /predict) so the point-in-time
    # customer/loan/financial/credit rows land in the database too --
    # monitoring's drift/data-quality checks below read from there.
    from creditguard.api.main import app
    from creditguard.api.schemas import _EXAMPLE_PAYLOAD

    payload = dict(_EXAMPLE_PAYLOAD)
    payload["customer_id"] = f"E2E-{run_id}"

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/applications",
            json=payload,
            headers={"X-API-Key": settings.api_key},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["latest_prediction"] is not None

    predictions = PredictionRepository().fetch_dataframe(filters={"model_id": model_id})
    assert len(predictions) >= 1

    # -- monitor -----------------------------------------------------------
    summary = run_monitoring_cycle()
    assert summary.get("error") is None
    # run_monitoring_cycle deliberately swallows each check's own exceptions
    # (one broken check shouldn't skip the others) -- so a passing "drift"
    # dict without an "error"/"skipped" key is the only way to know the
    # check actually completed cleanly, not just that persist_drift_findings
    # ran before some later step in run_drift_check failed silently.
    assert "drift" in summary
    assert "error" not in summary["drift"]
    assert "skipped" not in summary["drift"]
    assert "error" not in summary.get("performance", {})
    assert "error" not in summary.get("data_quality", {})

    drift_rows = DriftReportRepository().fetch_dataframe(filters={"model_id": model_id})
    assert not drift_rows.empty
