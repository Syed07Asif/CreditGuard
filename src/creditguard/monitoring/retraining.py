"""Retraining: FR-026's decision (`should_retrain`) and action
(`trigger_retraining`), plus `rollback_to_version` as the safety valve.

`trigger_retraining` re-runs the Phase 6 training pipeline
(`creditguard.models.train`) on an already-prepared dataset version,
registers the result as a new, strictly-higher semantic version
(`creditguard.models.registry.register_model(..., activate=False)` --
CLAUDE.md hard rule 6: never overwrite, and this phase's own "does not
auto-promote"), and only flips it active
(`creditguard.models.registry.activate_model`) if it beats the current
champion on PR-AUC (by a configured margin, on a common holdout -- the
challenger's own untouched test split, since the champion was trained
before that data existed) and is no worse on calibration. A challenger that
loses stays registered but inactive, and an alert asks for human review
rather than the loss being silent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow

from creditguard.config import get_settings
from creditguard.db.repository import ModelRegistryRepository
from creditguard.models import registry, tracking
from creditguard.models.evaluate import full_metric_suite
from creditguard.models.train import build_leaderboard as _build_leaderboard
from creditguard.models.train import (
    finalise_best_model,
    load_features_config,
    load_model_config,
    load_training_data,
    log_leaderboard_entry,
)
from creditguard.monitoring import alerts
from creditguard.monitoring.baseline import (
    build_and_save_baseline_for_model,
    load_monitoring_config,
)
from creditguard.validation.engine import load_rule_config

DEFAULT_MODEL_CONFIG_PATH = "config/model_config.yaml"
DEFAULT_FEATURES_CONFIG_PATH = "config/features.yaml"
DEFAULT_CLEANING_CONFIG_PATH = "config/validation_rules.yaml"
DEFAULT_MONITORING_CONFIG_PATH = "config/monitoring.yaml"
DEFAULT_MODEL_FAMILIES: tuple[str, ...] = (
    "logistic_regression",
    "random_forest",
    "xgboost",
)


def count_new_labeled_loans(since: datetime, as_of: datetime | None = None) -> int:
    """Loans whose 12-month outcome was observed after `since` (typically a
    model's own `training_date`) and up to `as_of` -- "enough new labelled
    data" per FR-026.
    """
    from sqlalchemy import func, select

    from creditguard.db.engine import get_engine
    from creditguard.db.models import LoanOutcome

    as_of = as_of or datetime.now(UTC)
    since_date = since.date() if isinstance(since, datetime) else since
    with get_engine().connect() as conn:
        result = conn.execute(
            select(func.count())
            .select_from(LoanOutcome)
            .where(
                LoanOutcome.outcome_observed_date > since_date,
                LoanOutcome.outcome_observed_date <= as_of.date(),
            )
        )
        return int(result.scalar_one())


def should_retrain(
    *,
    model_id: str | None = None,
    performance_summary: dict[str, Any] | None = None,
    drift_result: Any | None = None,
    n_new_labeled_loans: int | None = None,
    monitoring_config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
    as_of: datetime | None = None,
) -> tuple[bool, list[str]]:
    """FR-026: retrain if ANY of --

      * a predictive performance metric has degraded beyond tolerance,
      * significant (DRIFT-status) feature/prediction drift was detected,
      * enough new labelled loans have accumulated since training.

    Each signal can be supplied directly (already-computed performance/
    drift results, or an explicit count) to test one trigger in isolation
    without needing the other two checks' full database/baseline
    infrastructure; any left as `None` is computed here against the real
    database via `performance.run_performance_check`/
    `drift.run_drift_check`/`count_new_labeled_loans`.

    Returns `(should_retrain, reasons)` -- `reasons` is empty iff
    `should_retrain` is `False`.
    """
    from creditguard.monitoring import drift as drift_module
    from creditguard.monitoring import performance as performance_module

    config = load_monitoring_config(monitoring_config_path)

    if model_id is None:
        model_row = registry.get_active_model()
        if model_row is None:
            raise RuntimeError("No active model registered -- nothing to evaluate.")
        model_id = model_row["model_id"]
    else:
        model_row = ModelRegistryRepository().get_by_id(model_id)
        if model_row is None:
            raise RuntimeError(f"No model_registry row for model_id={model_id!r}")

    reasons: list[str] = []

    if performance_summary is None:
        performance_summary = performance_module.run_performance_check(
            model_id=model_id,
            monitoring_config_path=monitoring_config_path,
            as_of=as_of,
        )
    if performance_summary["degraded_metrics"]:
        degraded_names = [d["metric"] for d in performance_summary["degraded_metrics"]]
        reasons.append(f"performance degraded beyond tolerance: {degraded_names}")

    if drift_result is None:
        drift_result = drift_module.run_drift_check(
            model_id=model_id,
            monitoring_config_path=monitoring_config_path,
            as_of=as_of,
        )
    if drift_result.any_drift:
        drifted_names = [
            f.feature_name
            for f in drift_result.findings
            if f.status == drift_module.STATUS_DRIFT
        ]
        reasons.append(f"significant drift detected: {drifted_names}")

    if n_new_labeled_loans is None:
        n_new_labeled_loans = count_new_labeled_loans(model_row["training_date"], as_of)
    min_required = config["retraining"]["min_new_labeled_loans"]
    if n_new_labeled_loans >= min_required:
        reasons.append(
            f"{n_new_labeled_loans} new labelled loans available "
            f"(>= configured minimum {min_required})"
        )

    return len(reasons) > 0, reasons


@dataclass(frozen=True)
class RetrainingResult:
    champion_model_id: str
    challenger_model_id: str
    promoted: bool
    comparison: dict[str, Any]


def _write_comparison_report(comparison: dict[str, Any], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = reports_dir / f"champion_vs_challenger_{timestamp}.md"
    verdict = "PROMOTED" if comparison["promoted"] else "NOT PROMOTED (stays inactive)"
    lines = [
        "# Champion vs challenger comparison",
        "",
        f"- champion: {comparison['champion_model_id']}",
        f"- challenger: {comparison['challenger_model_id']}",
        f"- verdict: **{verdict}**",
        "",
        "| metric | champion | challenger | required |",
        "|---|---:|---:|---|",
        (
            f"| PR-AUC | {comparison['champion_pr_auc']:.4f} | "
            f"{comparison['challenger_pr_auc']:.4f} | "
            f">= champion + {comparison['pr_auc_margin_required']:.4f} |"
        ),
        (
            f"| calibration slope | {comparison['champion_calibration_slope']:.4f} | "
            f"{comparison['challenger_calibration_slope']:.4f} | "
            f"within 1.0 +/- {comparison['calibration_slope_tolerance']:.2f} |"
        ),
        "",
        f"PR-AUC margin achieved: {comparison['pr_auc_margin_achieved']:.4f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _log_comparison_to_mlflow(comparison: dict[str, Any]) -> str:
    experiment_id = tracking.configure_tracking()
    with mlflow.start_run(
        experiment_id=experiment_id, run_name="champion_vs_challenger"
    ) as run:
        mlflow.set_tags(
            {
                "champion_model_id": comparison["champion_model_id"],
                "challenger_model_id": comparison["challenger_model_id"],
                "stage": "champion_vs_challenger",
            }
        )
        mlflow.log_metrics(
            {
                "champion_pr_auc": comparison["champion_pr_auc"],
                "challenger_pr_auc": comparison["challenger_pr_auc"],
                "pr_auc_margin_achieved": comparison["pr_auc_margin_achieved"],
                "champion_calibration_slope": comparison["champion_calibration_slope"],
                "challenger_calibration_slope": comparison[
                    "challenger_calibration_slope"
                ],
                "promoted": float(comparison["promoted"]),
            }
        )
        return run.info.run_id


def trigger_retraining(
    *,
    dataset_version: str,
    model_config_path: str = DEFAULT_MODEL_CONFIG_PATH,
    features_config_path: str = DEFAULT_FEATURES_CONFIG_PATH,
    cleaning_config_path: str = DEFAULT_CLEANING_CONFIG_PATH,
    monitoring_config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
    model_families: tuple[str, ...] = DEFAULT_MODEL_FAMILIES,
    data_dir: Path | None = None,
    reports_dir: Path | None = None,
) -> RetrainingResult:
    """Re-run Phase 6 training on `dataset_version` (an already generated/
    validated/cleaned/featurised dataset -- typically the extended dataset
    the pipeline orchestrator's earlier stages just produced), register the
    winner as a new inactive version, and promote it only if it beats the
    current champion.
    """
    settings = get_settings()
    model_config = load_model_config(model_config_path)
    features_config = load_features_config(features_config_path)
    cleaning_config = load_rule_config(cleaning_config_path)
    monitoring_config = load_monitoring_config(monitoring_config_path)
    seed = model_config["random_seed"]

    champion_row = registry.get_active_model()
    if champion_row is None:
        raise RuntimeError("No active champion model to compare a challenger against.")

    data = load_training_data(
        dataset_version, features_config, cleaning_config, data_dir
    )
    leaderboard = _build_leaderboard(list(model_families), model_config, data, seed)
    for entry in leaderboard:
        log_leaderboard_entry(entry, dataset_version, seed, data)

    scoring_metric = model_config["search"]["scoring"]
    from creditguard.models.evaluate import select_best_model

    best_entry = select_best_model(leaderboard, scoring_metric)
    final = finalise_best_model(
        best_entry,
        model_config,
        data,
        dataset_version,
        seed,
        Path("reports/figures/models"),
    )

    model_path = (
        settings.model_dir
        / f"calibrated_model_{dataset_version}_{best_entry['family']}_challenger.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["calibrated_model"], model_path)

    challenger_row = registry.register_model(
        algorithm=best_entry["family"],
        training_date=datetime.now(UTC),
        dataset_version=dataset_version,
        feature_list=data.feature_names,
        hyperparameters=best_entry["params"],
        metrics={
            **final["test_metrics"],
            "chosen_threshold": final["threshold"]["chosen_threshold"],
        },
        mlflow_run_id=final["mlflow_run_id"],
        artifact_path=str(model_path),
        activate=False,
    )

    # Common holdout: the CHALLENGER's own untouched test split -- the
    # champion was trained before this data existed, so scoring it here
    # (without retraining it) is a fair "how would each model do on this
    # slice" comparison, not a rematch on the champion's own home turf.
    champion_model = joblib.load(champion_row["artifact_path"])
    champion_threshold = float(champion_row["metrics"].get("chosen_threshold", 0.5))
    p_champion = champion_model.predict_proba(data.X_test)[:, 1]
    champion_test_metrics = full_metric_suite(
        data.y_test, p_champion, champion_threshold
    )

    retraining_config = monitoring_config["retraining"]
    margin = retraining_config["promotion_pr_auc_margin"]
    slope_tolerance = retraining_config["calibration_slope_tolerance"]

    challenger_pr_auc = final["test_metrics"]["pr_auc"]
    champion_pr_auc = champion_test_metrics["pr_auc"]
    challenger_slope = final["test_metrics"]["calibration_slope"]

    beats_on_pr_auc = (challenger_pr_auc - champion_pr_auc) >= margin
    calibration_ok = abs(challenger_slope - 1.0) <= slope_tolerance
    promoted = beats_on_pr_auc and calibration_ok

    comparison = {
        "champion_model_id": champion_row["model_id"],
        "challenger_model_id": challenger_row["model_id"],
        "champion_pr_auc": champion_pr_auc,
        "challenger_pr_auc": challenger_pr_auc,
        "pr_auc_margin_required": margin,
        "pr_auc_margin_achieved": challenger_pr_auc - champion_pr_auc,
        "champion_calibration_slope": champion_test_metrics["calibration_slope"],
        "challenger_calibration_slope": challenger_slope,
        "calibration_slope_tolerance": slope_tolerance,
        "promoted": promoted,
    }

    reports_dir = reports_dir or (settings.reports_dir / "models")
    _write_comparison_report(comparison, Path(reports_dir))
    _log_comparison_to_mlflow(comparison)

    if promoted:
        registry.activate_model(challenger_row["model_id"])
        from creditguard.scoring import engine as scoring_engine

        scoring_engine.reload_active_model()
        build_and_save_baseline_for_model(
            challenger_row["model_id"],
            features_config_path=features_config_path,
            cleaning_config_path=cleaning_config_path,
            monitoring_config_path=monitoring_config_path,
            data_dir=data_dir,
        )
    else:
        dispatcher = alerts.build_default_dispatcher(monitoring_config_path)
        dispatcher.dispatch(
            alerts.Alert(
                severity="WARNING",
                category="retraining",
                message=(
                    f"Challenger {challenger_row['model_id']} did not beat "
                    f"champion {champion_row['model_id']} -- staying "
                    "registered but inactive."
                ),
                triggering_values=comparison,
                recommended_action=(
                    "Review the champion-vs-challenger report under "
                    "reports/models/ and decide whether to manually promote "
                    "(monitoring.retraining.rollback_to_version can also "
                    "reactivate any prior version), discard, or investigate "
                    "further."
                ),
                model_id=challenger_row["model_id"],
            )
        )

    return RetrainingResult(
        champion_model_id=champion_row["model_id"],
        challenger_model_id=challenger_row["model_id"],
        promoted=promoted,
        comparison=comparison,
    )


def rollback_to_version(model_id: str) -> dict[str, Any]:
    """Reactivate a previous model version. Never touches any artifact
    (CLAUDE.md hard rule 6: artifacts are never overwritten, so any prior
    version's `.joblib` file is still exactly what it always was) --
    this only flips which `model_registry` row is `is_active`.
    """
    row = ModelRegistryRepository().get_by_id(model_id)
    if row is None:
        raise RuntimeError(f"No model_registry row for model_id={model_id!r}")
    activated = registry.activate_model(model_id)

    from creditguard.scoring import engine as scoring_engine

    scoring_engine.reload_active_model()
    return activated


def main(argv: list[str] | None = None) -> None:
    """CLI:

    python -m creditguard.monitoring.retraining check
    python -m creditguard.monitoring.retraining trigger --dataset-version ds_...
    python -m creditguard.monitoring.retraining rollback --model-id ...
    """
    parser = argparse.ArgumentParser(description="CreditGuard Phase 10 retraining.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="Evaluate should_retrain().")
    check_parser.add_argument("--model-id", default=None)

    trigger_parser = subparsers.add_parser("trigger", help="Run trigger_retraining().")
    trigger_parser.add_argument("--dataset-version", required=True)
    trigger_parser.add_argument("--models", default=",".join(DEFAULT_MODEL_FAMILIES))

    rollback_parser = subparsers.add_parser(
        "rollback", help="Run rollback_to_version()."
    )
    rollback_parser.add_argument("--model-id", required=True)

    args = parser.parse_args(argv)

    if args.command == "check":
        retrain, reasons = should_retrain(model_id=args.model_id)
        print(f"should_retrain={retrain}")
        for reason in reasons:
            print(f"  - {reason}")
    elif args.command == "trigger":
        families = tuple(
            name.strip() for name in args.models.split(",") if name.strip()
        )
        result = trigger_retraining(
            dataset_version=args.dataset_version, model_families=families
        )
        print(
            f"Champion {result.champion_model_id} vs challenger "
            f"{result.challenger_model_id}: promoted={result.promoted}"
        )
    elif args.command == "rollback":
        row = rollback_to_version(args.model_id)
        print(
            f"Reactivated model_id={row['model_id']} (version {row['model_version']})"
        )


if __name__ == "__main__":
    main()
