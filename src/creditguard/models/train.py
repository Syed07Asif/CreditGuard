"""CLI: search, evaluate, calibrate, threshold-optimise and (optionally)
register the best CreditGuard model.

    python -m creditguard.models.train --dataset-version ds_..._clean \\
        --models logistic,random_forest,xgboost --imbalance-grid --register-best

Reuses the Phase 4 feature pipeline's own components (`temporal_split`,
`filter_tables`, `align_target`, `build_feature_pipeline`) rather than
reading pre-built `X_*.parquet` files, so this CLI needs only the clean
dataset version on disk. It keeps the pre-`ColumnTransformer` frame
(`train_frame`/`val_frame`/`test_frame`) alongside the numeric matrices,
row-aligned by construction, because `evaluate.per_segment_metrics` needs
human-readable `loan_type`/`age_band`/`income_band` values that don't survive
one-hot/ordinal encoding.

Pipeline, in order:
  1. RandomizedSearchCV per requested model family (StratifiedKFold,
     scoring=average_precision -- never accuracy), searched under the
     `class_weight` imbalance strategy (see `reports/eda/findings.md`'s
     "Decisions for Phase 6": class weighting over resampling for the model
     that actually gets registered). XGBoost's winning params are then
     refit once more with early stopping against the real validation split.
  2. Leaderboard across families, evaluated on the validation split.
  3. Optional `--imbalance-grid`: every family x every strategy in
     `config/model_config.yaml`'s `imbalance.strategies`, written to
     `reports/models/imbalance_comparison.md`.
  4. `evaluate.select_best_model` picks the winner (never on accuracy).
  5. `calibration.calibrate_model` wraps the winner's raw estimator,
     validation-set Brier score picks isotonic vs. sigmoid.
  6. `threshold.optimise_threshold` picks the expected-cost-minimising
     threshold on the calibrated validation-set probabilities.
  7. Final evaluation on the untouched test split: overall + per-segment +
     lift/gains, at the chosen threshold.
  8. Every stage logs an MLflow run; `--register-best` also writes a
     `model_registry` row.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from creditguard.config import get_settings
from creditguard.data.versioning import read_dataset_tables
from creditguard.features.build import align_target, filter_tables, temporal_split
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.models import calibration, imbalance, registry, threshold, tracking
from creditguard.models.evaluate import (
    full_metric_suite,
    lift_gains_table,
    per_segment_metrics,
    select_best_model,
    stratified_cv_metrics,
)
from creditguard.validation.engine import load_rule_config

TARGET_COL = "default_12m"

# The task brief's `--models` example uses short names; map them onto the
# `config/model_config.yaml` / `imbalance.MODEL_CLASSES` keys.
MODEL_FAMILY_ALIASES: dict[str, str] = {
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest",
    "xgboost": "xgboost",
}

# Params that encode an imbalance strategy rather than a model architecture
# choice -- stripped before reusing a family's winning search params as the
# `base_params` for the imbalance-strategy grid, so `imbalance.py` can apply
# each strategy's own weighting/resampling without conflicting with it.
_IMBALANCE_PARAM_KEYS: frozenset[str] = frozenset(
    {"class_weight", "scale_pos_weight", "early_stopping_rounds"}
)


def load_model_config(path: str | Path) -> dict[str, Any]:
    """Load config/model_config.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features_config(path: str | Path) -> dict[str, Any]:
    """Load config/features.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TrainingData:
    """Everything a training run needs: numeric matrices for every split,
    the pre-`ColumnTransformer` frames (for segment labels), and the fitted
    feature pipeline (for versioning/persistence).
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    feature_names: list[str]


def load_training_data(
    dataset_version: str,
    features_config: dict[str, Any],
    cleaning_config: dict[str, Any],
    data_dir: Path | None = None,
) -> TrainingData:
    """Build train/val/test matrices (and their pre-encoding frames) the
    same way `creditguard.features.build.build_features` does, but keep the
    intermediate ratios+behavioural frame around for segment labels.
    """
    settings = get_settings()
    data_root = data_dir if data_dir else settings.data_dir / "processed"
    tables = read_dataset_tables(dataset_version, data_root)

    train_ids, val_ids, test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    val_tables = filter_tables(tables, val_ids)
    test_tables = filter_tables(tables, test_ids)

    pipeline = build_feature_pipeline(features_config, cleaning_config)
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    ratios_behavioural = pipeline[1:-1]
    preprocess = pipeline.named_steps["preprocess"]

    train_merged = merge_step.fit_transform(train_tables)
    val_merged = merge_step.transform(val_tables)
    test_merged = merge_step.transform(test_tables)

    y_train = align_target(train_merged, train_tables["loan_outcomes"]).reset_index(
        drop=True
    )
    y_val = align_target(val_merged, val_tables["loan_outcomes"]).reset_index(drop=True)
    y_test = align_target(test_merged, test_tables["loan_outcomes"]).reset_index(
        drop=True
    )

    train_frame = ratios_behavioural.fit_transform(train_merged).reset_index(drop=True)
    val_frame = ratios_behavioural.transform(val_merged).reset_index(drop=True)
    test_frame = ratios_behavioural.transform(test_merged).reset_index(drop=True)

    X_train_arr = preprocess.fit_transform(train_frame, y_train)
    X_val_arr = preprocess.transform(val_frame)
    X_test_arr = preprocess.transform(test_frame)

    feature_names = list(preprocess.get_feature_names_out())
    return TrainingData(
        X_train=pd.DataFrame(X_train_arr, columns=feature_names),
        y_train=y_train,
        X_val=pd.DataFrame(X_val_arr, columns=feature_names),
        y_val=y_val,
        X_test=pd.DataFrame(X_test_arr, columns=feature_names),
        y_test=y_test,
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        feature_names=feature_names,
    )


def _strip_imbalance_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in _IMBALANCE_PARAM_KEYS}


def search_model_family(
    model_family: str,
    model_config: dict[str, Any],
    data: TrainingData,
    seed: int,
) -> dict[str, Any]:
    """RandomizedSearchCV for one model family under the `class_weight`
    imbalance strategy, then (XGBoost only) a final refit of the winning
    params with early stopping against the real validation split.

    Returns `{"family", "estimator", "params", "cv_score", "metrics"}`
    where `estimator` is the fitted raw sklearn/xgboost estimator and
    `metrics` is `full_metric_suite` on the validation split.
    """
    family_config = model_config["models"][model_family]
    search_config = model_config["search"]

    fixed_params = dict(family_config.get("fixed_params", {}))
    fixed_params = imbalance.class_weight_params(
        model_family, fixed_params, data.y_train, "class_weight"
    )

    # Parallelism strategy differs for XGBoost: running many XGBoost fits in
    # separate joblib worker *processes* at once (RandomizedSearchCV's
    # n_jobs=-1) crashes with "Check failed: err == cudaGetLastError()"
    # on this machine -- confirmed to be a race in XGBoost's CUDA-runtime
    # probing when several processes touch it simultaneously, reproducible
    # even with device="cpu" and CUDA_VISIBLE_DEVICES="" (both otherwise
    # inert here since there's no working GPU). Serial search (search
    # n_jobs=1) with XGBoost's own multi-threading (n_jobs=-1 *within* the
    # one process) sidesteps the race entirely and was verified safe at
    # full 67k-row scale. Every other family is unaffected (no CUDA calls
    # at all) and keeps the faster process-parallel search.
    if model_family == "xgboost":
        search_n_jobs = 1
        search_fixed_params = dict(fixed_params)
    else:
        search_n_jobs = -1
        # Avoid oversubscribing CPU threads: parallelise across CV trials
        # (RandomizedSearchCV's n_jobs), not within each individual fit.
        search_fixed_params = dict(fixed_params)
        if "n_jobs" in search_fixed_params:
            search_fixed_params["n_jobs"] = 1

    search_estimator = imbalance.MODEL_CLASSES[model_family]().build(
        search_fixed_params
    )
    param_grid = family_config["param_grid"]
    cv = StratifiedKFold(
        n_splits=search_config["cv_folds"], shuffle=True, random_state=seed
    )

    with warnings.catch_warnings():
        # logistic_regression's grid pairs every `penalty` with every
        # `l1_ratio`; sklearn warns (doesn't error) when l1_ratio is set
        # for penalty="l2", where it's simply unused. Expected, not a bug.
        warnings.filterwarnings("ignore", message=".*l1_ratio.*", category=UserWarning)
        # sklearn 1.8+ soft-deprecated the `penalty` param itself in favour
        # of `l1_ratio`/`C` alone; the Phase 6 brief explicitly asks for an
        # "L2 and elastic-net" grid via `penalty`, which still works
        # correctly on the installed sklearn (1.9) -- just noisy.
        warnings.filterwarnings(
            "ignore", message=".*'penalty' was deprecated.*", category=FutureWarning
        )
        search = RandomizedSearchCV(
            estimator=search_estimator,
            param_distributions=param_grid,
            n_iter=search_config["n_iter"],
            scoring=search_config["scoring"],
            cv=cv,
            random_state=seed,
            n_jobs=search_n_jobs,
            refit=False,
            error_score="raise",
        )
        search.fit(data.X_train, data.y_train)

    best_params = {**fixed_params, **search.best_params_}

    if model_family == "xgboost":
        refit_params = dict(best_params)
        refit_params["early_stopping_rounds"] = family_config["early_stopping_rounds"]
        wrapper = imbalance.MODEL_CLASSES[model_family](refit_params)
        wrapper.fit(data.X_train, data.y_train, eval_set=(data.X_val, data.y_val))
    else:
        wrapper = imbalance.MODEL_CLASSES[model_family](best_params)
        wrapper.fit(data.X_train, data.y_train)

    p_val = wrapper.predict_proba(data.X_val)
    metrics = full_metric_suite(data.y_val, p_val)
    return {
        "family": model_family,
        "wrapper": wrapper,
        "estimator": wrapper.model_,
        "params": best_params,
        "cv_score": float(search.best_score_),
        "metrics": metrics,
    }


def build_leaderboard(
    model_families: list[str],
    model_config: dict[str, Any],
    data: TrainingData,
    seed: int,
) -> list[dict[str, Any]]:
    """`search_model_family` for every requested family, sorted by the
    search's scoring metric on the validation split, descending.
    """
    scoring = model_config["search"]["scoring"]
    results = [
        search_model_family(family, model_config, data, seed)
        for family in model_families
    ]
    return sorted(results, key=lambda r: r["metrics"][scoring], reverse=True)


def _feature_pipeline_version(dataset_version: str) -> str:
    return f"{dataset_version}_feat"


def log_leaderboard_entry(
    entry: dict[str, Any],
    dataset_version: str,
    seed: int,
    data: TrainingData,
) -> str:
    """MLflow-log one leaderboard entry's search result. Logs the raw fitted
    sklearn/xgboost estimator (`entry["estimator"]`), not the
    `BaseCreditModel` wrapper -- mlflow's sklearn flavor expects a genuine
    scikit-learn-API object (2-column `predict_proba`, `get_params`, etc.)
    for reliable signature inference and later loading.
    """
    return tracking.log_training_run(
        model=entry["estimator"],
        params=entry["params"],
        metrics={**entry["metrics"], f"cv_{entry['family']}_score": entry["cv_score"]},
        dataset_version=dataset_version,
        feature_pipeline_version=_feature_pipeline_version(dataset_version),
        seed=seed,
        model_family=entry["family"],
        imbalance_strategy="class_weight",
        X_example=data.X_train.head(50),
        run_name=f"{entry['family']}_search_best",
    )


def run_imbalance_grid(
    leaderboard: list[dict[str, Any]],
    model_config: dict[str, Any],
    data: TrainingData,
    dataset_version: str,
    seed: int,
    output_path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """`imbalance.run_all_imbalance_experiments` using each leaderboard
    entry's winning hyperparameters as that family's `base_params`, logs
    every (family, strategy) cell as its own MLflow run, and writes the
    comparison report.
    """
    strategies = tuple(model_config["imbalance"]["strategies"])
    recall_target = model_config["imbalance"]["recall_at_precision"]
    family_params = {
        entry["family"]: _strip_imbalance_params(entry["params"])
        for entry in leaderboard
    }

    results = imbalance.run_all_imbalance_experiments(
        family_params,
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        seed,
        strategies=strategies,
        recall_at_precision_target=recall_target,
    )

    for family, strategy_results in results.items():
        for strategy_name, metrics in strategy_results.items():
            estimator = imbalance.build_estimator_pipeline(
                family, family_params[family], data.y_train, strategy_name, seed
            )
            estimator.fit(data.X_train, data.y_train)
            tracking.log_training_run(
                model=estimator,
                params=family_params[family],
                metrics=metrics,
                dataset_version=dataset_version,
                feature_pipeline_version=_feature_pipeline_version(dataset_version),
                seed=seed,
                model_family=family,
                imbalance_strategy=strategy_name,
                X_example=data.X_train.head(50),
                run_name=f"{family}_{strategy_name}",
            )

    imbalance.write_imbalance_comparison_report(results, output_path, recall_target)
    return results


def finalise_best_model(
    best_entry: dict[str, Any],
    model_config: dict[str, Any],
    data: TrainingData,
    dataset_version: str,
    seed: int,
    figures_dir: Path,
) -> dict[str, Any]:
    """Calibrate the winning model, optimise its decision threshold, and
    evaluate it on the untouched test split.
    """
    calib_methods = tuple(model_config["calibration"]["methods"])
    calib_result = calibration.calibrate_model(
        best_entry["estimator"], data.X_val, data.y_val, methods=calib_methods
    )
    calibrated_model = calib_result["calibrated_model"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = figures_dir / "reliability_diagram.png"
    calibration.plot_reliability_diagram(
        data.y_val,
        calib_result["p_before"],
        calib_result["p_after"],
        save_path=reliability_path,
    )

    cost_matrix = threshold.CostMatrix.from_config(model_config["cost_matrix"])
    threshold_result = threshold.optimise_threshold(
        data.y_val, calib_result["p_after"], cost_matrix
    )
    chosen_threshold = threshold_result["chosen_threshold"]

    p_test = calibrated_model.predict_proba(data.X_test)[:, 1]
    test_metrics = full_metric_suite(data.y_test, p_test, chosen_threshold)
    segment_columns = model_config["evaluation"]["segment_columns"]
    test_segment_metrics = per_segment_metrics(
        data.test_frame, data.y_test, p_test, segment_columns, chosen_threshold
    )
    test_lift_gains = lift_gains_table(
        data.y_test, p_test, model_config["evaluation"]["lift_gains_bins"]
    )

    def _cv_model_builder() -> Any:
        # Keep the winning class_weight/scale_pos_weight (CV should reflect
        # the actual chosen configuration) but drop early_stopping_rounds --
        # `stratified_cv_metrics` fits each fold with no eval_set, and
        # XGBoost requires one whenever early stopping is configured.
        wrapper_cls = imbalance.MODEL_CLASSES[best_entry["family"]]
        cv_params = {
            k: v
            for k, v in best_entry["params"].items()
            if k != "early_stopping_rounds"
        }
        return wrapper_cls(cv_params)

    cv_metrics = stratified_cv_metrics(
        _cv_model_builder,
        data.X_train,
        data.y_train,
        n_splits=model_config["evaluation"]["cv_folds"],
        seed=seed,
        threshold=chosen_threshold,
    )

    run_id = tracking.log_training_run(
        model=calibrated_model,
        params=best_entry["params"],
        metrics={
            **test_metrics,
            "brier_before_calibration": calib_result["brier_before"],
            "brier_after_calibration": calib_result["brier_after"],
            "calibration_slope_before": calib_result["calibration_slope_before"],
            "calibration_slope_after": calib_result["calibration_slope_after"],
            "chosen_threshold": chosen_threshold,
        },
        dataset_version=dataset_version,
        feature_pipeline_version=_feature_pipeline_version(dataset_version),
        seed=seed,
        model_family=best_entry["family"],
        imbalance_strategy="class_weight",
        X_example=data.X_val.head(50),
        figures={"reliability_diagram": reliability_path},
        extra_tags={"stage": "final_calibrated_model"},
        run_name=f"{best_entry['family']}_calibrated_final",
    )

    return {
        "calibrated_model": calibrated_model,
        "calibration": calib_result,
        "threshold": threshold_result,
        "test_metrics": test_metrics,
        "test_segment_metrics": test_segment_metrics,
        "test_lift_gains": test_lift_gains,
        "cv_metrics": cv_metrics,
        "mlflow_run_id": run_id,
    }


def _print_summary(
    leaderboard: list[dict[str, Any]],
    final: dict[str, Any],
    best_family: str,
) -> None:
    print("\n=== Leaderboard (validation split, sorted by PR-AUC) ===")
    for entry in leaderboard:
        m = entry["metrics"]
        print(
            f"  {entry['family']:<20} pr_auc={m['pr_auc']:.4f} "
            f"roc_auc={m['roc_auc']:.4f} brier={m['brier_score']:.4f} "
            f"cv_score={entry['cv_score']:.4f}"
        )
    print(f"\nSelected: {best_family}")

    t = final["threshold"]
    print(f"\n=== Chosen threshold: {t['chosen_threshold']:.4f} ===")
    print(f"Rationale: {t['chosen_rationale']}")
    print(
        f"  min-cost threshold:    {t['min_cost']['threshold']:.4f} "
        f"(expected cost={t['min_cost']['expected_cost']:.2f})"
    )
    print(
        f"  max-F1 threshold:      {t['max_f1']['threshold']:.4f} "
        f"(F1={t['max_f1']['f1']:.4f})"
    )
    print(
        f"  max-Youden's-J thresh: {t['max_youden_j']['threshold']:.4f} "
        f"(J={t['max_youden_j']['youden_j']:.4f})"
    )

    print("\n=== Test-set metrics (chosen threshold, calibrated model) ===")
    for key, value in final["test_metrics"].items():
        print(f"  {key}: {value:.4f}")


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint: search, evaluate, calibrate, threshold-optimise and
    (optionally) register the best CreditGuard model.
    """
    parser = argparse.ArgumentParser(
        description="Train CreditGuard's credit-risk model."
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--models", default="logistic,random_forest,xgboost")
    parser.add_argument("--imbalance-grid", action="store_true")
    parser.add_argument("--register-best", action="store_true")
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--features-config", default="config/features.yaml")
    parser.add_argument("--cleaning-config", default="config/validation_rules.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="reports/models")
    parser.add_argument("--figures-dir", default="reports/figures/models")
    args = parser.parse_args(argv)

    model_config = load_model_config(args.model_config)
    features_config = load_features_config(args.features_config)
    cleaning_config = load_rule_config(args.cleaning_config)
    data_dir = Path(args.data_dir) if args.data_dir else None
    seed = model_config["random_seed"]

    model_families = [
        MODEL_FAMILY_ALIASES[name.strip()]
        for name in args.models.split(",")
        if name.strip()
    ]

    data = load_training_data(
        args.dataset_version, features_config, cleaning_config, data_dir
    )

    leaderboard = build_leaderboard(model_families, model_config, data, seed)
    for entry in leaderboard:
        log_leaderboard_entry(entry, args.dataset_version, seed, data)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.imbalance_grid:
        run_imbalance_grid(
            leaderboard,
            model_config,
            data,
            args.dataset_version,
            seed,
            output_dir / "imbalance_comparison.md",
        )

    scoring = model_config["search"]["scoring"]
    best_entry = select_best_model(leaderboard, scoring)

    final = finalise_best_model(
        best_entry,
        model_config,
        data,
        args.dataset_version,
        seed,
        Path(args.figures_dir),
    )

    final["test_lift_gains"].to_csv(output_dir / "test_lift_gains.csv", index=False)
    for column, segment_df in final["test_segment_metrics"].items():
        segment_df.to_csv(
            output_dir / f"test_segment_metrics_{column}.csv", index=False
        )
    final["cv_metrics"].to_csv(output_dir / "train_cv_metrics.csv")

    settings = get_settings()
    model_path = (
        settings.model_dir
        / f"calibrated_model_{args.dataset_version}_{best_entry['family']}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final["calibrated_model"], model_path)

    if args.register_best:
        registry.register_model(
            algorithm=best_entry["family"],
            training_date=datetime.now(UTC),
            dataset_version=args.dataset_version,
            feature_list=data.feature_names,
            hyperparameters=best_entry["params"],
            metrics={
                **final["test_metrics"],
                "chosen_threshold": final["threshold"]["chosen_threshold"],
            },
            mlflow_run_id=final["mlflow_run_id"],
            artifact_path=str(model_path),
        )

    _print_summary(leaderboard, final, best_entry["family"])


if __name__ == "__main__":
    main()
