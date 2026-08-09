"""Imbalance-handling strategies, compared honestly rather than assumed.

Every resampling strategy is applied via an `imblearn.pipeline.Pipeline`
wrapping the raw estimator, never by resampling `X_train`/`y_train` in place
before any split. `imblearn.pipeline.Pipeline.fit` resamples only the data
it's given -- the current fold's training split when used inside
`cross_validate`/`RandomizedSearchCV` -- and `predict`/`predict_proba` never
resample, so a resampling strategy structurally cannot see validation rows,
not just by convention.

`docs/feature_dictionary.md`-adjacent reasoning: `reports/eda/findings.md`'s
"Decisions for Phase 6" recommends class weighting over resampling for the
*production* model (moderate 11.10% imbalance doesn't need synthetic
oversampling); this module exists to make that comparison concrete and
reproducible, not to override it by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from creditguard.models.base import BaseCreditModel
from creditguard.models.evaluate import full_metric_suite, recall_at_precision
from creditguard.models.logistic import LogisticRegressionModel
from creditguard.models.random_forest import RandomForestModel
from creditguard.models.xgboost_model import XGBoostModel

MODEL_CLASSES: dict[str, type[BaseCreditModel]] = {
    "logistic_regression": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}

# "class_weight" isn't a resampler -- see `class_weight_params` -- so it's
# handled separately from the true resampling strategies below.
RESAMPLING_STRATEGIES: frozenset[str] = frozenset(
    {"smote", "random_undersampling", "smote_tomek"}
)
STRATEGIES: tuple[str, ...] = (
    "none",
    "class_weight",
    "smote",
    "random_undersampling",
    "smote_tomek",
)


def build_resampler(strategy: str, seed: int) -> Any:
    """The imblearn resampler for a resampling strategy, or `None` for
    `'none'`/`'class_weight'` (neither resamples).
    """
    if strategy == "smote":
        return SMOTE(random_state=seed)
    if strategy == "random_undersampling":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote_tomek":
        return SMOTETomek(random_state=seed)
    if strategy in ("none", "class_weight"):
        return None
    raise ValueError(f"Unknown imbalance strategy: {strategy!r}")


def class_weight_params(
    model_family: str, params: dict[str, Any], y_train: Any, strategy: str
) -> dict[str, Any]:
    """`params` with class weighting added if `strategy == 'class_weight'`.

    `class_weight="balanced"` for logistic regression/random forest, and its
    XGBoost-native equivalent `scale_pos_weight = n_negative / n_positive`
    for xgboost (XGBoost's sklearn API has no `class_weight` parameter) --
    same idea, different parameter name per library.
    """
    if strategy != "class_weight":
        return dict(params)
    params = dict(params)
    if model_family == "xgboost":
        y_train = np.asarray(y_train)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        params["scale_pos_weight"] = n_neg / n_pos if n_pos else 1.0
    else:
        params["class_weight"] = "balanced"
    return params


def build_estimator_pipeline(
    model_family: str,
    params: dict[str, Any],
    y_train: Any,
    strategy: str,
    seed: int,
) -> Any:
    """A single sklearn/imblearn-compatible estimator implementing `strategy`:
    the bare classifier for `'none'`/`'class_weight'` (weighting is baked
    into `params`), or an `imblearn.pipeline.Pipeline` wrapping a resampler
    for the resampling strategies. Usable directly inside
    `cross_validate`/`RandomizedSearchCV` -- resampling then only ever
    touches the fold it's fitting on.
    """
    strat_params = class_weight_params(model_family, params, y_train, strategy)
    estimator = MODEL_CLASSES[model_family]().build(strat_params)
    resampler = build_resampler(strategy, seed)
    if resampler is None:
        return estimator
    return ImbPipeline(steps=[("resample", resampler), ("clf", estimator)])


def run_imbalance_experiment(
    model_family: str,
    base_params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    strategies: tuple[str, ...] = STRATEGIES,
    recall_at_precision_target: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Fit `model_family` under every strategy in `strategies` (train split
    only) and evaluate on `X_val`/`y_val`. Returns `{strategy: metrics}`.
    """
    results: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        estimator = build_estimator_pipeline(
            model_family, base_params, y_train, strategy, seed
        )
        estimator.fit(X_train, y_train)
        p_val = estimator.predict_proba(X_val)[:, 1]
        metrics = full_metric_suite(y_val, p_val)
        metrics["recall_at_precision"] = recall_at_precision(
            y_val, p_val, recall_at_precision_target
        )
        results[strategy] = metrics
    return results


def run_all_imbalance_experiments(
    model_family_params: dict[str, dict[str, Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int,
    strategies: tuple[str, ...] = STRATEGIES,
    recall_at_precision_target: float = 0.5,
) -> dict[str, dict[str, dict[str, float]]]:
    """`run_imbalance_experiment` for every `{family: base_params}` entry."""
    return {
        family: run_imbalance_experiment(
            family,
            params,
            X_train,
            y_train,
            X_val,
            y_val,
            seed,
            strategies,
            recall_at_precision_target,
        )
        for family, params in model_family_params.items()
    }


def format_imbalance_comparison_report(
    results: dict[str, dict[str, dict[str, float]]],
    recall_at_precision_target: float = 0.5,
) -> str:
    """Render `run_all_imbalance_experiments`' output as the markdown report
    `reports/models/imbalance_comparison.md` expects.
    """
    lines = ["# Imbalance strategy comparison", ""]
    lines.append(
        "Same model family, same train/validation split, one imbalance "
        "strategy varied at a time. Resampling strategies (SMOTE, random "
        "undersampling, SMOTE+Tomek) are applied to the training fold only, "
        "via an `imblearn.pipeline.Pipeline` -- never to the validation data "
        "these metrics are computed on."
    )
    lines.append("")
    for family, strategy_results in results.items():
        lines.append(f"## {family}")
        lines.append("")
        lines.append(
            "| Strategy | PR-AUC | ROC-AUC | "
            f"Recall@P>={recall_at_precision_target:.2f} | Brier | Calibration slope |"
        )
        lines.append("|---|---|---|---|---|---|")
        for strategy, m in strategy_results.items():
            lines.append(
                f"| {strategy} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | "
                f"{m['recall_at_precision']:.3f} | {m['brier_score']:.4f} | "
                f"{m['calibration_slope']:.3f} |"
            )
        lines.append("")
    lines.append("## Reading this table")
    lines.append("")
    lines.append(
        "Resampling (SMOTE / undersampling / SMOTE+Tomek) typically raises "
        "recall at a fixed precision relative to `none`, because it shifts "
        "the decision boundary toward the minority class -- but it does so "
        "by training on a class distribution that no longer matches "
        "reality, which is exactly what breaks calibration: the resulting "
        "probabilities are shifted toward the resampled (roughly 50/50) "
        "rate rather than the true ~11% default rate, so calibration slope "
        "and Brier score are typically worse for the resampling strategies "
        "than for `none` or `class_weight`, even when PR-AUC/recall improve. "
        "**If a resampling strategy is used in production, its probabilities "
        "must be recalibrated** (see `calibration.py`) before they're "
        "treated as default probabilities anywhere downstream (Phase 7's "
        "score conversion assumes a calibrated probability)."
    )
    lines.append("")
    lines.append(
        "`class_weight` reweights the existing data rather than fabricating "
        "or discarding rows, so it changes the loss function's emphasis "
        "without changing what the model is actually trained on -- it "
        "usually recovers most of resampling's recall gain with much less "
        "calibration damage. Recommendation: prefer `class_weight` "
        '(`class_weight="balanced"` / XGBoost\'s `scale_pos_weight`) over '
        "resampling for the model that actually gets registered, consistent "
        "with `reports/eda/findings.md`'s Phase 6 decision -- this dataset's "
        "~11% default rate is a moderate imbalance, not the kind of extreme "
        "(<1%) imbalance where synthetic oversampling earns back its "
        "calibration cost."
    )
    return "\n".join(lines)


def write_imbalance_comparison_report(
    results: dict[str, dict[str, dict[str, float]]],
    output_path: str | Path,
    recall_at_precision_target: float = 0.5,
) -> None:
    """Write `format_imbalance_comparison_report`'s output to `output_path`."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        format_imbalance_comparison_report(results, recall_at_precision_target),
        encoding="utf-8",
    )
