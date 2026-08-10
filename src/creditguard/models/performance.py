"""One-off backfill of ROC/PR/confusion/calibration/lift-gains/feature-
importance data for a registered model, added in Phase 9 so the dashboard's
Model Performance page can render real curves through the read-only `GET
/api/v1/model/performance` API endpoint -- without the dashboard touching
the database or model directly, and without the API recomputing these on
every request (that would make the "thin transport layer" API rebuild a
~97k-row test split and refit a pipeline per page view).

`full_metric_suite`'s scalar metrics (ROC-AUC, PR-AUC, KS, Brier, ...) are
already registered at training time (`creditguard.models.train`) and don't
need backfilling -- only the curve/table data this module adds is new.

Run once for the currently active model (or after training a new one):

    python -m creditguard.models.performance

Does not retrain or touch the model artifact -- CLAUDE.md hard rule 6
("never overwrite a trained model") is about the artifact, not this
metadata enrichment (`creditguard.models.registry.update_metrics`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from creditguard.explain.shap_explainer import (
    ExplainabilityError,
    unwrap_base_estimator,
)
from creditguard.models import registry
from creditguard.models.evaluate import confusion_counts_at_threshold, lift_gains_table
from creditguard.models.train import load_training_data
from creditguard.scoring.engine import (
    DEFAULT_CLEANING_CONFIG_PATH,
    DEFAULT_FEATURES_CONFIG_PATH,
    ScoringEngineError,
)
from creditguard.validation.engine import load_rule_config

LIFT_GAINS_BINS = 10
CALIBRATION_BINS = 10


def _feature_importance(
    calibrated_model: Any, feature_names: list[str]
) -> list[dict[str, Any]]:
    """Global feature importance, ranked descending by magnitude.

    Uses `coef_` (linear models -- the registered CreditGuard model is
    logistic regression) if present, falling back to `feature_importances_`
    (tree ensembles) so this stays correct if a future promotion swaps the
    winning algorithm.
    """
    try:
        base_estimator = unwrap_base_estimator(calibrated_model)
    except ExplainabilityError:
        return []

    if hasattr(base_estimator, "coef_"):
        raw = np.asarray(base_estimator.coef_).ravel()
    elif hasattr(base_estimator, "feature_importances_"):
        raw = np.asarray(base_estimator.feature_importances_).ravel()
    else:
        return []

    pairs = sorted(
        zip(feature_names, raw.tolist(), strict=True),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [{"feature": name, "importance": value} for name, value in pairs]


def compute_performance_artifacts(
    model_row: dict[str, Any],
    *,
    features_config_path: str | Path = DEFAULT_FEATURES_CONFIG_PATH,
    cleaning_config_path: str | Path = DEFAULT_CLEANING_CONFIG_PATH,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Rebuild `model_row`'s own test split (the same temporal split its
    training run used, via `creditguard.models.train.load_training_data`)
    and compute every curve/table Phase 9's Model Performance page needs.
    """
    calibrated_model = joblib.load(model_row["artifact_path"])
    chosen_threshold = float(model_row["metrics"]["chosen_threshold"])

    with open(features_config_path, encoding="utf-8") as f:
        features_config = yaml.safe_load(f)
    cleaning_config = load_rule_config(cleaning_config_path)

    data = load_training_data(
        model_row["dataset_version"], features_config, cleaning_config, data_dir
    )
    p_test = calibrated_model.predict_proba(data.X_test)[:, 1]

    fpr, tpr, _ = roc_curve(data.y_test, p_test)
    precision, recall, _ = precision_recall_curve(data.y_test, p_test)
    fraction_positive, mean_predicted = calibration_curve(
        data.y_test, p_test, n_bins=CALIBRATION_BINS, strategy="quantile"
    )
    gains = lift_gains_table(data.y_test, p_test, n_bins=LIFT_GAINS_BINS)

    return {
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve": {"precision": precision.tolist(), "recall": recall.tolist()},
        "confusion_matrix": confusion_counts_at_threshold(
            data.y_test, p_test, chosen_threshold
        ),
        "calibration_curve": {
            "mean_predicted": mean_predicted.tolist(),
            "fraction_positive": fraction_positive.tolist(),
        },
        "lift_gains": gains.to_dict("records"),
        "feature_importance": _feature_importance(calibrated_model, data.feature_names),
    }


def backfill(model_id: str | None = None) -> str:
    """Compute and persist performance artifacts for `model_id` (the active
    model if omitted). Returns the model_id backfilled.
    """
    model_row = registry.get_active_model() if model_id is None else None
    if model_row is None and model_id is not None:
        from creditguard.db.repository import ModelRegistryRepository

        model_row = ModelRegistryRepository().get_by_id(model_id)
    if model_row is None:
        raise ScoringEngineError(
            "No active model registered -- run `python -m creditguard.models.train "
            "--register-best` first."
        )

    artifacts = compute_performance_artifacts(model_row)
    registry.update_metrics(model_row["model_id"], {"performance": artifacts})
    return str(model_row["model_id"])


def main() -> None:
    """Backfill performance artifacts for the active model and report success."""
    model_id = backfill()
    print(f"Backfilled performance artifacts for model_id={model_id}")


if __name__ == "__main__":
    main()
