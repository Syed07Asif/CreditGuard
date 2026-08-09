"""MLflow tracking: every run this phase creates -- one per model family's
search result, one per (family, imbalance strategy) comparison cell, and one
for the final calibrated model -- is logged to the `'creditguard'`
experiment with enough metadata to reproduce it later.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from creditguard.config import get_settings

EXPERIMENT_NAME = "creditguard"


def get_git_commit_sha() -> str:
    """The current git commit SHA, or `'unknown'` if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def configure_tracking() -> str:
    """Point mlflow at the configured tracking URI and ensure the
    `'creditguard'` experiment exists. Returns its `experiment_id`.
    """
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(EXPERIMENT_NAME)


def log_training_run(
    *,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    dataset_version: str,
    feature_pipeline_version: str,
    seed: int,
    model_family: str,
    imbalance_strategy: str,
    X_example: Any,
    figures: dict[str, str | Path] | None = None,
    extra_tags: dict[str, str] | None = None,
    run_name: str | None = None,
) -> str:
    """Log one full run: params, every numeric metric, the dataset version,
    feature pipeline version, git commit SHA, seed, every figure, and the
    model artifact (with signature, input example, and mlflow's
    auto-generated `requirements.txt`). Tags the run with `model_family` and
    `imbalance_strategy`. Returns the mlflow `run_id`.
    """
    experiment_id = configure_tracking()
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        tags = {"model_family": model_family, "imbalance_strategy": imbalance_strategy}
        tags.update(extra_tags or {})
        mlflow.set_tags(tags)

        mlflow.log_params(params)
        mlflow.log_param("dataset_version", dataset_version)
        mlflow.log_param("feature_pipeline_version", feature_pipeline_version)
        mlflow.log_param("git_commit_sha", get_git_commit_sha())
        mlflow.log_param("seed", seed)

        numeric_metrics = {
            key: float(value)
            for key, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        mlflow.log_metrics(numeric_metrics)

        for figure_path in (figures or {}).values():
            mlflow.log_artifact(str(figure_path), artifact_path="figures")

        signature = infer_signature(X_example, model.predict_proba(X_example))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_example.head(5),
        )
        return run.info.run_id
