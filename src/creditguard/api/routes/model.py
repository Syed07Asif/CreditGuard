"""`GET /model/info`, `GET /model/versions`.

Reads `model_registry` directly rather than going through
`creditguard.scoring.engine`'s singleton -- these endpoints don't need the
full feature pipeline or SHAP explainer warm, so they stay fast (and
answerable) even before the first `/predict` call has triggered that load.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from creditguard.api.dependencies import enforce_rate_limit
from creditguard.api.schemas import (
    ModelInfoResponse,
    ModelPerformanceResponse,
    ModelVersionsResponse,
    ModelVersionSummary,
)
from creditguard.db.repository import ModelRegistryRepository
from creditguard.models import registry
from creditguard.scoring.engine import ScoringEngineError

router = APIRouter(tags=["model"], dependencies=[Depends(enforce_rate_limit)])


def _scalar_metrics(raw_metrics: dict) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in raw_metrics.items()
        if isinstance(value, (int, float))
    }


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """The active model's id, version, algorithm, training date, feature
    count, key metrics and decision threshold.
    """
    model_row = registry.get_active_model()
    if model_row is None:
        raise ScoringEngineError(
            "No active model registered -- run `python -m creditguard.models.train "
            "--register-best` first."
        )
    metrics = _scalar_metrics(model_row["metrics"])
    return ModelInfoResponse(
        model_id=model_row["model_id"],
        model_version=model_row["model_version"],
        algorithm=model_row["algorithm"],
        training_date=model_row["training_date"],
        dataset_version=model_row["dataset_version"],
        feature_count=len(model_row["feature_list"]),
        metrics=metrics,
        chosen_threshold=float(model_row["metrics"]["chosen_threshold"]),
        is_active=bool(model_row["is_active"]),
    )


@router.get("/model/versions", response_model=ModelVersionsResponse)
def model_versions() -> ModelVersionsResponse:
    """Every registered model version, newest training date first."""
    rows = ModelRegistryRepository().fetch_dataframe().to_dict("records")
    versions = [
        ModelVersionSummary(
            model_id=row["model_id"],
            model_version=row["model_version"],
            algorithm=row["algorithm"],
            training_date=row["training_date"],
            is_active=bool(row["is_active"]),
            metrics=_scalar_metrics(row["metrics"]),
        )
        for row in rows
    ]
    versions.sort(key=lambda version: version.training_date, reverse=True)
    return ModelVersionsResponse(versions=versions)


@router.get("/model/performance", response_model=ModelPerformanceResponse)
def model_performance() -> ModelPerformanceResponse:
    """ROC/PR/confusion/calibration/lift-gains/feature-importance data for
    the active model -- read back from `model_registry.metrics.performance`,
    backfilled once by `python -m creditguard.models.performance` (see that
    module's docstring). Never recomputed on request: that would make this
    "thin transport layer" endpoint refit a pipeline over the full test
    split on every dashboard page view.
    """
    model_row = registry.get_active_model()
    if model_row is None:
        raise ScoringEngineError(
            "No active model registered -- run `python -m creditguard.models.train "
            "--register-best` first."
        )
    performance = model_row["metrics"].get("performance")
    if performance is None:
        raise ScoringEngineError(
            "Performance artifacts not yet computed for the active model -- run "
            "`python -m creditguard.models.performance` first."
        )
    return ModelPerformanceResponse(
        model_id=model_row["model_id"],
        model_version=model_row["model_version"],
        **performance,
    )
