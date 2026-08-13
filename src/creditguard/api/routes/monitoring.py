"""`GET /monitoring/drift`, `GET /monitoring/performance`,
`GET /monitoring/data-quality`.

Every endpoint here reads back rows already written by
`creditguard.monitoring.scheduler`'s periodic cycle (or the pipeline
orchestrator's `--monitor` stage) -- never recomputes a drift/performance/
data-quality check on request, matching every other "thin transport layer"
read endpoint in this API (see `model.py`'s `/model/performance` docstring
for the same reasoning: recomputing on every dashboard page view would
refit/rescan real data per request).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from creditguard.api.dependencies import enforce_rate_limit
from creditguard.api.schemas import (
    DataQualityTrendItem,
    DriftFindingItem,
    MonitoringDataQualityResponse,
    MonitoringDriftResponse,
    MonitoringMetricItem,
    MonitoringPerformanceResponse,
)
from creditguard.db.repository import DriftReportRepository, MonitoringMetricRepository
from creditguard.models import registry
from creditguard.monitoring.data_quality import rule_violation_trend
from creditguard.scoring.engine import ScoringEngineError

router = APIRouter(tags=["monitoring"], dependencies=[Depends(enforce_rate_limit)])


def _resolve_model_id(model_id: str | None) -> str:
    if model_id:
        return model_id
    model_row = registry.get_active_model()
    if model_row is None:
        raise ScoringEngineError(
            "No active model registered -- run `python -m creditguard.models.train "
            "--register-best` first."
        )
    return str(model_row["model_id"])


@router.get("/monitoring/drift", response_model=MonitoringDriftResponse)
def monitoring_drift(
    model_id: str | None = Query(default=None),
) -> MonitoringDriftResponse:
    """The most recent drift run's findings for a model (the active model
    by default).
    """
    resolved_model_id = _resolve_model_id(model_id)
    frame = DriftReportRepository().fetch_dataframe(
        filters={"model_id": resolved_model_id}
    )
    if frame.empty:
        return MonitoringDriftResponse(
            model_id=resolved_model_id,
            findings=[],
            n_ok=0,
            n_warning=0,
            n_drift=0,
            latest_run_at=None,
        )

    latest_run_at = frame["created_at"].max()
    latest = frame.loc[frame["created_at"] == latest_run_at]
    findings = [
        DriftFindingItem(
            feature_name=row["feature_name"],
            method=row["method"],
            baseline_stat=float(row["baseline_stat"]),
            current_stat=float(row["current_stat"]),
            drift_score=float(row["drift_score"]),
            status=row["status"],
            created_at=row["created_at"],
        )
        for _, row in latest.iterrows()
    ]
    status_counts = latest["status"].value_counts()
    return MonitoringDriftResponse(
        model_id=resolved_model_id,
        findings=findings,
        n_ok=int(status_counts.get("OK", 0)),
        n_warning=int(status_counts.get("WARNING", 0)),
        n_drift=int(status_counts.get("DRIFT", 0)),
        latest_run_at=latest_run_at,
    )


@router.get("/monitoring/performance", response_model=MonitoringPerformanceResponse)
def monitoring_performance(
    model_id: str | None = Query(default=None),
) -> MonitoringPerformanceResponse:
    """Every persisted `monitoring_metrics` row for a model, newest first --
    both predictive (ROC-AUC, PR-AUC, ...) and operational (volume,
    approve/review/reject mix, latency) metrics, undifferentiated by this
    endpoint; the dashboard groups them by `metric_name` for display.
    """
    resolved_model_id = _resolve_model_id(model_id)
    frame = MonitoringMetricRepository().fetch_dataframe(
        filters={"model_id": resolved_model_id}
    )
    if frame.empty:
        return MonitoringPerformanceResponse(model_id=resolved_model_id, metrics=[])

    frame = frame.sort_values("created_at", ascending=False)
    metrics = [
        MonitoringMetricItem(
            metric_name=row["metric_name"],
            metric_value=float(row["metric_value"]),
            window_start=row["window_start"],
            window_end=row["window_end"],
            created_at=row["created_at"],
        )
        for _, row in frame.iterrows()
    ]
    return MonitoringPerformanceResponse(model_id=resolved_model_id, metrics=metrics)


@router.get("/monitoring/data-quality", response_model=MonitoringDataQualityResponse)
def monitoring_data_quality(
    window_days: int = Query(default=90, ge=1, le=365),
) -> MonitoringDataQualityResponse:
    """Validation rule violation counts per day over the trailing
    `window_days`, from `data_quality_issues`.
    """
    trend_frame = rule_violation_trend(window_days)
    if trend_frame.empty:
        return MonitoringDataQualityResponse(trend=[], total_issues=0)

    trend = [
        DataQualityTrendItem(
            rule_name=row["rule_name"],
            severity=row["severity"],
            day=row["day"],
            n=int(row["n"]),
        )
        for _, row in trend_frame.iterrows()
    ]
    return MonitoringDataQualityResponse(
        trend=trend, total_issues=int(trend_frame["n"].sum())
    )
