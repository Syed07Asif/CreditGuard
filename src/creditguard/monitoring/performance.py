"""Production performance monitoring: for loans whose 12-month outcome is
now observable ("matured" loans -- a `predictions` row whose `loan_id` has
a matching `loan_outcomes` row), recompute the same metric suite Phase 6
uses at training time (`creditguard.models.evaluate.full_metric_suite`, not
a reimplementation of it) and compare against that model's training-time
values.

Also tracks operational metrics that don't need a label at all: prediction
volume, the approve/review/reject mix, average score/probability, the
high-risk share, and API p95 latency (read from `predictions.latency_ms`,
persisted per request -- not the API process's own in-memory
`MetricsStore`, which resets on restart and isn't visible to this
out-of-process monitoring job).

Every metric -- predictive and operational -- becomes one `monitoring_metrics`
row, so `GET /api/v1/monitoring/performance` and Phase 9's Monitoring page
can chart them over time without recomputing anything.
"""

from __future__ import annotations

import argparse
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from creditguard.db.engine import get_engine
from creditguard.db.models import LoanOutcome, Prediction
from creditguard.db.repository import MonitoringMetricRepository
from creditguard.models import registry
from creditguard.models.evaluate import full_metric_suite
from creditguard.monitoring.baseline import load_monitoring_config

# Direction each predictive metric is "better" in, for degradation checks.
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {"roc_auc", "pr_auc", "ks_statistic", "precision", "recall"}
)
_LOWER_IS_BETTER: frozenset[str] = frozenset({"brier_score"})
_CALIBRATION_METRIC = "calibration_slope"

HIGH_RISK_CATEGORIES = frozenset({"HIGH", "VERY_HIGH"})


def fetch_matured_predictions(
    model_id: str, window_start: date, window_end: date
) -> pd.DataFrame:
    """`predictions` rows for `model_id` whose loan has a matured
    (outcome-observed) `loan_outcomes` row within the window -- joined on
    `loan_id`, since a `predictions` row from a stateless `/predict` call
    (no `loan_id` persisted upstream) simply won't match anything and is
    correctly excluded.
    """
    with get_engine().connect() as conn:
        return pd.read_sql(
            select(
                Prediction.loan_id,
                Prediction.default_probability,
                Prediction.credit_score,
                Prediction.risk_category,
                Prediction.recommendation,
                Prediction.latency_ms,
                Prediction.created_at,
                LoanOutcome.default_12m,
                LoanOutcome.outcome_observed_date,
            )
            .join(LoanOutcome, LoanOutcome.loan_id == Prediction.loan_id)
            .where(
                Prediction.model_id == model_id,
                LoanOutcome.outcome_observed_date >= window_start,
                LoanOutcome.outcome_observed_date <= window_end,
            ),
            conn,
        )


def fetch_operational_predictions(
    model_id: str, window_start: datetime, window_end: datetime
) -> pd.DataFrame:
    """Every `predictions` row for `model_id` in the window, matured or
    not -- operational metrics (volume, decision mix, latency) don't need a
    label.
    """
    with get_engine().connect() as conn:
        return pd.read_sql(
            select(Prediction).where(
                Prediction.model_id == model_id,
                Prediction.created_at >= window_start,
                Prediction.created_at <= window_end,
            ),
            conn,
        )


def compute_predictive_metrics(
    matured: pd.DataFrame, min_matured_loans: int
) -> dict[str, float] | None:
    """`full_metric_suite` on the matured-loan window, or `None` if there
    aren't enough matured loans to trust the result (per
    `config/monitoring.yaml`'s `performance.min_matured_loans`).
    """
    if len(matured) < min_matured_loans:
        return None
    return full_metric_suite(matured["default_12m"], matured["default_probability"])


def compute_operational_metrics(operational: pd.DataFrame) -> dict[str, float]:
    """Volume/decision-mix/score/latency metrics that don't need a label."""
    if operational.empty:
        return {
            "prediction_volume": 0.0,
            "approve_rate": float("nan"),
            "review_rate": float("nan"),
            "reject_rate": float("nan"),
            "avg_credit_score": float("nan"),
            "avg_default_probability": float("nan"),
            "high_risk_share": float("nan"),
            "latency_p95_ms": float("nan"),
        }
    n = len(operational)
    recommendation_counts = operational["recommendation"].value_counts()
    return {
        "prediction_volume": float(n),
        "approve_rate": float(recommendation_counts.get("APPROVE", 0)) / n,
        "review_rate": float(recommendation_counts.get("REVIEW", 0)) / n,
        "reject_rate": float(recommendation_counts.get("REJECT", 0)) / n,
        "avg_credit_score": float(operational["credit_score"].mean()),
        "avg_default_probability": float(operational["default_probability"].mean()),
        "high_risk_share": float(
            operational["risk_category"].isin(HIGH_RISK_CATEGORIES).mean()
        ),
        "latency_p95_ms": float(np.percentile(operational["latency_ms"], 95)),
    }


def is_degraded(
    metric_name: str, training_value: float, current_value: float, tolerance: float
) -> bool:
    """Whether `current_value` has degraded from `training_value` by more
    than `tolerance` (a relative fraction, default 10%).

    `calibration_slope`'s target is 1.0, not "as high/low as possible", so a
    relative-to-training-value comparison doesn't apply the same way (and
    would be hypersensitive whenever training happened to be very close to
    1.0 already -- a relative tolerance around a near-zero base is nearly
    always "degraded"). Instead, degradation is measured as the *distance
    from 1.0* growing by more than `tolerance` in absolute terms: e.g. with
    the default 10% tolerance, a training slope of 1.0 can drift to
    anywhere in roughly [0.90, 1.10] before this counts as degraded.
    """
    if training_value is None or current_value is None:
        return False
    if math.isnan(training_value) or math.isnan(current_value):
        return False
    if metric_name == _CALIBRATION_METRIC:
        training_distance = abs(training_value - 1.0)
        current_distance = abs(current_value - 1.0)
        return current_distance > training_distance + tolerance
    if metric_name in _HIGHER_IS_BETTER:
        return current_value < training_value * (1 - tolerance)
    if metric_name in _LOWER_IS_BETTER:
        return current_value > training_value * (1 + tolerance)
    return False


def persist_metrics(
    model_id: str, metrics: dict[str, float], window_start: date, window_end: date
) -> int:
    records = [
        {
            "model_id": model_id,
            "metric_name": name,
            "metric_value": value,
            "window_start": window_start,
            "window_end": window_end,
        }
        for name, value in metrics.items()
        if not math.isnan(value)
    ]
    return MonitoringMetricRepository().insert_many(records)


def run_performance_check(
    *,
    model_id: str | None = None,
    window_days: int | None = None,
    monitoring_config_path: str = "config/monitoring.yaml",
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Recompute predictive + operational metrics for a model's trailing
    monitoring window, persist them all to `monitoring_metrics`, and report
    which predictive metrics (if any) have degraded beyond tolerance versus
    that model's own training-time values.
    """
    perf_config = load_monitoring_config(monitoring_config_path)["performance"]
    as_of = as_of or datetime.now(UTC)
    window_days = window_days if window_days is not None else perf_config["window_days"]
    window_end = as_of.date()
    window_start = window_end - timedelta(days=window_days)

    if model_id is None:
        model_row = registry.get_active_model()
        if model_row is None:
            raise RuntimeError("No active model registered -- nothing to monitor.")
        model_id = model_row["model_id"]
    else:
        from creditguard.db.repository import ModelRegistryRepository

        model_row = ModelRegistryRepository().get_by_id(model_id)
        if model_row is None:
            raise RuntimeError(f"No model_registry row for model_id={model_id!r}")

    matured = fetch_matured_predictions(model_id, window_start, window_end)
    predictive_metrics = compute_predictive_metrics(
        matured, perf_config["min_matured_loans"]
    )

    window_start_dt = datetime.combine(window_start, datetime.min.time(), tzinfo=UTC)
    window_end_dt = datetime.combine(window_end, datetime.max.time(), tzinfo=UTC)
    operational = fetch_operational_predictions(
        model_id, window_start_dt, window_end_dt
    )
    operational_metrics = compute_operational_metrics(operational)

    all_metrics = {**operational_metrics}
    degraded: list[dict[str, Any]] = []
    if predictive_metrics is not None:
        all_metrics.update(
            {k: v for k, v in predictive_metrics.items() if k in perf_config["metrics"]}
        )
        training_metrics = model_row["metrics"]
        tolerance = perf_config["degradation_tolerance"]
        for name in perf_config["metrics"]:
            if name not in predictive_metrics or name not in training_metrics:
                continue
            current_value = predictive_metrics[name]
            training_value = float(training_metrics[name])
            if is_degraded(name, training_value, current_value, tolerance):
                degraded.append(
                    {
                        "metric": name,
                        "training_value": training_value,
                        "current_value": current_value,
                        "tolerance": tolerance,
                    }
                )

    n_persisted = persist_metrics(model_id, all_metrics, window_start, window_end)

    return {
        "model_id": model_id,
        "window_start": window_start,
        "window_end": window_end,
        "n_matured_loans": len(matured),
        "predictive_metrics": predictive_metrics,
        "operational_metrics": operational_metrics,
        "degraded_metrics": degraded,
        "n_metrics_persisted": n_persisted,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: run a performance-monitoring check for a model (the active
    model if `--model-id` is omitted).
    """
    parser = argparse.ArgumentParser(
        description="CreditGuard Phase 10 performance monitoring check."
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--monitoring-config", default="config/monitoring.yaml")
    args = parser.parse_args(argv)

    summary = run_performance_check(
        model_id=args.model_id,
        window_days=args.window_days,
        monitoring_config_path=args.monitoring_config,
    )
    print(
        f"Performance check for {summary['model_id']}: "
        f"{summary['n_matured_loans']} matured loans, "
        f"{len(summary['degraded_metrics'])} metric(s) degraded beyond tolerance"
    )
    for degradation in summary["degraded_metrics"]:
        print(f"  DEGRADED: {degradation}")


if __name__ == "__main__":
    main()
