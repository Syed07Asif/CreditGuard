"""Pluggable alerting: a console sink, a log-file sink, and an optional
webhook sink (e.g. a Slack incoming webhook), all driven by the same
`Alert` shape -- severity, the triggering values, and a recommended action,
so any sink can render an alert without knowing which check produced it.

Triggers (per the Phase 10 brief): any feature at DRIFT status, prediction
PSI above threshold, a performance metric degraded beyond tolerance, a data
quality violation rate above threshold, or the active model failing to
load. The `alerts_for_*` builders below turn each check module's result
into zero or more `Alert`s; `dispatch_all` fans them out to every
configured sink, logging (not raising) if an individual sink fails, since a
broken webhook must never take down the monitoring run that's trying to
report a real problem.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from creditguard.config import get_settings
from creditguard.monitoring.baseline import load_monitoring_config

logger = logging.getLogger("creditguard.monitoring.alerts")

Severity = Literal["INFO", "WARNING", "CRITICAL"]


@dataclass(frozen=True)
class Alert:
    severity: Severity
    category: str
    message: str
    triggering_values: dict[str, Any]
    recommended_action: str
    model_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlertSink(ABC):
    """One alert destination. `send` must not raise for a transient
    delivery failure -- callers use `dispatch_all`, which already isolates
    each sink's own exceptions, but a sink implementation should still
    avoid raising for expected failure modes (e.g. an unreachable webhook).
    """

    @abstractmethod
    def send(self, alert: Alert) -> None: ...


class ConsoleAlertSink(AlertSink):
    def send(self, alert: Alert) -> None:
        print(f"[{alert.severity}] {alert.category}: {alert.message}")


class LogFileAlertSink(AlertSink):
    """Appends one JSON line per alert to a log file -- simple, greppable,
    and durable across process restarts (unlike the API's in-memory
    `MetricsStore`).
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def send(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert.to_dict()) + "\n")


class WebhookAlertSink(AlertSink):
    """POSTs the alert as JSON to a configured webhook URL (e.g. a Slack
    incoming webhook). Optional -- only constructed when `ALERT_WEBHOOK_URL`
    is set (see `creditguard.config.Settings`).
    """

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def send(self, alert: Alert) -> None:
        import requests

        requests.post(self.url, json=alert.to_dict(), timeout=self.timeout)


class AlertDispatcher:
    """Fans one alert out to every configured sink, isolating each sink's
    own failures so a broken webhook never blocks the console/log-file
    sinks (or the caller's own monitoring run).
    """

    def __init__(self, sinks: list[AlertSink]) -> None:
        self.sinks = sinks

    def dispatch(self, alert: Alert) -> None:
        for sink in self.sinks:
            try:
                sink.send(alert)
            except Exception:
                logger.error(
                    "alert_sink_failed",
                    extra={"sink": type(sink).__name__, "category": alert.category},
                    exc_info=True,
                )

    def dispatch_all(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            self.dispatch(alert)


def build_default_dispatcher(
    monitoring_config_path: str = "config/monitoring.yaml",
) -> AlertDispatcher:
    """Console + log-file sinks always; a webhook sink too if
    `ALERT_WEBHOOK_URL` is configured.
    """
    config = load_monitoring_config(monitoring_config_path)["alerts"]
    settings = get_settings()
    sinks: list[AlertSink] = []
    if config.get("console", True):
        sinks.append(ConsoleAlertSink())
    sinks.append(LogFileAlertSink(Path(config["log_path"])))
    if settings.alert_webhook_url:
        sinks.append(WebhookAlertSink(settings.alert_webhook_url))
    return AlertDispatcher(sinks)


# -- alert builders: one per check module's result shape ---------------------


def alerts_for_drift_run(result: Any) -> list[Alert]:
    """`result` is a `creditguard.monitoring.drift.DriftRunResult`. Typed as
    `Any` to avoid a hard import-time dependency on `drift.py` (which
    itself imports `data_quality.py`) -- keeps this module import-order
    agnostic for callers that only need one specific alert builder.
    """
    from creditguard.monitoring.drift import (
        PREDICTION_PROBABILITY_FEATURE_NAME,
        STATUS_DRIFT,
        STATUS_WARNING,
    )

    alerts: list[Alert] = []
    for finding in result.findings:
        if finding.status != STATUS_DRIFT:
            continue
        if finding.feature_name == PREDICTION_PROBABILITY_FEATURE_NAME:
            alerts.append(
                Alert(
                    severity="CRITICAL",
                    category="prediction_drift",
                    message=(
                        f"Predicted-probability distribution has drifted "
                        f"(PSI={finding.current_stat:.4f}) -- the earliest "
                        "warning signal that the scored population has shifted."
                    ),
                    triggering_values={
                        "psi": finding.current_stat,
                        "method": finding.method,
                    },
                    recommended_action=(
                        "Review recent applicant traffic for a population shift; "
                        "if sustained, evaluate retraining "
                        "(creditguard.monitoring.retraining.should_retrain)."
                    ),
                    model_id=result.model_id,
                )
            )
        else:
            alerts.append(
                Alert(
                    severity="WARNING",
                    category="drift",
                    message=(
                        f"Feature '{finding.feature_name}' is at DRIFT status "
                        f"({finding.method}={finding.current_stat:.4f})."
                    ),
                    triggering_values={
                        "feature": finding.feature_name,
                        "method": finding.method,
                        "statistic": finding.current_stat,
                        "detail": finding.detail,
                    },
                    recommended_action=(
                        "Inspect the feature's distribution in "
                        f"{result.report_md_path or 'the drift report'}; "
                        "if several features are drifting together, treat it "
                        "as a population shift rather than a single bad batch."
                    ),
                    model_id=result.model_id,
                )
            )

    if result.concept_drift.status == STATUS_WARNING:
        alerts.append(
            Alert(
                severity="WARNING",
                category="concept_drift",
                message=(
                    f"Rolling default rate ({result.concept_drift.current_rate:.4f}) "
                    "has moved outside the confidence interval implied by the "
                    f"baseline rate ({result.concept_drift.baseline_rate:.4f})."
                ),
                triggering_values={
                    "current_rate": result.concept_drift.current_rate,
                    "baseline_rate": result.concept_drift.baseline_rate,
                    "ci_low": result.concept_drift.ci_low,
                    "ci_high": result.concept_drift.ci_high,
                    "n": result.concept_drift.n,
                },
                recommended_action=(
                    "Confirm with a larger matured-loan window before acting; "
                    "if the shift persists, treat it as evidence for retraining."
                ),
                model_id=result.model_id,
            )
        )
    return alerts


def alerts_for_performance_run(summary: dict[str, Any]) -> list[Alert]:
    """`summary` is `creditguard.monitoring.performance.run_performance_check`'s
    return dict.
    """
    return [
        Alert(
            severity="WARNING",
            category="performance",
            message=(
                f"Metric '{degradation['metric']}' degraded beyond tolerance: "
                f"training={degradation['training_value']:.4f}, "
                f"current={degradation['current_value']:.4f} "
                f"(tolerance={degradation['tolerance']:.0%})."
            ),
            triggering_values=degradation,
            recommended_action=(
                "Confirm with more matured loans if the sample is small; if "
                "degradation persists, evaluate retraining "
                "(creditguard.monitoring.retraining.should_retrain)."
            ),
            model_id=summary["model_id"],
        )
        for degradation in summary["degraded_metrics"]
    ]


def alerts_for_data_quality_run(summary: dict[str, Any]) -> list[Alert]:
    """`summary` is `creditguard.monitoring.data_quality.run_data_quality_check`'s
    return dict.
    """
    if summary["status"] == "OK":
        return []
    severity: Severity = "CRITICAL" if summary["status"] == "ALERT" else "WARNING"
    return [
        Alert(
            severity=severity,
            category="data_quality",
            message=(
                f"Production data-quality violation rate "
                f"({summary['violation_rate']:.2%}) is at {summary['status']} "
                f"over {summary['window_start']}..{summary['window_end']}."
            ),
            triggering_values={
                "violation_rate": summary["violation_rate"],
                "n_rows": summary["n_rows"],
                "n_issues_persisted": summary["n_issues_persisted"],
            },
            recommended_action=(
                "Review the top offending rules in data_quality_issues; a "
                "sustained rise often points at an upstream data source "
                "change rather than random noise."
            ),
        )
    ]


def alert_for_model_load_failure(
    error: Exception, model_id: str | None = None
) -> Alert:
    """The active model failed to load -- a `/health/ready` failure that
    monitoring should surface proactively, not wait for someone to notice.
    """
    return Alert(
        severity="CRITICAL",
        category="model_load",
        message=f"Active model failed to load: {error}",
        triggering_values={"error_type": type(error).__name__, "error": str(error)},
        recommended_action=(
            "Check the API's /health/ready endpoint and logs; confirm the "
            "active model_registry row's artifact_path is reachable from "
            "this process, then restart the affected service."
        ),
        model_id=model_id,
    )
