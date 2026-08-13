"""APScheduler-driven monitoring loop for the `monitoring` container:
drift + performance + data-quality checks on a fixed interval, dispatching
whatever alerts each check produces.

Does not itself call `retraining.trigger_retraining` -- that needs an
already-prepared extended dataset version (the pipeline orchestrator's
`--generate`/`--validate`/`--clean`/`--features` stages), which is out of
scope for a periodic in-process job to decide on its own. `should_retrain`
is still evaluated and logged every cycle, so the recommendation is visible
(via logs/`GET /api/v1/monitoring/*`) even though acting on it stays a
deliberate, separate step.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import UTC, datetime
from typing import Any

from creditguard.config import get_settings
from creditguard.models import registry
from creditguard.monitoring import alerts, data_quality, performance
from creditguard.monitoring import drift as drift_module
from creditguard.monitoring import retraining as retraining_module
from creditguard.monitoring.baseline import BaselineNotFoundError

logger = logging.getLogger("creditguard.monitoring.scheduler")

DEFAULT_MONITORING_CONFIG_PATH = "config/monitoring.yaml"


def run_monitoring_cycle(
    monitoring_config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> dict[str, Any]:
    """One full monitoring pass. A failure in any single check is logged
    and recorded in the summary rather than raised -- one broken check
    (e.g. no baseline built yet for a freshly-promoted model) must not
    silently skip the others.
    """
    dispatcher = alerts.build_default_dispatcher(monitoring_config_path)
    summary: dict[str, Any] = {"started_at": datetime.now(UTC).isoformat()}

    model_row = registry.get_active_model()
    if model_row is None:
        logger.warning("monitoring_cycle_no_active_model")
        summary["error"] = "no active model registered"
        return summary
    model_id = model_row["model_id"]
    summary["model_id"] = model_id

    try:
        drift_result = drift_module.run_drift_check(
            model_id=model_id, monitoring_config_path=monitoring_config_path
        )
        dispatcher.dispatch_all(alerts.alerts_for_drift_run(drift_result))
        summary["drift"] = {
            "n_findings": len(drift_result.findings),
            "any_drift": drift_result.any_drift,
        }
    except BaselineNotFoundError as exc:
        logger.warning("monitoring_cycle_drift_skipped", extra={"reason": str(exc)})
        summary["drift"] = {"skipped": str(exc)}
    except Exception:
        logger.error("monitoring_cycle_drift_failed", exc_info=True)
        summary["drift"] = {"error": "check failed, see logs"}

    try:
        perf_summary = performance.run_performance_check(
            model_id=model_id, monitoring_config_path=monitoring_config_path
        )
        dispatcher.dispatch_all(alerts.alerts_for_performance_run(perf_summary))
        summary["performance"] = {
            "n_matured_loans": perf_summary["n_matured_loans"],
            "n_degraded": len(perf_summary["degraded_metrics"]),
        }
    except Exception:
        logger.error("monitoring_cycle_performance_failed", exc_info=True)
        summary["performance"] = {"error": "check failed, see logs"}

    try:
        dq_summary = data_quality.run_data_quality_check(
            monitoring_config_path=monitoring_config_path
        )
        dispatcher.dispatch_all(alerts.alerts_for_data_quality_run(dq_summary))
        summary["data_quality"] = {
            "violation_rate": dq_summary["violation_rate"],
            "status": dq_summary["status"],
        }
    except Exception:
        logger.error("monitoring_cycle_data_quality_failed", exc_info=True)
        summary["data_quality"] = {"error": "check failed, see logs"}

    try:
        should, reasons = retraining_module.should_retrain(
            model_id=model_id, monitoring_config_path=monitoring_config_path
        )
        summary["should_retrain"] = {"result": should, "reasons": reasons}
        if should:
            logger.info(
                "retraining_recommended",
                extra={"model_id": model_id, "reasons": reasons},
            )
    except Exception:
        logger.error("monitoring_cycle_should_retrain_failed", exc_info=True)
        summary["should_retrain"] = {"error": "check failed, see logs"}

    summary["finished_at"] = datetime.now(UTC).isoformat()
    return summary


def run_forever(
    interval_minutes: int | None = None,
    monitoring_config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
) -> None:
    """Run `run_monitoring_cycle` on a fixed interval, blocking forever --
    the monitoring container's main process.
    """
    from apscheduler.schedulers.background import BackgroundScheduler

    interval = interval_minutes or get_settings().monitoring_interval_minutes
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_monitoring_cycle,
        "interval",
        minutes=interval,
        kwargs={"monitoring_config_path": monitoring_config_path},
        next_run_time=datetime.now(UTC),
    )
    scheduler.start()
    logger.info("scheduler_started", extra={"interval_minutes": interval})

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


def main(argv: list[str] | None = None) -> None:
    """CLI:

    python -m creditguard.monitoring.scheduler --once   # single cycle, for testing/CI
    python -m creditguard.monitoring.scheduler           # run forever on an interval
    """
    parser = argparse.ArgumentParser(
        description="CreditGuard Phase 10 monitoring scheduler."
    )
    parser.add_argument("--interval-minutes", type=int, default=None)
    parser.add_argument(
        "--once", action="store_true", help="Run a single monitoring cycle and exit."
    )
    parser.add_argument("--monitoring-config", default=DEFAULT_MONITORING_CONFIG_PATH)
    args = parser.parse_args(argv)

    if args.once:
        summary = run_monitoring_cycle(args.monitoring_config)
        print(summary)
        return

    run_forever(args.interval_minutes, args.monitoring_config)


if __name__ == "__main__":
    main()
