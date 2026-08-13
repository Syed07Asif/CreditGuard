"""Tests for creditguard.monitoring.alerts."""

from __future__ import annotations

import json

import pytest

from creditguard.monitoring import alerts


def _alert(**overrides) -> alerts.Alert:
    defaults = dict(
        severity="WARNING",
        category="test",
        message="something happened",
        triggering_values={"x": 1},
        recommended_action="do something",
    )
    defaults.update(overrides)
    return alerts.Alert(**defaults)


def test_log_file_alert_sink_appends_json_lines(tmp_path) -> None:
    path = tmp_path / "alerts.log"
    sink = alerts.LogFileAlertSink(path)
    sink.send(_alert(message="first"))
    sink.send(_alert(message="second"))

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "first"
    assert json.loads(lines[1])["message"] == "second"


class _FailingSink(alerts.AlertSink):
    def send(self, alert: alerts.Alert) -> None:
        raise RuntimeError("sink is down")


def test_dispatcher_isolates_a_failing_sink(tmp_path) -> None:
    path = tmp_path / "alerts.log"
    dispatcher = alerts.AlertDispatcher([_FailingSink(), alerts.LogFileAlertSink(path)])
    # Must not raise even though the first sink always fails.
    dispatcher.dispatch(_alert())
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8").strip())["category"] == "test"


def test_alerts_for_performance_run_builds_one_alert_per_degradation() -> None:
    summary = {
        "model_id": "m1",
        "degraded_metrics": [
            {
                "metric": "roc_auc",
                "training_value": 0.8,
                "current_value": 0.6,
                "tolerance": 0.1,
            },
            {
                "metric": "brier_score",
                "training_value": 0.1,
                "current_value": 0.2,
                "tolerance": 0.1,
            },
        ],
    }
    result = alerts.alerts_for_performance_run(summary)
    assert len(result) == 2
    assert all(a.category == "performance" for a in result)
    assert all(a.model_id == "m1" for a in result)


def test_alerts_for_performance_run_empty_when_nothing_degraded() -> None:
    assert (
        alerts.alerts_for_performance_run({"model_id": "m1", "degraded_metrics": []})
        == []
    )


def test_alerts_for_data_quality_run_ok_status_produces_no_alert() -> None:
    summary = {
        "status": "OK",
        "violation_rate": 0.01,
        "window_start": "2024-01-01",
        "window_end": "2024-01-31",
        "n_rows": 100,
        "n_issues_persisted": 1,
    }
    assert alerts.alerts_for_data_quality_run(summary) == []


@pytest.mark.parametrize(
    "status,expected_severity", [("WARNING", "WARNING"), ("ALERT", "CRITICAL")]
)
def test_alerts_for_data_quality_run_maps_status_to_severity(
    status, expected_severity
) -> None:
    summary = {
        "status": status,
        "violation_rate": 0.2,
        "window_start": "2024-01-01",
        "window_end": "2024-01-31",
        "n_rows": 100,
        "n_issues_persisted": 20,
    }
    result = alerts.alerts_for_data_quality_run(summary)
    assert len(result) == 1
    assert result[0].severity == expected_severity


def test_alert_for_model_load_failure() -> None:
    alert = alerts.alert_for_model_load_failure(ValueError("boom"), model_id="m1")
    assert alert.severity == "CRITICAL"
    assert alert.category == "model_load"
    assert alert.model_id == "m1"
    assert "boom" in alert.message
