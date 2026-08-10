"""Tests for dashboard chart/formatting helpers, plus a smoke test that
every page module imports and runs (via Streamlit's own `AppTest` harness,
not a real server) without raising. HTTP is mocked with `responses` --
nothing here talks to a real API or model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import pandas as pd
import pytest
import responses
from streamlit.testing.v1 import AppTest

from creditguard.dashboard.components import charts, tables

DASHBOARD_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "creditguard" / "dashboard"
)
BASE_URL = "http://testserver"

# -- formatting helpers -------------------------------------------------------


def test_format_currency() -> None:
    assert tables.format_currency(1234567.8) == "₹1,234,568"
    assert tables.format_currency(0) == "₹0"
    assert tables.format_currency(None) == "--"


def test_format_percentage() -> None:
    assert tables.format_percentage(0.0583) == "5.8%"
    assert tables.format_percentage(1.0) == "100.0%"
    assert tables.format_percentage(None) == "--"


def test_format_ratio() -> None:
    assert tables.format_ratio(0.23456) == "0.23"
    assert tables.format_ratio(None) == "--"


def test_score_band_color_known_and_unknown() -> None:
    assert tables.score_band_color("VERY_LOW") == "#1a7a4c"
    assert tables.score_band_color("VERY_HIGH") == "#c1352c"
    assert tables.score_band_color("NOT_A_BAND") == "#666666"
    assert tables.score_band_color(None) == "#666666"


# -- chart functions: valid frame + empty frame, neither raises -------------


def test_score_distribution_histogram_valid_and_empty() -> None:
    fig = charts.score_distribution_histogram(pd.Series([600, 650, 700, 750, 800]))
    assert isinstance(fig, matplotlib.figure.Figure)
    empty_fig = charts.score_distribution_histogram(pd.Series([], dtype=float))
    assert isinstance(empty_fig, matplotlib.figure.Figure)


def test_risk_category_bar_valid_and_empty() -> None:
    order = ("VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH")
    counts = pd.Series({"LOW": 5, "HIGH": 2})
    assert isinstance(charts.risk_category_bar(counts, order), matplotlib.figure.Figure)
    assert isinstance(
        charts.risk_category_bar(pd.Series(dtype=int), order), matplotlib.figure.Figure
    )


def test_applications_over_time_valid_and_empty() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "n": [5, 8, 3],
            "default_rate": [0.1, 0.2, 0.05],
        }
    )
    assert isinstance(charts.applications_over_time(daily), matplotlib.figure.Figure)
    assert isinstance(
        charts.applications_over_time(pd.DataFrame()), matplotlib.figure.Figure
    )


def test_rate_by_segment_bar_valid_and_empty() -> None:
    rates = pd.Series({"PERSONAL": 0.1, "AUTO": 0.2})
    fig = charts.rate_by_segment_bar(rates, "Rate", "Title")
    assert isinstance(fig, matplotlib.figure.Figure)
    empty_fig = charts.rate_by_segment_bar(pd.Series(dtype=float), "Rate", "Title")
    assert isinstance(empty_fig, matplotlib.figure.Figure)


def test_score_gauge_returns_figure() -> None:
    assert isinstance(charts.score_gauge(720, "LOW"), matplotlib.figure.Figure)
    assert isinstance(charts.score_gauge(300, "VERY_HIGH"), matplotlib.figure.Figure)
    assert isinstance(charts.score_gauge(900, "VERY_LOW"), matplotlib.figure.Figure)


def test_shap_contribution_chart_valid_and_empty() -> None:
    risk_factors = [{"feature": "dti", "impact": 0.3, "description": "..."}]
    positive_factors = [
        {"feature": "savings_to_income", "impact": -0.2, "description": "..."}
    ]
    fig = charts.shap_contribution_chart(risk_factors, positive_factors)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(charts.shap_contribution_chart([], []), matplotlib.figure.Figure)


def test_roc_pr_confusion_calibration_lift_feature_charts_valid_and_empty() -> None:
    assert isinstance(
        charts.roc_curve_chart([0, 0.5, 1], [0, 0.8, 1], 0.87), matplotlib.figure.Figure
    )
    assert isinstance(charts.roc_curve_chart([], []), matplotlib.figure.Figure)

    assert isinstance(
        charts.pr_curve_chart([1, 0.5, 0.1], [0, 0.5, 1], 0.54),
        matplotlib.figure.Figure,
    )
    assert isinstance(charts.pr_curve_chart([], []), matplotlib.figure.Figure)

    assert isinstance(
        charts.confusion_matrix_heatmap({"tn": 10, "fp": 2, "fn": 1, "tp": 5}),
        matplotlib.figure.Figure,
    )
    assert isinstance(charts.confusion_matrix_heatmap({}), matplotlib.figure.Figure)

    assert isinstance(
        charts.calibration_chart([0.1, 0.5, 0.9], [0.12, 0.48, 0.91]),
        matplotlib.figure.Figure,
    )
    assert isinstance(charts.calibration_chart([], []), matplotlib.figure.Figure)

    lift_rows = [
        {
            "decile": 1,
            "n": 10,
            "n_positive": 5,
            "default_rate": 0.5,
            "cum_n": 10,
            "cum_positive": 5,
            "cum_gain": 0.5,
            "lift": 4.5,
            "cum_lift": 4.5,
        }
    ]
    assert isinstance(charts.lift_gains_chart(lift_rows), matplotlib.figure.Figure)
    assert isinstance(charts.lift_gains_chart([]), matplotlib.figure.Figure)

    importance_rows = [
        {"feature": "dti", "importance": 0.8},
        {"feature": "gender_MALE", "importance": 0.01},
    ]
    assert isinstance(
        charts.feature_importance_chart(importance_rows), matplotlib.figure.Figure
    )
    assert isinstance(charts.feature_importance_chart([]), matplotlib.figure.Figure)


# -- page smoke tests ---------------------------------------------------------

_MODEL_INFO = {
    "model_id": "logistic_regression_test",
    "model_version": "1.0.0",
    "algorithm": "logistic_regression",
    "training_date": "2026-01-01T00:00:00Z",
    "dataset_version": "ds_test",
    "feature_count": 77,
    "metrics": {
        "roc_auc": 0.877,
        "pr_auc": 0.545,
        "ks_statistic": 0.592,
        "brier_score": 0.068,
        "precision": 0.288,
        "recall": 0.833,
        "f1": 0.428,
        "calibration_slope": 1.0,
    },
    "chosen_threshold": 0.084,
    "is_active": True,
}
_MODEL_PERFORMANCE = {
    "model_id": "logistic_regression_test",
    "model_version": "1.0.0",
    "roc_curve": {"fpr": [0, 0.5, 1], "tpr": [0, 0.8, 1]},
    "pr_curve": {"precision": [1, 0.5, 0.1], "recall": [0, 0.5, 1]},
    "confusion_matrix": {"tn": 100, "fp": 20, "fn": 10, "tp": 30},
    "calibration_curve": {
        "mean_predicted": [0.1, 0.5, 0.9],
        "fraction_positive": [0.12, 0.48, 0.91],
    },
    "lift_gains": [
        {
            "decile": 1,
            "n": 10,
            "n_positive": 5,
            "default_rate": 0.5,
            "cum_n": 10,
            "cum_positive": 5,
            "cum_gain": 0.5,
            "lift": 4.5,
            "cum_lift": 4.5,
        }
    ],
    "feature_importance": [{"feature": "dti", "importance": 0.8}],
}
_MODEL_VERSIONS = {"versions": [{**_MODEL_INFO, "metrics": _MODEL_INFO["metrics"]}]}
_PREDICTIONS_PAGE = {
    "items": [
        {
            "prediction_id": 1,
            "loan_id": "LOAN-1",
            "customer_id": "CUST-1",
            "model_id": "logistic_regression_test",
            "default_probability": 0.05,
            "credit_score": 750,
            "risk_category": "LOW",
            "recommendation": "APPROVE",
            "latency_ms": 40,
            "request_source": "api",
            "loan_type": "PERSONAL",
            "age": 35,
            "annual_income": 600000.0,
            "employment_type": "SALARIED",
            "created_at": "2026-01-01T00:00:00Z",
        }
    ],
    "total": 1,
    "page": 1,
    "page_size": 100,
}


def _mock_all_endpoints() -> None:
    responses.add(
        responses.GET, f"{BASE_URL}/health", json={"status": "ok"}, status=200
    )
    responses.add(
        responses.GET, f"{BASE_URL}/api/v1/model/info", json=_MODEL_INFO, status=200
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/performance",
        json=_MODEL_PERFORMANCE,
        status=200,
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/model/versions",
        json=_MODEL_VERSIONS,
        status=200,
    )
    responses.add(
        responses.GET,
        f"{BASE_URL}/api/v1/predictions",
        json=_PREDICTIONS_PAGE,
        status=200,
    )


PAGE_FILES = [
    DASHBOARD_DIR / "app.py",
    DASHBOARD_DIR / "pages" / "1_Applicant_Scoring.py",
    DASHBOARD_DIR / "pages" / "2_Portfolio_Analytics.py",
    DASHBOARD_DIR / "pages" / "3_Model_Performance.py",
    DASHBOARD_DIR / "pages" / "4_Monitoring.py",
]


@pytest.fixture(autouse=True)
def _api_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_BASE_URL", BASE_URL)
    monkeypatch.setenv("API_KEY", "test-key")
    # st.cache_data's cache is process-global (persists across tests within
    # the same pytest session), so a successful call cached by one test
    # would silently make a later test's "API unreachable" scenario look
    # reachable. Clear it before every test in this module.
    import streamlit as st

    st.cache_data.clear()


@pytest.mark.parametrize("page_file", PAGE_FILES, ids=lambda p: p.name)
@responses.activate
def test_page_imports_and_runs_without_exception(page_file: Path) -> None:
    """Every page module must import and execute top-to-bottom (Streamlit's
    own execution model for a page script) without an unhandled exception,
    with a fully-available (mocked) API behind it.
    """
    _mock_all_endpoints()
    app_test = AppTest.from_file(str(page_file))
    app_test.run(timeout=30)
    assert not app_test.exception


@responses.activate
def test_portfolio_analytics_shows_friendly_error_when_api_unreachable() -> None:
    """Acceptance criterion 4: stopping the API produces a friendly error
    state, not a traceback, on a page that calls the API at module level.
    """
    # No endpoints registered -- every request is "unreachable".
    app_test = AppTest.from_file(
        str(DASHBOARD_DIR / "pages" / "2_Portfolio_Analytics.py")
    )
    app_test.run(timeout=30)
    assert not app_test.exception
    assert len(app_test.error) > 0
