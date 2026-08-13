"""Tests for creditguard.monitoring.data_quality."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from creditguard.monitoring import data_quality
from creditguard.validation.engine import run
from creditguard.validation.rules import NegativeFinancialRule, RuleRegistry, Severity


def test_violation_rate_computes_share_of_flagged_rows() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "annual_income": [1000, -500, 2000, -100],
        }
    )
    registry_ = RuleRegistry()
    registry_.register(
        NegativeFinancialRule(
            table="customers", columns=["annual_income"], severity=Severity.ERROR
        )
    )
    result = run({"customers": customers}, registry_, dataset_version="test")
    assert data_quality.violation_rate(result) == pytest.approx(0.5)


def test_violation_rate_zero_rows_is_zero() -> None:
    result = run({}, RuleRegistry(), dataset_version="test")
    assert data_quality.violation_rate(result) == 0.0


def test_violation_rate_no_violations_is_zero() -> None:
    customers = pd.DataFrame({"customer_id": ["C1"], "annual_income": [1000]})
    registry_ = RuleRegistry()
    registry_.register(
        NegativeFinancialRule(table="customers", columns=["annual_income"])
    )
    result = run({"customers": customers}, registry_, dataset_version="test")
    assert data_quality.violation_rate(result) == 0.0


def test_fetch_production_tables_empty_window_returns_empty_frames() -> None:
    tables = data_quality.fetch_production_tables(date(2020, 1, 1), date(2020, 1, 2))
    assert tables["loan_applications"].empty
    assert set(tables.keys()) == set(data_quality._EMPTY_TABLES)


def test_rule_violation_trend_empty_is_empty_frame() -> None:
    trend = data_quality.rule_violation_trend(window_days=30)
    assert trend.empty
