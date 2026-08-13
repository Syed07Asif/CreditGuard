"""Data-quality monitoring: re-run the Phase 3 rule engine
(`creditguard.validation.engine`) over incoming production records on a
schedule, rather than only once at dataset-generation time.

`fetch_production_tables` is the shared "what counts as incoming production
records" query -- it reads `customers`/`loan_applications`/
`financial_profiles`/`credit_history`/`loan_outcomes` directly from the
live database for a date window, in the same `dict[str, DataFrame]` shape
`creditguard.validation.engine.run` and `creditguard.features.leakage.
point_in_time_join` already expect. `monitoring/drift.py` reuses it too, so
both modules agree on exactly what "recent production data" means.
"""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import select

from creditguard.db.engine import get_engine
from creditguard.db.models import (
    CreditHistory,
    Customer,
    FinancialProfile,
    LoanApplication,
    LoanOutcome,
)
from creditguard.db.repository import DataQualityIssueRepository
from creditguard.monitoring.baseline import load_monitoring_config
from creditguard.validation.engine import (
    ValidationResult,
    build_registry,
    load_rule_config,
    run,
)

DEFAULT_VALIDATION_CONFIG_PATH = "config/validation_rules.yaml"
DEFAULT_MONITORING_CONFIG_PATH = "config/monitoring.yaml"

_EMPTY_TABLES = (
    "customers",
    "loan_applications",
    "financial_profiles",
    "credit_history",
    "loan_outcomes",
)


def fetch_production_tables(
    window_start: date, window_end: date
) -> dict[str, pd.DataFrame]:
    """`customers`/`loan_applications`/`financial_profiles`/`credit_history`/
    `loan_outcomes` rows for loans applied within `[window_start,
    window_end]`, read live from the database (not a generated dataset
    version's parquet files -- this is production traffic, mostly written
    by `POST /api/v1/applications`).
    """
    with get_engine().connect() as conn:
        loans = pd.read_sql(
            select(LoanApplication).where(
                LoanApplication.application_date >= window_start,
                LoanApplication.application_date <= window_end,
            ),
            conn,
        )
        if loans.empty:
            return {name: pd.DataFrame() for name in _EMPTY_TABLES}

        customer_ids = loans["customer_id"].unique().tolist()
        loan_ids = loans["loan_id"].tolist()

        customers = pd.read_sql(
            select(Customer).where(Customer.customer_id.in_(customer_ids)), conn
        )
        financial_profiles = pd.read_sql(
            select(FinancialProfile).where(
                FinancialProfile.customer_id.in_(customer_ids)
            ),
            conn,
        )
        credit_history = pd.read_sql(
            select(CreditHistory).where(CreditHistory.customer_id.in_(customer_ids)),
            conn,
        )
        loan_outcomes = pd.read_sql(
            select(LoanOutcome).where(LoanOutcome.loan_id.in_(loan_ids)), conn
        )

    return {
        "customers": customers,
        "loan_applications": loans,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def persist_issues(result: ValidationResult, batch_size: int = 5000) -> int:
    """Write every ERROR/WARNING violation to `data_quality_issues`,
    batched -- the same pattern `creditguard.validation.cli` already uses.
    """
    records = result.to_records()
    repo = DataQualityIssueRepository()
    for start in range(0, len(records), batch_size):
        repo.insert_many(records[start : start + batch_size])
    return len(records)


def violation_rate(result: ValidationResult) -> float:
    """Share of rows across every table that were flagged by at least one
    rule (ERROR or WARNING) -- the trend-worthy summary number, since
    per-rule counts alone don't say what fraction of *traffic* was affected.
    """
    total_rows = sum(result.row_counts.values())
    if total_rows == 0:
        return 0.0
    flagged_keys: set[str] = set()
    for keys in result.quarantined_keys.values():
        flagged_keys.update(keys)
    if not result.violations.empty:
        flagged_keys.update(result.violations["record_key"].unique().tolist())
    return len(flagged_keys) / total_rows


def rule_violation_trend(window_days: int = 90) -> pd.DataFrame:
    """Violation counts per rule per day over the trailing `window_days`,
    from `data_quality_issues` -- what Phase 9's Monitoring page's "data
    quality trend" panel reads.
    """
    since = datetime.now(UTC) - timedelta(days=window_days)
    frame = DataQualityIssueRepository().fetch_dataframe(
        sql=(
            "SELECT rule_name, severity, detected_at::date AS day, count(*) AS n "
            "FROM data_quality_issues "
            f"WHERE detected_at >= '{since.date().isoformat()}' "
            "GROUP BY rule_name, severity, detected_at::date "
            "ORDER BY day"
        )
    )
    return frame


def run_data_quality_check(
    *,
    window_days: int | None = None,
    validation_config_path: str = DEFAULT_VALIDATION_CONFIG_PATH,
    monitoring_config_path: str = DEFAULT_MONITORING_CONFIG_PATH,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Re-run the Phase 3 rule engine over the trailing `window_days` of
    production records, persist every violation, and return a summary
    (including the violation rate against `config/monitoring.yaml`'s
    warning/alert thresholds).
    """
    dq_config = load_monitoring_config(monitoring_config_path)["data_quality"]
    as_of = as_of or datetime.now(UTC)
    window_days = window_days if window_days is not None else dq_config["window_days"]
    window_end = as_of.date()
    window_start = window_end - timedelta(days=window_days)

    tables = fetch_production_tables(window_start, window_end)
    rule_config = load_rule_config(validation_config_path)
    registry = build_registry(rule_config)
    result = run(
        tables, registry, dataset_version=f"production_{window_start}_{window_end}"
    )

    n_persisted = persist_issues(result)
    rate = violation_rate(result)
    status = "OK"
    if rate >= dq_config["violation_rate_alert"]:
        status = "ALERT"
    elif rate >= dq_config["violation_rate_warning"]:
        status = "WARNING"

    return {
        "window_start": window_start,
        "window_end": window_end,
        "n_rows": sum(result.row_counts.values()),
        "n_issues_persisted": n_persisted,
        "violation_rate": rate,
        "status": status,
        "rule_counts": result.rule_counts,
        "passed": result.passed,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI: re-run Phase 3 validation over recent production records."""
    parser = argparse.ArgumentParser(
        description="CreditGuard Phase 10 data-quality monitoring check."
    )
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--validation-config", default=DEFAULT_VALIDATION_CONFIG_PATH)
    parser.add_argument("--monitoring-config", default=DEFAULT_MONITORING_CONFIG_PATH)
    args = parser.parse_args(argv)

    summary = run_data_quality_check(
        window_days=args.window_days,
        validation_config_path=args.validation_config,
        monitoring_config_path=args.monitoring_config,
    )
    print(
        f"Data quality check {summary['window_start']}..{summary['window_end']}: "
        f"{summary['n_rows']} rows, violation_rate={summary['violation_rate']:.4f} "
        f"-> {summary['status']}"
    )


if __name__ == "__main__":
    main()
