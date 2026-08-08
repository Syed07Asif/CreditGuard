"""CLI that loads a versioned parquet dataset into PostgreSQL, quarantining bad rows.

Rows that violate database constraints (deliberately injected by the generator, or
otherwise) are caught individually via SAVEPOINTs and logged to data_quality_issues
instead of aborting the whole load.

Each table loads as a sequence of independently-committed batches rather than one
transaction for the whole ~400k-row load. A single all-encompassing transaction was
tried first and is closer to the letter of "single transaction" bulk-load designs,
but at these foreign-key-child row counts it holds tens of thousands of row-level
locks (one per FK-referenced parent row) for the whole load: Postgres's lock
manager degrades badly well before it hits the hard "out of shared memory" ceiling,
which is what made a full run take over an hour without finishing. Each batch is
still atomic (bad rows within it are isolated via SAVEPOINT, not by aborting the
batch), so failures never corrupt data -- the loss is only "the whole dataset" as a
single atomic unit, in exchange for the load actually completing.

CLI: python -m creditguard.data.ingest --dataset-version ds_... [--truncate]
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import insert, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from creditguard.config import get_settings
from creditguard.data.versioning import read_dataset_tables
from creditguard.db.engine import get_session
from creditguard.db.models import (
    Base,
    CreditHistory,
    Customer,
    DataQualityIssue,
    FinancialProfile,
    LoanApplication,
    LoanOutcome,
)

COMMIT_BATCH_SIZE = 2000

# When a bulk executemany insert fails partway through, psycopg3's pipeline-mode
# cleanup logs a secondary "error ignored terminating <Pipeline>" warning so it
# doesn't clobber the real IntegrityError we already handle below. It's expected
# and harmless (see psycopg._pipeline.Pipeline.__exit__), just noisy; quiet it so
# the load summary printed by this module stays readable.
logging.getLogger("psycopg").setLevel(logging.ERROR)

# FK-safe load order: parents before children.
TABLE_LOAD_ORDER: list[tuple[str, type[Base]]] = [
    ("customers", Customer),
    ("loan_applications", LoanApplication),
    ("financial_profiles", FinancialProfile),
    ("credit_history", CreditHistory),
    ("loan_outcomes", LoanOutcome),
]


@dataclass
class TableLoadResult:
    """Per-table load outcome: rows attempted, inserted and quarantined."""

    attempted: int = 0
    inserted: int = 0
    quarantined: int = 0


@dataclass
class IngestSummary:
    """Overall ingest outcome, keyed by table name."""

    dataset_version: str
    tables: dict[str, TableLoadResult] = field(default_factory=dict)


def _record_key(table_name: str, record: dict[str, Any]) -> str:
    """Build a human-readable natural-key string for a data_quality_issues row."""
    if table_name in ("customers",):
        return str(record.get("customer_id"))
    if table_name in ("loan_applications", "loan_outcomes"):
        return str(record.get("loan_id"))
    if table_name in ("financial_profiles", "credit_history"):
        return f"{record.get('customer_id')}|{record.get('as_of_date')}"
    return "unknown"


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of dicts with NaN/NaT replaced by None."""
    sanitized = df.astype(object).where(pd.notnull(df), None)
    return sanitized.to_dict("records")


def _log_quality_issue(
    session: Session, table_name: str, record: dict[str, Any], exc: IntegrityError
) -> None:
    """Write a single failed row's constraint violation to data_quality_issues."""
    orig = getattr(exc, "orig", None)
    diag = getattr(orig, "diag", None)
    constraint_name = getattr(diag, "constraint_name", None) if diag else None
    column_name = getattr(diag, "column_name", None) if diag else None
    message = getattr(diag, "message_primary", None) if diag else None
    if constraint_name:
        rule_name = constraint_name
    elif column_name:
        rule_name = f"not_null_violation:{column_name}"
    else:
        rule_name = "constraint_violation"
    session.add(
        DataQualityIssue(
            table_name=table_name,
            record_key=_record_key(table_name, record),
            rule_name=rule_name,
            severity="ERROR",
            detail=message or str(exc.orig or exc),
        )
    )
    session.flush()


def _insert_batch(
    session: Session,
    model: type[Base],
    table_name: str,
    batch: list[dict[str, Any]],
    result: TableLoadResult,
) -> None:
    """Bulk-insert `batch`; on failure, bisect until the bad rows are isolated.

    Bad rows are injected at random and spread uniformly through the data, so a
    single flat chunk-then-fallback-to-per-row strategy degrades to O(n) DB
    round trips per table (almost every chunk contains a bad row). Recursive
    bisection instead costs roughly O(bad_rows * log n): most of a table is
    clean and gets inserted in a handful of large bulk statements, and only
    genuinely bad rows -- plus O(log n) probe splits per bad row -- pay a
    per-row SAVEPOINT round trip. Each attempt runs inside a SAVEPOINT so a
    failure rolls back only that attempt, not the surrounding transaction.
    """
    if not batch:
        return

    if len(batch) == 1:
        record = batch[0]
        try:
            with session.begin_nested():
                # A single dict (not a list) forces a plain single-row execute
                # rather than executemany, which avoids psycopg3's
                # pipeline-mode error-recovery entirely.
                session.execute(insert(model), record)
            result.inserted += 1
        except IntegrityError as row_exc:
            result.quarantined += 1
            _log_quality_issue(session, table_name, record, row_exc)
        return

    try:
        with session.begin_nested():
            session.execute(insert(model), batch)
        result.inserted += len(batch)
        return
    except IntegrityError:
        pass

    mid = len(batch) // 2
    _insert_batch(session, model, table_name, batch[:mid], result)
    _insert_batch(session, model, table_name, batch[mid:], result)


def _insert_table(
    model: type[Base],
    table_name: str,
    records: list[dict[str, Any]],
    progress: bool = True,
) -> TableLoadResult:
    """Load one table as a sequence of independently-committed batches.

    Each batch of up to COMMIT_BATCH_SIZE rows gets its own transaction (bisected
    internally via SAVEPOINTs on failure), so lock accumulation never grows past
    one batch's worth of rows -- see the module docstring for why.
    """
    result = TableLoadResult(attempted=len(records))
    for start in range(0, len(records), COMMIT_BATCH_SIZE):
        batch = records[start : start + COMMIT_BATCH_SIZE]
        with get_session() as session:
            _insert_batch(session, model, table_name, batch, result)
        if progress:
            done = min(start + COMMIT_BATCH_SIZE, len(records))
            print(
                f"  {table_name}: {done}/{len(records)} attempted "
                f"(inserted={result.inserted} quarantined={result.quarantined})"
            )
    return result


def _truncate_tables() -> None:
    """Truncate the five ingest target tables plus the prior data_quality_issues log."""
    table_names = ", ".join(f'"{name}"' for name, _ in TABLE_LOAD_ORDER)
    with get_session() as session:
        session.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))
        session.execute(text('TRUNCATE "data_quality_issues" RESTART IDENTITY'))


def ingest_dataset(
    dataset_version: str,
    truncate: bool = False,
    data_dir: str | Path | None = None,
    progress: bool = True,
) -> IngestSummary:
    """Load a versioned parquet dataset into PostgreSQL, quarantining bad rows."""
    settings = get_settings()
    data_root = Path(data_dir) if data_dir else settings.data_dir / "raw"
    tables = read_dataset_tables(dataset_version, data_root)

    if truncate:
        _truncate_tables()

    summary = IngestSummary(dataset_version=dataset_version)
    for table_name, model in TABLE_LOAD_ORDER:
        records = _records(tables[table_name])
        summary.tables[table_name] = _insert_table(model, table_name, records, progress)
    return summary


def _print_summary(summary: IngestSummary) -> None:
    print(f"Ingested dataset {summary.dataset_version}")
    for table_name, result in summary.tables.items():
        print(
            f"  {table_name}: attempted={result.attempted} "
            f"inserted={result.inserted} quarantined={result.quarantined}"
        )


def main() -> None:
    """CLI entrypoint: ingest a versioned dataset into PostgreSQL."""
    parser = argparse.ArgumentParser(
        description="Ingest a CreditGuard dataset into PostgreSQL."
    )
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    summary = ingest_dataset(
        args.dataset_version, truncate=args.truncate, data_dir=args.data_dir
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
