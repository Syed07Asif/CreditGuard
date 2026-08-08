"""CLI for the Phase 3 validation and cleaning pipeline.

    python -m creditguard.validation.cli validate --dataset-version ds_...
    python -m creditguard.validation.cli clean --dataset-version ds_... \
        --output-version ds_..._clean

`validate` only observes a raw dataset version and reports; `clean` repairs it
into a new versioned dataset. Both read the raw parquet tables written by
`creditguard.data.generator` (via `dataset_version`), not the post-ingest
database -- most injected Phase 2 errors violate a database CHECK/PK/NOT NULL
constraint and are quarantined out of the database at ingest time already, so
this phase must catch them upstream, on the raw generated data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from creditguard.config import get_settings
from creditguard.data.versioning import (
    GeneratedDataset,
    read_dataset_tables,
    write_dataset,
)
from creditguard.db.repository import DataQualityIssueRepository
from creditguard.validation.cleaning import DataCleaner, save_cleaner
from creditguard.validation.engine import (
    ValidationResult,
    build_registry,
    load_rule_config,
    run,
)
from creditguard.validation.report import write_report
from creditguard.validation.rules import KEY_COLUMNS

ISSUE_INSERT_BATCH_SIZE = 5000
DEFAULT_CONFIG_PATH = "config/validation_rules.yaml"


def _persist_issues(result: ValidationResult) -> int:
    """Write every ERROR/WARNING violation to data_quality_issues, batched."""
    records = result.to_records()
    repo = DataQualityIssueRepository()
    for start in range(0, len(records), ISSUE_INSERT_BATCH_SIZE):
        repo.insert_many(records[start : start + ISSUE_INSERT_BATCH_SIZE])
    return len(records)


def _print_validation_summary(result: ValidationResult) -> None:
    print(
        f"Validation result for {result.dataset_version}: "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    print(f"{'rule_name':<45} {'table':<20} {'severity':<8} {'count':>8}")
    print("-" * 84)
    for _, row in result.rule_counts.iterrows():
        print(
            f"{row['rule_name']:<45} {row['table']:<20} {row['severity']:<8} "
            f"{row['count']:>8}"
        )
    if result.rule_counts.empty:
        print("(no violations found)")
    print()
    for table, count in result.row_counts.items():
        quarantined = len(result.quarantined_keys.get(table, set()))
        print(f"  {table}: {count} rows, {quarantined} ERROR-flagged (quarantined)")


def validate_command(args: argparse.Namespace) -> ValidationResult:
    """Run the validation engine over a dataset version and persist/report results."""
    settings = get_settings()
    data_root = Path(args.data_dir) if args.data_dir else settings.data_dir / "raw"
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir
        else settings.reports_dir / "data_quality"
    )

    tables = read_dataset_tables(args.dataset_version, data_root)
    config = load_rule_config(args.config)
    registry = build_registry(config)
    result = run(tables, registry, dataset_version=args.dataset_version)

    n_persisted = _persist_issues(result)
    md_path, html_path = write_report(
        args.dataset_version, result, before=tables, after=None, reports_dir=reports_dir
    )

    print(f"Wrote report: {md_path}")
    print(f"Wrote report: {html_path}")
    print(f"Persisted {n_persisted} issues to data_quality_issues")
    print()
    _print_validation_summary(result)
    return result


def clean_command(args: argparse.Namespace) -> tuple[ValidationResult, dict]:
    """Validate, then clean a dataset version into a new versioned clean dataset."""
    settings = get_settings()
    data_root = Path(args.data_dir) if args.data_dir else settings.data_dir / "raw"
    output_root = (
        Path(args.output_dir) if args.output_dir else settings.data_dir / "processed"
    )
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir
        else settings.reports_dir / "data_quality"
    )

    tables = read_dataset_tables(args.dataset_version, data_root)
    config = load_rule_config(args.config)
    registry = build_registry(config)

    result = run(tables, registry, dataset_version=args.dataset_version)
    n_persisted = _persist_issues(result)

    cleaner = DataCleaner(config)
    cleaned_tables = cleaner.fit_transform(tables)
    report = cleaner.last_report_

    for table, quarantined_df in report.quarantined_rows.items():
        if quarantined_df.empty:
            continue
        extra_records = [
            {
                "table_name": table,
                "record_key": "|".join(
                    str(quarantined_df.loc[i, c]) for c in _key_columns_for(table)
                ),
                "rule_name": "cleaning:structural_quarantine",
                "severity": "ERROR",
                "detail": (
                    "row dropped from modelling set after clip/dedup/impute repair"
                    " failed"
                ),
            }
            for i in quarantined_df.index
        ]
        repo = DataQualityIssueRepository()
        for start in range(0, len(extra_records), ISSUE_INSERT_BATCH_SIZE):
            repo.insert_many(extra_records[start : start + ISSUE_INSERT_BATCH_SIZE])

    metadata = {
        "source_dataset_version": args.dataset_version,
        "output_version": args.output_version,
        "row_counts_before": {name: len(df) for name, df in tables.items()},
        "row_counts_after": {name: len(df) for name, df in cleaned_tables.items()},
        "dedup_log": report.dedup_log,
        "clip_log": [vars(e) for e in report.clip_log],
        "winsorize_log": [vars(e) for e in report.winsorize_log],
        "imputed_log": report.imputed_log,
        "quarantined_counts": {t: len(df) for t, df in report.quarantined_rows.items()},
    }
    dataset = GeneratedDataset(
        customers=cleaned_tables["customers"],
        loan_applications=cleaned_tables["loan_applications"],
        financial_profiles=cleaned_tables["financial_profiles"],
        credit_history=cleaned_tables["credit_history"],
        loan_outcomes=cleaned_tables["loan_outcomes"],
        metadata=metadata,
        injection_manifest={},
    )
    out_dir = write_dataset(dataset, args.output_version, output_root)

    cleaner_path = (
        settings.model_dir / "validation" / f"cleaner_{args.output_version}.joblib"
    )
    cleaner_path.parent.mkdir(parents=True, exist_ok=True)
    save_cleaner(cleaner, str(cleaner_path))

    md_path, html_path = write_report(
        args.output_version,
        result,
        before=tables,
        after=cleaned_tables,
        reports_dir=reports_dir,
    )

    print(f"Wrote report: {md_path}")
    print(f"Wrote report: {html_path}")
    print(f"Persisted {n_persisted} pre-cleaning issues to data_quality_issues")
    print(f"Wrote cleaned dataset {args.output_version} to {out_dir}")
    print(f"Saved fitted cleaner to {cleaner_path}")
    print()
    _print_validation_summary(result)
    print()
    print("Cleaning summary:")
    for table in tables:
        before_n = len(tables[table])
        after_n = len(cleaned_tables[table])
        print(f"  {table}: {before_n} -> {after_n} rows ({before_n - after_n} removed)")

    return result, metadata


def _key_columns_for(table: str) -> list[str]:
    return KEY_COLUMNS.get(table, [])


def build_parser() -> argparse.ArgumentParser:
    """Build the `creditguard.validation.cli` argument parser."""
    parser = argparse.ArgumentParser(
        description="CreditGuard data validation and cleaning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a dataset version."
    )
    validate_parser.add_argument("--dataset-version", required=True)
    validate_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    validate_parser.add_argument("--data-dir", default=None)
    validate_parser.add_argument("--reports-dir", default=None)
    validate_parser.set_defaults(func=validate_command)

    clean_parser = subparsers.add_parser("clean", help="Clean a dataset version.")
    clean_parser.add_argument("--dataset-version", required=True)
    clean_parser.add_argument("--output-version", required=True)
    clean_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    clean_parser.add_argument("--data-dir", default=None)
    clean_parser.add_argument("--output-dir", default=None)
    clean_parser.add_argument("--reports-dir", default=None)
    clean_parser.set_defaults(func=clean_command)

    return parser


def main(argv: list[str] | None = None) -> Any:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
