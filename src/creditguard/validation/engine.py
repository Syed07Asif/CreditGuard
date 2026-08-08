"""Validation engine: builds a rule registry from config and runs it over a dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from creditguard.validation.rules import (
    KEY_COLUMNS,
    CreditUtilizationRule,
    DuplicateRecordRule,
    EmploymentPlausibilityRule,
    ExpenseConsistencyRule,
    ImpossibleAgeRule,
    IncomeConsistencyRule,
    InvalidDateRule,
    MissingValueRule,
    NegativeFinancialRule,
    NumericRangeRule,
    OrphanRecordRule,
    Rule,
    RuleRegistry,
    Severity,
    TemporalLeakageRule,
    record_key,
)

VIOLATION_RECORD_COLUMNS = ["table", "record_key", "rule_name", "severity", "detail"]


def load_rule_config(path: str | Path) -> dict[str, Any]:
    """Load config/validation_rules.yaml into a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_registry(config: dict[str, Any]) -> RuleRegistry:
    """Build a RuleRegistry of concrete Rules from the `rules` config section."""
    registry = RuleRegistry()
    rules_cfg = config.get("rules", {})

    for entry in rules_cfg.get("missing_value", []):
        registry.register(
            MissingValueRule(
                table=entry["table"],
                column=entry["column"],
                threshold=entry.get("threshold", 0.0),
                severity=Severity(entry.get("severity", "WARNING")),
            )
        )

    for entry in rules_cfg.get("duplicate_record", []):
        registry.register(
            DuplicateRecordRule(
                table=entry["table"],
                keys=entry.get("keys"),
                exact=entry.get("exact", False),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    for entry in rules_cfg.get("numeric_range", []):
        registry.register(
            NumericRangeRule(
                table=entry["table"],
                column=entry["column"],
                min=entry.get("min"),
                max=entry.get("max"),
                severity=Severity(entry.get("severity", "WARNING")),
            )
        )

    for entry in rules_cfg.get("negative_financial", []):
        registry.register(
            NegativeFinancialRule(
                table=entry["table"],
                columns=entry["columns"],
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    if "impossible_age" in rules_cfg:
        entry = rules_cfg["impossible_age"]
        registry.register(
            ImpossibleAgeRule(
                table=entry.get("table", "customers"),
                column=entry.get("column", "age"),
                min=entry.get("min", 18),
                max=entry.get("max", 100),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    for entry in rules_cfg.get("invalid_date", []):
        registry.register(
            InvalidDateRule(
                table=entry["table"],
                columns=entry["columns"],
                order_pairs=entry.get("order_pairs"),
                allow_future=entry.get("allow_future", False),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    if "credit_utilization" in rules_cfg:
        entry = rules_cfg["credit_utilization"]
        registry.register(
            CreditUtilizationRule(
                table=entry.get("table", "credit_history"),
                min=entry.get("min", 0.0),
                max=entry.get("max", 1.5),
                outstanding_over_limit_margin=entry.get(
                    "outstanding_over_limit_margin", 1.5
                ),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    if "income_consistency" in rules_cfg:
        entry = rules_cfg["income_consistency"]
        registry.register(
            IncomeConsistencyRule(
                table=entry.get("table", "customers"),
                annual_income_column=entry.get("annual_income_column", "annual_income"),
                monthly_income_column=entry.get(
                    "monthly_income_column", "monthly_income"
                ),
                tolerance_fraction=entry.get("tolerance_fraction", 0.1),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    if "expense_consistency" in rules_cfg:
        entry = rules_cfg["expense_consistency"]
        registry.register(
            ExpenseConsistencyRule(
                table=entry.get("table", "financial_profiles"),
                income_multiple=entry.get("income_multiple", 1.5),
                severity=Severity(entry.get("severity", "WARNING")),
            )
        )

    for entry in rules_cfg.get("orphan_record", []):
        registry.register(
            OrphanRecordRule(
                child_table=entry["child_table"],
                parent_table=entry["parent_table"],
                key=entry["key"],
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    if "employment_plausibility" in rules_cfg:
        entry = rules_cfg["employment_plausibility"]
        registry.register(
            EmploymentPlausibilityRule(
                table=entry.get("table", "customers"),
                min_working_age=entry.get("min_working_age", 16),
                severity=Severity(entry.get("severity", "WARNING")),
            )
        )

    for entry in rules_cfg.get("temporal_leakage", []):
        registry.register(
            TemporalLeakageRule(
                snapshot_table=entry["snapshot_table"],
                loan_table=entry.get("loan_table", "loan_applications"),
                severity=Severity(entry.get("severity", "ERROR")),
            )
        )

    return registry


@dataclass
class ValidationResult:
    """The outcome of running a RuleRegistry over a set of dataset tables."""

    dataset_version: str
    violations: pd.DataFrame
    rule_counts: pd.DataFrame
    row_counts: dict[str, int]
    quarantined_keys: dict[str, set[str]]
    dataframes: dict[str, pd.DataFrame] = field(repr=False)

    @property
    def passed(self) -> bool:
        """True if no ERROR-severity violation was found anywhere."""
        if self.violations.empty:
            return True
        return not (self.violations["severity"] == Severity.ERROR.value).any()

    def to_records(self) -> list[dict[str, Any]]:
        """Shape all violations for insertion into the data_quality_issues table."""
        if self.violations.empty:
            return []
        shaped = self.violations[VIOLATION_RECORD_COLUMNS].rename(
            columns={"table": "table_name"}
        )
        return shaped.to_dict("records")

    def split_clean_and_quarantined(
        self, table: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (clean_rows, quarantined_rows) for `table`, using is_quarantined."""
        df = self.dataframes[table]
        mask = df["is_quarantined"]
        clean = df.loc[~mask].drop(columns=["is_quarantined"]).reset_index(drop=True)
        quarantined = (
            df.loc[mask].drop(columns=["is_quarantined"]).reset_index(drop=True)
        )
        return clean, quarantined


def run(
    dataframes: dict[str, pd.DataFrame],
    registry: RuleRegistry,
    dataset_version: str = "unknown",
) -> ValidationResult:
    """Run every rule in `registry` over `dataframes` and build a ValidationResult."""
    violation_frames = []
    for rule in registry.rules:
        found = rule.check(dataframes)
        if found.empty:
            continue
        found = found.copy()
        found["rule_name"] = rule.name
        found["severity"] = rule.severity.value
        violation_frames.append(found)

    violations = (
        pd.concat(violation_frames, ignore_index=True)
        if violation_frames
        else pd.DataFrame(columns=VIOLATION_RECORD_COLUMNS)
    )

    row_counts = {name: len(df) for name, df in dataframes.items()}

    quarantined_keys: dict[str, set[str]] = {name: set() for name in dataframes}
    if not violations.empty:
        errors = violations.loc[violations["severity"] == Severity.ERROR.value]
        for table, group in errors.groupby("table"):
            quarantined_keys.setdefault(table, set()).update(group["record_key"])

    annotated: dict[str, pd.DataFrame] = {}
    for name, df in dataframes.items():
        df = df.copy()
        if name in KEY_COLUMNS:
            keys = record_key(name, df)
            df["is_quarantined"] = keys.isin(quarantined_keys.get(name, set()))
        else:
            df["is_quarantined"] = False
        annotated[name] = df

    if violations.empty:
        rule_counts = pd.DataFrame(columns=["rule_name", "severity", "table", "count"])
    else:
        rule_counts = (
            violations.groupby(["rule_name", "severity", "table"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )

    return ValidationResult(
        dataset_version=dataset_version,
        violations=violations,
        rule_counts=rule_counts,
        row_counts=row_counts,
        quarantined_keys=quarantined_keys,
        dataframes=annotated,
    )


__all__ = [
    "Rule",
    "RuleRegistry",
    "ValidationResult",
    "build_registry",
    "load_rule_config",
    "run",
]
