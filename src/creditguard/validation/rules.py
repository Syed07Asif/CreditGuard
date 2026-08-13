"""Rule-based data-quality primitives.

A `Rule` only *observes* data and reports violations; it never mutates a
DataFrame. Cleaning (repair) lives in `creditguard.validation.cleaning` and is
deliberately a separate module so "what's wrong" and "how we fixed it" never
get tangled together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import pandas as pd

DataFrames = dict[str, pd.DataFrame]

VIOLATION_COLUMNS = ["table", "record_key", "detail"]

# Natural-key columns per table, used to build a human-readable `record_key`
# for a violation and to look up which rows a rule's findings refer to.
# Mirrors creditguard.data.ingest._record_key so keys match data_quality_issues
# rows already written by ingest.
KEY_COLUMNS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "loan_applications": ["loan_id"],
    "financial_profiles": ["customer_id", "as_of_date"],
    "credit_history": ["customer_id", "as_of_date"],
    "loan_outcomes": ["loan_id"],
}


class Severity(StrEnum):
    """Rule violation severity."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class Scope(StrEnum):
    """What a rule evaluates over."""

    ROW = "row"
    COLUMN = "column"
    DATASET = "dataset"


def record_key(table: str, df: pd.DataFrame) -> pd.Series:
    """Build the natural-key string for each row of `df` from `table`."""
    cols = KEY_COLUMNS[table]
    if df.empty:
        # pandas' .agg(func, axis=1) on a zero-row, multi-column frame
        # can't infer the reduction produces one column per row and
        # returns an empty DataFrame instead of a Series -- never hit by
        # Phase 3's own tests (always non-empty generated datasets), but
        # real once Phase 10 started validating genuinely empty windows
        # of live production data.
        return pd.Series(dtype=str)
    return df[cols].astype(str).agg("|".join, axis=1)


def empty_violations() -> pd.DataFrame:
    """An empty, correctly-shaped violations DataFrame."""
    return pd.DataFrame(columns=VIOLATION_COLUMNS)


def make_violations(
    table: str, keys: pd.Series | list[str], details: pd.Series | list[str]
) -> pd.DataFrame:
    """Build a violations DataFrame from parallel keys/details sequences."""
    keys = list(keys)
    if not keys:
        return empty_violations()
    return pd.DataFrame(
        {"table": table, "record_key": keys, "details": list(details)}
    ).rename(columns={"details": "detail"})


class Rule(ABC):
    """A single, named data-quality rule."""

    def __init__(
        self,
        name: str,
        description: str,
        severity: Severity,
        scope: Scope,
    ) -> None:
        self.name = name
        self.description = description
        self.severity = severity
        self.scope = scope

    @abstractmethod
    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        """Return violations as a DataFrame with columns table, record_key, detail."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid only
        return (
            f"{type(self).__name__}(name={self.name!r}, severity={self.severity.value})"
        )


class RuleRegistry:
    """An ordered collection of rules to run together."""

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        """Add a rule to the registry."""
        self._rules.append(rule)

    @property
    def rules(self) -> list[Rule]:
        """All registered rules, in registration order."""
        return list(self._rules)


class MissingValueRule(Rule):
    """Flags NULLs in one column once the missing fraction exceeds `threshold`."""

    def __init__(
        self,
        table: str,
        column: str,
        threshold: float = 0.0,
        severity: Severity = Severity.WARNING,
    ) -> None:
        super().__init__(
            name=f"missing_value:{table}.{column}",
            description=f"{table}.{column} has missing values above {threshold:.2%}",
            severity=severity,
            scope=Scope.COLUMN,
        )
        self.table = table
        self.column = column
        self.threshold = threshold

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty or self.column not in df.columns:
            return empty_violations()
        missing_mask = df[self.column].isna()
        if missing_mask.mean() <= self.threshold:
            return empty_violations()
        bad = df.loc[missing_mask]
        keys = record_key(self.table, bad)
        details = [f"{self.column} is missing" for _ in range(len(bad))]
        return make_violations(self.table, keys, details)


class DuplicateRecordRule(Rule):
    """Flags exact-row duplicates, or duplicates on a natural key."""

    def __init__(
        self,
        table: str,
        keys: list[str] | None = None,
        exact: bool = False,
        severity: Severity = Severity.ERROR,
    ) -> None:
        kind = "exact-row" if exact else "+".join(keys or [])
        super().__init__(
            name=f"duplicate_record:{table}:{kind}",
            description=f"{table} rows duplicated on {kind}",
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.keys = keys
        self.exact = exact

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        subset = None if self.exact else self.keys
        dup_mask = df.duplicated(subset=subset, keep=False)
        if not dup_mask.any():
            return empty_violations()
        bad = df.loc[dup_mask]
        keys = record_key(self.table, bad)
        kind = "exact-row" if self.exact else "+".join(self.keys or [])
        details = [f"duplicate record ({kind})" for _ in range(len(bad))]
        return make_violations(self.table, keys, details)


class NumericRangeRule(Rule):
    """Flags values outside a configured [min, max] plausibility band."""

    def __init__(
        self,
        table: str,
        column: str,
        min: float | None = None,  # noqa: A002
        max: float | None = None,  # noqa: A002
        severity: Severity = Severity.WARNING,
    ) -> None:
        super().__init__(
            name=f"numeric_range:{table}.{column}",
            description=f"{table}.{column} outside [{min}, {max}]",
            severity=severity,
            scope=Scope.COLUMN,
        )
        self.table = table
        self.column = column
        self.min = min
        self.max = max

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty or self.column not in df.columns:
            return empty_violations()
        values = pd.to_numeric(df[self.column], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if self.min is not None:
            mask |= values < self.min
        if self.max is not None:
            mask |= values > self.max
        mask &= values.notna()
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = [
            f"{self.column}={v} outside [{self.min}, {self.max}]"
            for v in values.loc[mask]
        ]
        return make_violations(self.table, keys, details)


class NegativeFinancialRule(Rule):
    """Flags negative values in financial columns that must be >= 0."""

    def __init__(
        self, table: str, columns: list[str], severity: Severity = Severity.ERROR
    ) -> None:
        super().__init__(
            name=f"negative_financial:{table}",
            description=f"{table} columns {columns} must be >= 0",
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.columns = columns

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        frames = []
        for column in self.columns:
            if column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            mask = values < 0
            if not mask.any():
                continue
            bad = df.loc[mask]
            keys = record_key(self.table, bad)
            details = [f"{column}={v} is negative" for v in values.loc[mask]]
            frames.append(make_violations(self.table, keys, details))
        if not frames:
            return empty_violations()
        return pd.concat(frames, ignore_index=True)


class ImpossibleAgeRule(Rule):
    """Flags customer ages outside a plausible human range."""

    def __init__(
        self,
        table: str = "customers",
        column: str = "age",
        min: int = 18,
        max: int = 100,  # noqa: A002
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name="impossible_age",
            description=f"{table}.{column} outside [{min}, {max}]",
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.column = column
        self.min = min
        self.max = max

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        values = pd.to_numeric(df[self.column], errors="coerce")
        mask = (values < self.min) | (values > self.max)
        mask &= values.notna()
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = [
            f"age={v} outside [{self.min}, {self.max}]" for v in values.loc[mask]
        ]
        return make_violations(self.table, keys, details)


class InvalidDateRule(Rule):
    """Flags unparseable dates, future dates, and out-of-order date pairs."""

    def __init__(
        self,
        table: str,
        columns: list[str],
        order_pairs: list[list[str]] | None = None,
        allow_future: bool = False,
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name=f"invalid_date:{table}",
            description=(
                f"{table} date columns {columns} unparseable/future/out-of-order"
            ),
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.columns = columns
        self.order_pairs = order_pairs or []
        self.allow_future = allow_future

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()

        now = pd.Timestamp.now(tz=None).normalize()
        frames = []
        for column in self.columns:
            if column not in df.columns:
                continue
            raw = df[column]
            parsed = pd.to_datetime(raw, errors="coerce")
            unparseable_mask = parsed.isna() & raw.notna()
            if unparseable_mask.any():
                bad = df.loc[unparseable_mask]
                keys = record_key(self.table, bad)
                details = [f"{column} is unparseable" for _ in range(len(bad))]
                frames.append(make_violations(self.table, keys, details))

            if not self.allow_future:
                future_mask = parsed.notna() & (parsed > now)
                if future_mask.any():
                    bad = df.loc[future_mask]
                    keys = record_key(self.table, bad)
                    details = [
                        f"{column}={v.date()} is in the future"
                        for v in parsed.loc[future_mask]
                    ]
                    frames.append(make_violations(self.table, keys, details))

        for before_col, after_col in self.order_pairs:
            if before_col not in df.columns or after_col not in df.columns:
                continue
            before = pd.to_datetime(df[before_col], errors="coerce")
            after = pd.to_datetime(df[after_col], errors="coerce")
            mask = before.notna() & after.notna() & (after < before)
            if mask.any():
                bad = df.loc[mask]
                keys = record_key(self.table, bad)
                details = [
                    f"{after_col}={a.date()} is before {before_col}={b.date()}"
                    for a, b in zip(after.loc[mask], before.loc[mask], strict=True)
                ]
                frames.append(make_violations(self.table, keys, details))

        if not frames:
            return empty_violations()
        return pd.concat(frames, ignore_index=True)


class CreditUtilizationRule(Rule):
    """Flags implausible bureau credit utilization."""

    def __init__(
        self,
        table: str = "credit_history",
        min: float = 0.0,  # noqa: A002
        max: float = 1.5,  # noqa: A002
        outstanding_over_limit_margin: float = 1.5,
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name="credit_utilization",
            description=f"{table}.credit_utilization outside [{min}, {max}] "
            f"or outstanding > limit * {outstanding_over_limit_margin}",
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.min = min
        self.max = max
        self.outstanding_over_limit_margin = outstanding_over_limit_margin

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()

        util = pd.to_numeric(df["credit_utilization"], errors="coerce")
        range_mask = util.notna() & ((util < self.min) | (util > self.max))

        outstanding = pd.to_numeric(df["total_outstanding"], errors="coerce")
        limit = pd.to_numeric(df["total_credit_limit"], errors="coerce")
        margin_mask = (
            outstanding.notna()
            & limit.notna()
            & (outstanding > limit * self.outstanding_over_limit_margin)
        )

        mask = range_mask | margin_mask
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = []
        for i in bad.index:
            parts = []
            if range_mask.loc[i]:
                parts.append(
                    f"credit_utilization={util.loc[i]} outside [{self.min}, {self.max}]"
                )
            if margin_mask.loc[i]:
                parts.append(
                    f"total_outstanding={outstanding.loc[i]} > "
                    f"total_credit_limit={limit.loc[i]} * "
                    f"{self.outstanding_over_limit_margin}"
                )
            details.append("; ".join(parts))
        return make_violations(self.table, keys, details)


class IncomeConsistencyRule(Rule):
    """Flags customers whose annual_income disagrees with monthly_income * 12."""

    def __init__(
        self,
        table: str = "customers",
        annual_income_column: str = "annual_income",
        monthly_income_column: str = "monthly_income",
        tolerance_fraction: float = 0.1,
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name="income_consistency",
            description=(
                f"{table}.{annual_income_column} vs 12 * {monthly_income_column} "
                f"disagree by more than {tolerance_fraction:.0%}"
            ),
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.annual_income_column = annual_income_column
        self.monthly_income_column = monthly_income_column
        self.tolerance_fraction = tolerance_fraction

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        annual = pd.to_numeric(df[self.annual_income_column], errors="coerce")
        monthly = pd.to_numeric(df[self.monthly_income_column], errors="coerce")
        implied_annual = 12 * monthly
        diff = (annual - implied_annual).abs()
        tolerance = self.tolerance_fraction * annual.abs()
        mask = annual.notna() & monthly.notna() & (diff > tolerance)
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = [
            f"{self.annual_income_column}={a} vs 12*{self.monthly_income_column}={m}"
            for a, m in zip(annual.loc[mask], implied_annual.loc[mask], strict=True)
        ]
        return make_violations(self.table, keys, details)


class ExpenseConsistencyRule(Rule):
    """Flags financial profiles where expenses + EMI dwarf income."""

    def __init__(
        self,
        table: str = "financial_profiles",
        income_multiple: float = 1.5,
        severity: Severity = Severity.WARNING,
    ) -> None:
        super().__init__(
            name="expense_consistency",
            description=(
                f"{table}.monthly_expenses + monthly_emi > "
                f"monthly_income * {income_multiple}"
            ),
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.income_multiple = income_multiple

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        expenses = pd.to_numeric(df["monthly_expenses"], errors="coerce")
        emi = pd.to_numeric(df["monthly_emi"], errors="coerce")
        income = pd.to_numeric(df["monthly_income"], errors="coerce")
        mask = (expenses + emi) > (income * self.income_multiple)
        mask &= expenses.notna() & emi.notna() & income.notna()
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = [
            f"monthly_expenses={e} + monthly_emi={m} > "
            f"monthly_income={i} * {self.income_multiple}"
            for e, m, i in zip(
                expenses.loc[mask], emi.loc[mask], income.loc[mask], strict=True
            )
        ]
        return make_violations(self.table, keys, details)


class OrphanRecordRule(Rule):
    """Flags child rows whose foreign key has no matching parent row."""

    def __init__(
        self,
        child_table: str,
        parent_table: str,
        key: str,
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name=f"orphan_record:{child_table}->{parent_table}",
            description=f"{child_table}.{key} has no matching {parent_table}.{key}",
            severity=severity,
            scope=Scope.ROW,
        )
        self.child_table = child_table
        self.parent_table = parent_table
        self.key = key

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        child = dataframes.get(self.child_table)
        parent = dataframes.get(self.parent_table)
        if child is None or child.empty or parent is None:
            return empty_violations()
        valid_keys = set(parent[self.key].astype(str))
        mask = ~child[self.key].astype(str).isin(valid_keys)
        if not mask.any():
            return empty_violations()
        bad = child.loc[mask]
        keys = record_key(self.child_table, bad)
        details = [
            f"{self.key}={v} has no matching {self.parent_table} row"
            for v in bad[self.key]
        ]
        return make_violations(self.child_table, keys, details)


class EmploymentPlausibilityRule(Rule):
    """Flags employment_years implying employment before `min_working_age`."""

    def __init__(
        self,
        table: str = "customers",
        min_working_age: int = 16,
        severity: Severity = Severity.WARNING,
    ) -> None:
        super().__init__(
            name="employment_plausibility",
            description=f"{table}.employment_years > age - {min_working_age}",
            severity=severity,
            scope=Scope.ROW,
        )
        self.table = table
        self.min_working_age = min_working_age

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        df = dataframes.get(self.table)
        if df is None or df.empty:
            return empty_violations()
        age = pd.to_numeric(df["age"], errors="coerce")
        employment_years = pd.to_numeric(df["employment_years"], errors="coerce")
        mask = (
            employment_years.notna()
            & age.notna()
            & (employment_years > (age - self.min_working_age))
        )
        if not mask.any():
            return empty_violations()
        bad = df.loc[mask]
        keys = record_key(self.table, bad)
        details = [
            f"employment_years={e} > age={a} - {self.min_working_age}"
            for e, a in zip(employment_years.loc[mask], age.loc[mask], strict=True)
        ]
        return make_violations(self.table, keys, details)


class TemporalLeakageRule(Rule):
    """Flags a snapshot dated after every one of that customer's applications.

    financial_profiles/credit_history rows are point-in-time snapshots keyed
    on (customer_id, as_of_date), not tied to a specific loan_id. A snapshot
    is expected to precede (support) at least one of that customer's loan
    applications; if its as_of_date is later than all of them, it could not
    have been available for any real decision -- a future-information leak.
    """

    def __init__(
        self,
        snapshot_table: str,
        loan_table: str = "loan_applications",
        severity: Severity = Severity.ERROR,
    ) -> None:
        super().__init__(
            name=f"temporal_leakage:{snapshot_table}",
            description=f"{snapshot_table}.as_of_date later than every "
            f"{loan_table}.application_date for that customer",
            severity=severity,
            scope=Scope.ROW,
        )
        self.snapshot_table = snapshot_table
        self.loan_table = loan_table

    def check(self, dataframes: DataFrames) -> pd.DataFrame:
        snapshots = dataframes.get(self.snapshot_table)
        loans = dataframes.get(self.loan_table)
        if snapshots is None or snapshots.empty or loans is None or loans.empty:
            return empty_violations()

        latest_application = (
            pd.to_datetime(loans["application_date"])
            .groupby(loans["customer_id"])
            .max()
        )

        as_of = pd.to_datetime(snapshots["as_of_date"])
        latest_for_row = snapshots["customer_id"].map(latest_application)
        mask = latest_for_row.notna() & (as_of > latest_for_row)
        if not mask.any():
            return empty_violations()
        bad = snapshots.loc[mask]
        keys = record_key(self.snapshot_table, bad)
        details = [
            f"as_of_date={a.date()} later than every application_date "
            f"({latest_for_row.loc[i].date()}) for {bad.loc[i, 'customer_id']}"
            for i, a in zip(bad.index, as_of.loc[mask], strict=True)
        ]
        return make_violations(self.snapshot_table, keys, details)
