"""Deterministic, documented cleaning pipeline for CreditGuard dataset tables.

Validation (`rules.py` / `engine.py`) only observes; this module is the only
place that transforms data. `DataCleaner` is a scikit-learn compatible
transformer (fit/transform) over the dict-of-DataFrames shape produced by
`creditguard.data.versioning.read_dataset_tables`, so Phase 4 can compose it
into a `sklearn.pipeline.Pipeline`.

Order of operations in `transform`, each logged, nothing silently dropped:
  1. Deduplicate on configured natural keys, keeping the most recent row.
  2. Clip physically impossible values to configured bounds (domain constants,
     not fit from data -- no leakage risk).
  3. Winsorise extreme financial outliers at percentile bounds *fit on
     training data only* (see `fit`).
  4. Median-impute numeric columns using *training-fit* medians, adding a
     `<col>_was_missing` indicator column.
  5. Fill categorical missing values with the explicit 'UNKNOWN' category.
  6. Re-run the validation rule registry; rows that still carry an
     ERROR-severity violation are structural issues steps 1-5 cannot repair
     (orphan records, temporal leakage, unreconcilable income, ...) and are
     dropped from the returned "modelling set" tables, with referential
     integrity cascaded to dependent tables. They are kept in
     `last_report_.quarantined_rows` for the caller to log/persist separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from creditguard.validation.engine import build_registry, run
from creditguard.validation.rules import KEY_COLUMNS, RuleRegistry

Tables = dict[str, pd.DataFrame]


@dataclass
class ClipLogEntry:
    """One clip operation: how many values were pulled into bounds."""

    table: str
    column: str
    n_clipped_low: int
    n_clipped_high: int


@dataclass
class WinsorizeLogEntry:
    """One winsorize operation: the fitted bounds and how many values moved."""

    table: str
    column: str
    lower_bound: float
    upper_bound: float
    n_clipped_low: int
    n_clipped_high: int


@dataclass
class CleaningReport:
    """Everything the most recent `transform()` call did."""

    dedup_log: dict[str, int] = field(default_factory=dict)
    clip_log: list[ClipLogEntry] = field(default_factory=list)
    winsorize_log: list[WinsorizeLogEntry] = field(default_factory=list)
    imputed_log: dict[str, dict[str, int]] = field(default_factory=dict)
    quarantined_rows: Tables = field(default_factory=dict)


class DataCleaner(BaseEstimator, TransformerMixin):
    """Deduplicate, clip, winsorise, impute and quarantine-drop CreditGuard tables."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def fit(self, X: Tables, y: Any = None) -> DataCleaner:  # noqa: N803
        """Learn training-only statistics: imputation medians and winsorize bounds.

        Statistics are computed after deduplication and physical-bound clipping
        (both deterministic, config-driven repairs, not statistics fit from data)
        so that duplicated rows and impossible values injected upstream don't
        skew the learned medians/percentiles. This is still "fit on training
        data only" -- it never looks at data passed to `transform`.
        """
        cleaning_cfg = self.config.get("cleaning", {})
        fit_source = self._clip(
            self._dedup({k: v.copy() for k, v in X.items()}, CleaningReport()),
            CleaningReport(),
        )

        medians: dict[str, dict[str, float]] = {}
        for entry in cleaning_cfg.get("impute", {}).get("numeric", []):
            table, column = entry["table"], entry["column"]
            df = fit_source.get(table)
            if df is None or column not in df.columns:
                continue
            median = pd.to_numeric(df[column], errors="coerce").median()
            medians.setdefault(table, {})[column] = float(median)
        self.medians_ = medians

        winsor_cfg = cleaning_cfg.get("winsorize", {})
        lower_q = winsor_cfg.get("lower_percentile", 0.01)
        upper_q = winsor_cfg.get("upper_percentile", 0.99)
        winsor_bounds: dict[str, dict[str, tuple[float, float]]] = {}
        for entry in winsor_cfg.get("columns", []):
            table, column = entry["table"], entry["column"]
            df = fit_source.get(table)
            if df is None or column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            lo, hi = values.quantile([lower_q, upper_q])
            winsor_bounds.setdefault(table, {})[column] = (float(lo), float(hi))
        self.winsor_bounds_ = winsor_bounds

        self.rule_registry_: RuleRegistry = build_registry(self.config)
        self.fitted_ = True
        return self

    def transform(self, X: Tables) -> Tables:
        """Apply dedup, clip, winsorize, impute and quarantine-drop to `X`."""
        if not getattr(self, "fitted_", False):
            raise RuntimeError("DataCleaner.transform() called before fit()")

        tables = {name: df.copy() for name, df in X.items()}
        report = CleaningReport()

        tables = self._dedup(tables, report)
        tables = self._clip(tables, report)
        tables = self._winsorize(tables, report)
        tables = self._impute_numeric(tables, report)
        tables = self._impute_categorical(tables, report)
        tables = self._quarantine_and_drop(tables, report)

        self.last_report_ = report
        return tables

    def _dedup(self, tables: Tables, report: CleaningReport) -> Tables:
        for entry in self.config.get("cleaning", {}).get("dedup", []):
            table, keys = entry["table"], entry["keys"]
            df = tables.get(table)
            if df is None:
                continue
            before = len(df)
            df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
            report.dedup_log[table] = report.dedup_log.get(table, 0) + (
                before - len(df)
            )
            tables[table] = df
        return tables

    def _clip(self, tables: Tables, report: CleaningReport) -> Tables:
        for entry in self.config.get("cleaning", {}).get("clip", []):
            table, column = entry["table"], entry["column"]
            df = tables.get(table)
            if df is None or column not in df.columns:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            lo, hi = entry.get("min"), entry.get("max")
            n_low = int((values < lo).sum()) if lo is not None else 0
            n_high = int((values > hi).sum()) if hi is not None else 0
            if n_low or n_high:
                df[column] = values.clip(lower=lo, upper=hi)
                report.clip_log.append(
                    ClipLogEntry(
                        table=table,
                        column=column,
                        n_clipped_low=n_low,
                        n_clipped_high=n_high,
                    )
                )
                tables[table] = df
        return tables

    def _winsorize(self, tables: Tables, report: CleaningReport) -> Tables:
        for table, columns in self.winsor_bounds_.items():
            df = tables.get(table)
            if df is None:
                continue
            for column, (lo, hi) in columns.items():
                if column not in df.columns:
                    continue
                values = pd.to_numeric(df[column], errors="coerce")
                n_low = int((values < lo).sum())
                n_high = int((values > hi).sum())
                if n_low or n_high:
                    df[column] = values.clip(lower=lo, upper=hi)
                    report.winsorize_log.append(
                        WinsorizeLogEntry(
                            table=table,
                            column=column,
                            lower_bound=lo,
                            upper_bound=hi,
                            n_clipped_low=n_low,
                            n_clipped_high=n_high,
                        )
                    )
            tables[table] = df
        return tables

    def _impute_numeric(self, tables: Tables, report: CleaningReport) -> Tables:
        for table, columns in self.medians_.items():
            df = tables.get(table)
            if df is None:
                continue
            for column, median in columns.items():
                if column not in df.columns:
                    continue
                missing = df[column].isna()
                df[f"{column}_was_missing"] = missing
                n_missing = int(missing.sum())
                if n_missing:
                    df.loc[missing, column] = median
                    report.imputed_log.setdefault(table, {})[column] = n_missing
            tables[table] = df
        return tables

    def _impute_categorical(self, tables: Tables, report: CleaningReport) -> Tables:
        for entry in (
            self.config.get("cleaning", {}).get("impute", {}).get("categorical", [])
        ):
            table, column = entry["table"], entry["column"]
            df = tables.get(table)
            if df is None or column not in df.columns:
                continue
            missing = df[column].isna()
            n_missing = int(missing.sum())
            if n_missing:
                df[column] = df[column].astype(object)
                df.loc[missing, column] = "UNKNOWN"
                report.imputed_log.setdefault(table, {})[column] = n_missing
                tables[table] = df
        return tables

    def _quarantine_and_drop(self, tables: Tables, report: CleaningReport) -> Tables:
        result = run(tables, self.rule_registry_, dataset_version="cleaning-pass")
        cleaned: Tables = {}
        for table, df in tables.items():
            if table in KEY_COLUMNS:
                clean_df, quarantined_df = result.split_clean_and_quarantined(table)
            else:
                clean_df, quarantined_df = df, df.iloc[0:0]
            cleaned[table] = clean_df
            if not quarantined_df.empty:
                report.quarantined_rows[table] = quarantined_df
        return _cascade_referential_integrity(cleaned)


def _cascade_referential_integrity(tables: Tables) -> Tables:
    """Drop dependent rows whose parent row was quarantined out, keeping FKs intact."""
    if "customers" in tables:
        valid_customers = set(tables["customers"]["customer_id"].astype(str))
        for child in ("loan_applications", "financial_profiles", "credit_history"):
            if child in tables:
                tables[child] = tables[child][
                    tables[child]["customer_id"].astype(str).isin(valid_customers)
                ].reset_index(drop=True)

    if "loan_applications" in tables and "loan_outcomes" in tables:
        valid_loans = set(tables["loan_applications"]["loan_id"].astype(str))
        tables["loan_outcomes"] = tables["loan_outcomes"][
            tables["loan_outcomes"]["loan_id"].astype(str).isin(valid_loans)
        ].reset_index(drop=True)

    return tables


def save_cleaner(cleaner: DataCleaner, path: str) -> None:
    """Persist a fitted DataCleaner (learned medians/winsorize bounds) with joblib."""
    joblib.dump(cleaner, path)


def load_cleaner(path: str) -> DataCleaner:
    """Load a fitted DataCleaner previously saved with `save_cleaner`."""
    return joblib.load(path)
