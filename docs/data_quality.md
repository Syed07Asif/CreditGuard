# Data quality: validation and cleaning (Phase 3)

This phase adds two deliberately separate concerns:

- **Validation** (`src/creditguard/validation/rules.py`, `engine.py`) only *observes*
  data and reports what is wrong. It never mutates a DataFrame.
- **Cleaning** (`src/creditguard/validation/cleaning.py`) is the only place data gets
  repaired, and every repair is logged.

Both operate on the **raw generated parquet tables** for a `dataset_version`
(`data/raw/<dataset_version>/`), not on what has already been loaded into
PostgreSQL. This matters: five of the six Phase 2 injected error types
(`out_of_range_age`, `negative_financial_value`, `duplicate_customer`,
`missing_value`, `impossible_utilization`) violate a database CHECK, PRIMARY
KEY or NOT NULL constraint, so `creditguard.data.ingest` already quarantines
those rows *out of the database* before this phase ever runs. Only
`inconsistent_income` passes every DB constraint and survives ingest
unmodified (see `config/data_generation.yaml`'s `data_quality_injection`
section). Validating against the raw dataset version is the only way to
measure detection rate against the full injected-error manifest.

## Rule catalogue

All rules and their parameters live in `config/validation_rules.yaml`, under
the `rules` section. Each entry becomes one instance of a `Rule` subclass
registered into a `RuleRegistry` by `engine.build_registry`.

| Rule | Checks | Typical severity |
|---|---|---|
| `MissingValueRule` | NULLs in one column, once missing fraction exceeds a threshold | WARNING (repairable via imputation) or ERROR for columns with zero tolerance |
| `DuplicateRecordRule` | Exact-row duplicates, or duplicates on a natural key (`customer_id`, `loan_id`) | ERROR (key-based), WARNING (exact-row, general robustness) |
| `NumericRangeRule` | Configurable `[min, max]` plausibility band on any numeric column | WARNING |
| `NegativeFinancialRule` | income/expense/EMI/limit/balance columns must be `>= 0` | ERROR |
| `ImpossibleAgeRule` | `age` outside `[18, 100]` | ERROR |
| `InvalidDateRule` | Unparseable dates, future dates, `decision_date` before `application_date` | ERROR (loan_applications), WARNING (other date columns) |
| `CreditUtilizationRule` | `credit_utilization` outside `[0, 1.5]`, or `total_outstanding` far exceeds `total_credit_limit` | ERROR |
| `IncomeConsistencyRule` | `abs(annual_income - 12*monthly_income)` exceeds a relative tolerance | ERROR |
| `ExpenseConsistencyRule` | `monthly_expenses + monthly_emi > monthly_income * 1.5` | WARNING |
| `OrphanRecordRule` | Child rows (loan/financial/credit/outcome) with no matching parent | ERROR |
| `EmploymentPlausibilityRule` | `employment_years > age - 16` | WARNING |
| `TemporalLeakageRule` | A financial/credit snapshot's `as_of_date` later than every one of that customer's `application_date`s | ERROR |

`financial_profiles` and `credit_history` are point-in-time snapshots keyed on
`(customer_id, as_of_date)`, not on a specific `loan_id` -- there is no
`loan_id` column to join against. `TemporalLeakageRule` therefore treats a
snapshot as leaking future information if its `as_of_date` postdates *every*
application for that customer (it could not have supported any real
decision), rather than trying to match it to one specific loan.

### Severity and the repair/drop split

- **ERROR** violations mark a row `is_quarantined = True` (see
  `ValidationResult.split_clean_and_quarantined`). Rows are never silently
  dropped by validation -- only flagged.
- Cleaning decides what happens next: `NegativeFinancialRule`,
  `ImpossibleAgeRule`, `CreditUtilizationRule` and `DuplicateRecordRule`
  violations are all **repairable** (clip into bounds, or deduplicate), so
  cleaning fixes them *before* re-checking. What's left after repair --
  `OrphanRecordRule`, `TemporalLeakageRule`, `InvalidDateRule`,
  `IncomeConsistencyRule` -- cannot be sensibly clipped or imputed (which of
  two conflicting income figures is correct?), so those rows are dropped from
  the modelling set, cascading to dependent tables to keep referential
  integrity.
- **WARNING** violations (`MissingValueRule` on `total_assets`,
  `ExpenseConsistencyRule`, `EmploymentPlausibilityRule`, exact-row
  duplicates) are logged but never quarantine a row -- they're either
  genuinely fixable by imputation or not worth losing a row over.

All ERROR and WARNING violations are persisted to `data_quality_issues`
(`ValidationResult.to_records()`), regardless of what cleaning later does with
the row.

## Cleaning pipeline

`DataCleaner` (`cleaning.py`) is a scikit-learn compatible transformer
(`fit`/`transform`, extends `BaseEstimator`/`TransformerMixin`) operating on
the same `dict[str, DataFrame]` shape as
`creditguard.data.versioning.read_dataset_tables`, so it composes into a
`sklearn.pipeline.Pipeline` in Phase 4. Order of operations in `transform`:

1. **Deduplicate** on configured natural keys (`customer_id`, `loan_id`),
   keeping the most recent (last) row.
2. **Clip** physically impossible values to configured bounds (`age`,
   negative financial columns, `credit_utilization`) -- domain constants, not
   statistics, so no fit/leakage concern.
3. **Winsorise** extreme financial outliers at the 1st/99th percentile,
   *fit on training data only* and stored on the transformer.
4. **Median-impute** numeric columns (`total_assets`, income columns),
   *fit on training data only*, adding a `<col>_was_missing` indicator column.
5. **Categorical-impute**: missing values become the explicit `'UNKNOWN'`
   category.
6. **Re-validate and drop**: run the same rule registry again; any row still
   carrying an ERROR violation is structural (steps 1-5 could not repair it)
   and is dropped from the returned tables, with `customer_id`/`loan_id`
   cascaded to dependent tables. Dropped rows are kept on
   `cleaner.last_report_.quarantined_rows` for the caller to log --
   never silently discarded.

Because every step is either idempotent by construction (clip, dedup) or a
no-op once its target condition no longer holds (impute, quarantine-drop),
cleaning already-clean data is idempotent: `transform(transform(X)) ==
transform(X)`.

**Fit/transform and training data**: `fit()` learns two things --
imputation medians and winsorize percentile bounds -- from whatever tables
are passed to it, and persists them on the transformer instance (`medians_`,
`winsor_bounds_`). `transform()` never recomputes them. The current CLI fits
on the same dataset version it cleans (Phase 3 does not yet define a
train/test split -- that's Phase 6's job); the important guarantee, exercised
directly by `tests/test_cleaning.py`, is that once fit, the *same* learned
statistics apply to any data passed to `transform()`, so a later phase can
fit once on a training split and reuse the persisted `DataCleaner` (see
`save_cleaner`/`load_cleaner`, backed by `joblib`) on validation/test/production
data without recomputing anything from it.

## Reports

`report.py` renders `reports/data_quality/<dataset_version>.md` and
`.html`, each containing: row counts in/out per table, a rule -> table ->
severity -> count -> % of rows table, the top 20 example violations, and
before/after summary statistics for every numeric column (only "before" for
`validate`; both for `clean`).

## CLI

```bash
# Validate a raw dataset version: writes a report and persists issues to
# data_quality_issues. Does not modify any data.
python -m creditguard.validation.cli validate --dataset-version ds_20260808_abcd1234

# Clean a raw dataset version into a new versioned, modelling-ready dataset
# under data/processed/<output-version>/, plus a persisted DataCleaner
# (models/artifacts/validation/cleaner_<output-version>.joblib).
python -m creditguard.validation.cli clean \
    --dataset-version ds_20260808_abcd1234 \
    --output-version ds_20260808_abcd1234_clean
```

Both subcommands accept `--config` (default `config/validation_rules.yaml`),
`--data-dir` and `--reports-dir` overrides; `clean` additionally accepts
`--output-dir`.
