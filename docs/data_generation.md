# Synthetic data generation

CreditGuard trains and evaluates entirely on synthetic data (`src/creditguard/data/`).
This document explains how that data is generated, so a reviewer can follow the
generating process end to end and so Phase 7's SHAP feature importances can later be
checked against the ground-truth process described here.

Every parameter referenced below lives in
[`config/data_generation.yaml`](../config/data_generation.yaml) — nothing is
hard-coded in `generator.py`.

## Overview

A single call to `generate_dataset(config, seed, n_customers)`
(`src/creditguard/data/generator.py`) produces five tables:

1. **customers** — one row per applicant: demographics, employment, income.
2. **loan_applications** — 1–3 rows per customer, one per loan applied for.
3. **financial_profiles** — one point-in-time financial snapshot per loan application.
4. **credit_history** — one point-in-time bureau snapshot per loan application.
5. **loan_outcomes** — the observed 12-month default label per loan, kept in a
   separate table (see [Point-in-time correctness](#point-in-time-correctness)).

Generation is fully vectorised with NumPy/pandas (no per-row Python loops), so
50,000 customers (~100,000 loans) generate in well under a second.

## Reproducibility

A single `numpy.random.Generator` is seeded once from `config["seed"]` (default 42,
overridable via `--seed`) and threaded through every sampling call. The dates window
is anchored to a **fixed** `dates.reference_date` in the config rather than
wall-clock "today", so the same seed and config always produce byte-identical
tables regardless of when the generator runs.

## Demographic and financial structure

Columns are not sampled independently; each is built from the ones before it so the
resulting dataset has the correlation structure a real credit bureau would show:

- **age** — truncated normal, clipped to `[21, 70]`.
- **education**, **employment_type** — `employment_type` is sampled conditionally on
  `education` (`demographics.employment_type.probs_given_education`), so, e.g.,
  postgraduates are more likely to be salaried or business owners than high-school
  graduates.
- **employment_years** — bounded above by `age - 18` (scaled down further for
  `UNEMPLOYED` customers), truncated normal within that bound.
- **annual_income** — lognormal, with the log-mean shifted by education, employment
  type, an age premium above 25, and city tier (`income.*_shift`); `monthly_income`
  is `annual_income / 12` plus small relative noise.
- **monthly_expenses**, **monthly_emi**, **existing_loan_count/amount**,
  **savings_balance** — all scale off `monthly_income`, with dependents raising
  expenses, existing loan count raising EMI burden, and DTI (derived from the two)
  reducing savings (`savings.dti_penalty`).
- **credit_history_months** — grows with age (`credit_bureau.months_per_adult_year_mean`).
- **total_credit_limit** — scales with income and credit history length.
- **credit_utilization** — `Beta(alpha, beta)`, shifted upward for high DTI and low
  savings-to-income (`credit_bureau.utilization_*_shift`), clipped to `[0, 2]`.
- **previous_defaults**, **late/missed_payments_12m** — zero-inflated / capped
  Poisson draws whose rate depends on a latent per-customer `risk_propensity`
  (see below).
- **loan_amount** — scales with income and a loan-type-specific multiple
  (`loan_application.amount_income_multiple`); **interest_rate** rises with a
  loan-type base rate plus the customer's `risk_propensity`.

### The latent risk_propensity device

Each customer gets a `risk_propensity ~ Normal(0, 1)`, generated once and used only
internally — it is **never written to any table**. It biases `previous_defaults`,
`late_payments_12m`, `missed_payments_12m` and `interest_rate` so that these
bureau-observable fields correlate with each other the way real credit-risk data
does (someone who has missed payments before is more likely to have prior defaults
too), without being a hidden input to the label itself. The label (below) reads
*only* from real, stored columns, so it stays fully explainable from the data a
model would actually see.

### Point-in-time snapshots

Each loan application gets its own `financial_profiles` and `credit_history` row,
dated `as_of_date = application_date - snapshot_lag_days` (1–30 days before, per
`dates.snapshot_lag_days_*`). Snapshot values are the customer's underlying trait
plus small independent noise (`credit_bureau.snapshot_noise_std`); `credit_history_months`
is additionally adjusted backward in time relative to the customer's most recent
snapshot, so a snapshot taken two loans ago correctly shows less history than the
customer's latest one.

### Point-in-time correctness

`application_date` is spread uniformly across a 36-month window
(`dates.window_months`) ending 12 months before the reference date
(`dates.outcome_lag_months`), so every loan has an observable 12-month outcome.
Every `financial_profiles`/`credit_history` snapshot is generated **strictly
before** the application it supports (`as_of_date < application_date`), and
`loan_outcomes` is a separate table from `loan_applications` specifically so
feature-engineering code (Phase 4) cannot accidentally join outcome information
into the feature set — the forbidden-feature registry there is enforced at fit and
transform time, not only in tests.

## The label

The 12-month default label comes from a documented linear model on the logit scale,
computed per loan application from real, stored columns:

```text
logit = b0
      + dti_coef              * dti
      + credit_utilization_coef * credit_utilization
      + previous_defaults_coef  * previous_defaults
      + late_payments_12m_coef  * late_payments_12m
      + emi_to_income_coef      * emi_to_income
      + loan_to_income_coef     * loan_to_income
      - log_monthly_income_coef * log(monthly_income)
      - employment_years_coef   * employment_years
      - credit_history_months_coef * credit_history_months
      - savings_to_income_coef  * savings_to_income
      + dti_x_credit_utilization_coef * dti * credit_utilization
      + Normal(0, noise_std)

p_default = sigmoid(logit)
default_12m ~ Bernoulli(p_default)
```

`dti = (monthly_expenses + monthly_emi) / monthly_income`,
`emi_to_income = monthly_emi / monthly_income`,
`loan_to_income = loan_amount / annual_income`,
`savings_to_income = savings_balance / monthly_income` — all computed from the same
point-in-time snapshot as the rest of the row, and all present as stored columns
(directly or via simple arithmetic on stored columns).

**Coefficients** (from `config/data_generation.yaml`, `label.coefficients` /
`label.interaction`):

| Term | Coefficient | Sign |
|---|---|---|
| `dti` | 2.4 | + |
| `credit_utilization` | 1.8 | + |
| `previous_defaults` | 0.55 | + |
| `late_payments_12m` | 0.30 | + |
| `emi_to_income` | 2.0 | + |
| `loan_to_income` | 0.7 | + |
| `log_monthly_income` | 0.85 | − |
| `employment_years` | 0.035 | − |
| `credit_history_months` | 0.010 | − |
| `savings_to_income` | 0.45 | − |
| `dti × credit_utilization` (interaction) | 0.9 | + |
| noise | `Normal(0, 0.45)` | — |

**Intercept calibration.** `b0` is not fixed in config: at generation time,
`_calibrate_intercept` runs a bisection search so that
`mean(sigmoid(logit_without_intercept + b0))` lands on the midpoint of
`[target_default_rate_min, target_default_rate_max]` = `[0.08, 0.14]` → target
`0.11`. The calibrated `b0` and the realised default rate are written to
`metadata.json` for every generated dataset (see [Versioning](#versioning-and-files)).

At the default config (`n_customers=50000`, `seed=42`), this run produced:

- **Calibrated intercept `b0`:** `4.6436` for the reference run
  (`seed=42`, `n_customers=50000`) used to validate this phase. Also written to
  `calibrated_intercept_b0` in every generated dataset's `metadata.json`
  (recomputed per run; stable given the same seed/config).
- **Realised default rate:** `0.1105` (11.05%), within the `[0.08, 0.14]` target band.

No single one of these terms dominates: a model that sees only one raw feature
(`credit_utilization`, `dti`, `annual_income`, `previous_defaults`, `loan_amount`, or
`credit_history_months`) cannot exceed AUC 0.75 on the label (enforced by
`tests/test_generator.py::test_no_single_feature_achieves_high_auc`) — the signal is
genuinely distributed across the feature set, which is what makes this a meaningful
modelling exercise rather than a lookup table.

## Injected data quality issues

`config/data_generation.yaml`'s `data_quality_injection` section controls six
categories of deliberately bad rows, each drawn from a **disjoint** random sample of
row indices (a customer or row is never hit by more than one category at once, so
counts are unambiguous). Every injected row is recorded in
`injection_manifest.json` (counts, table, affected keys) so Phase 3's tests can
assert detection rates against a known ground truth.

| Category | Default rate | Table / column | What happens at ingest |
|---|---|---|---|
| `missing_value` | 2% of financial_profiles | `financial_profiles.total_assets` → `NULL` | NOT NULL violation → quarantined |
| `duplicate_customer` | 1% of customers | duplicate `customers` row appended | PRIMARY KEY violation → quarantined |
| `out_of_range_age` | 0.5% of customers | `customers.age` set outside `[18, 100]` | CHECK violation → quarantined (cascades — see below) |
| `negative_financial_value` | 0.5% of customers | `customers.annual_income` negated | CHECK violation → quarantined (cascades) |
| `impossible_utilization` | 1% of credit_history | `credit_history.credit_utilization` set outside `[0, 2]` | CHECK violation → quarantined |
| `inconsistent_income` | 2% of customers | `customers.monthly_income` scaled by an implausible factor (0.1×–5×) | **Passes every DB constraint** — a semantic issue for Phase 3's validation rules, not caught here |

**Cascading failures.** `out_of_range_age` and `negative_financial_value` corrupt a
column on the `customers` row itself, so that customer row fails to insert — every
`loan_applications`, `financial_profiles`, `credit_history` and `loan_outcomes` row
that references that `customer_id` then also fails, via a foreign-key violation, and
is quarantined too. This is intentional and realistic (referential integrity doing
its job), and `tests/test_ingest.py` accounts for it explicitly when computing
expected quarantine counts.

## Ingestion

`python -m creditguard.data.ingest --dataset-version <v> [--truncate]`
(`src/creditguard/data/ingest.py`) loads the five parquet tables into PostgreSQL in
foreign-key-safe order (customers → loan_applications → financial_profiles →
credit_history → loan_outcomes), inside a single transaction.

Rather than aborting the whole load on the first bad row, each table's rows are
inserted via **recursive bisection**: try a bulk multi-row insert; if it fails,
split the batch in half and retry each half, recursing down to individual rows only
where a real constraint violation lives. Because injected bad rows are spread
uniformly (not clustered), this keeps the common case — the ~95%+ of rows that are
clean — in a handful of large bulk statements instead of one round trip per row.
Each attempt runs inside a `SAVEPOINT`, so a failure rolls back only that attempt,
not the surrounding transaction. Every row that ultimately fails is logged to
`data_quality_issues` with the table, a natural-key `record_key`, the violated
constraint (or column, for NOT NULL violations), and the database's own error
message — then the load continues.

## Versioning and files

`make_dataset_version(config)` names each run `ds_<YYYYMMDD>_<8-char config hash>`
(`src/creditguard/data/versioning.py`), where the hash covers the *effective*
config (including any `--n-customers`/`--seed` CLI overrides), so two runs with
different parameters never collide. Each run writes:

```text
data/raw/<dataset_version>/
├── customers.parquet
├── loans.parquet
├── financials.parquet
├── credit.parquet
├── outcomes.parquet
├── metadata.json             # row counts, full config snapshot, seed,
│                              # label coefficients, calibrated b0,
│                              # achieved default rate, generated_at
└── injection_manifest.json   # per-category counts and affected keys
```

## Known simplifications

- Every loan application is treated as approved (`status = 'APPROVED'`); there is no
  rejected-application branch, since every application needs an observable 12-month
  outcome for this modelling exercise.
- `loan_purpose` is sampled independently of `loan_type` rather than being coupled
  (e.g. a `HOME` loan could get purpose `MEDICAL`); this keeps the generator simple
  and doesn't affect the label design.
- Point-in-time snapshot noise is i.i.d. per loan rather than a true time-series
  (e.g. no autocorrelated income trend across a customer's snapshots), except for
  `credit_history_months`, which is explicitly adjusted for elapsed time so it never
  decreases going forward in time for the same customer.
