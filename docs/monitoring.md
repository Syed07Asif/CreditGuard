# Monitoring (Phase 10)

A credit model degrades silently: the applicant population shifts, the model keeps
returning confident numbers, and nobody notices until losses appear. This document
covers what CreditGuard watches, why the thresholds are set where they are, how to
read a PSI value, and how a drift/performance signal turns into a retraining decision.

This is an educational simulation. The thresholds below are reasonable defaults for a
synthetic-data portfolio project, not calibrated against real loss data — see
`docs/scoring_methodology.md`'s own disclaimer for the same caveat applied to scoring.

---

## What is monitored

| Module | What it computes | Reads from | Writes to |
|---|---|---|---|
| `monitoring/baseline.py` | A frozen snapshot of the active model's training population (per-feature mean/std/min/max/decile edges, category frequencies, predicted-probability deciles, observed default rate) | The active model's own training split | A JSON+parquet artifact under `models/artifacts/baselines/`, plus a pointer on `model_registry.metrics` |
| `monitoring/drift.py` | PSI, KS, chi-square, prediction-probability PSI, concept-drift proxy | Recent production applications (`customers`/`loan_applications`/`financial_profiles`/`credit_history`) + recent `predictions` + `loan_outcomes`, against the baseline | `drift_reports` (one row per feature per method per run) |
| `monitoring/performance.py` | ROC-AUC/PR-AUC/KS/Brier/precision/recall/calibration slope on matured loans; prediction volume, decision mix, avg score/probability, high-risk share, p95 latency | `predictions` joined to `loan_outcomes` (matured only); all `predictions` (operational metrics) | `monitoring_metrics` |
| `monitoring/data_quality.py` | Re-runs the Phase 3 rule engine over recent production records | `customers`/`loan_applications`/`financial_profiles`/`credit_history`/`loan_outcomes` | `data_quality_issues` |
| `monitoring/alerts.py` | Turns any of the above into an `Alert` (severity, triggering values, recommended action) | — | console, a log file (`reports/monitoring/alerts.log`), and an optional webhook |
| `monitoring/retraining.py` | `should_retrain()` / `trigger_retraining()` / `rollback_to_version()` | The above three checks | a new `model_registry` row (inactive until promoted) |

`monitoring/scheduler.py` runs drift + performance + data-quality + `should_retrain`
on an interval (`MONITORING_INTERVAL_MINUTES`, default 60) inside the `monitoring`
Docker service. `GET /api/v1/monitoring/{drift,performance,data-quality}` reads back
whatever the most recent run wrote — the dashboard's Monitoring page never triggers a
check itself.

---

## Why the baseline is the ACTIVE model's own training data

Drift is always measured against the baseline built when the currently active model
was promoted — never against "last week's production data". A moving baseline would
absorb genuine drift a little at a time and never cross a threshold; comparing against
a fixed reference point is what makes drift visible at all. `baseline.py`'s docstring
calls this out directly. Promoting a new model (via the pipeline orchestrator's
`--register` stage or `retraining.trigger_retraining`'s promotion path) always builds
a fresh baseline for it.

---

## Population Stability Index (PSI)

```
PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)
```

`expected%` is the baseline population's share of each bin (uniform 10% per bin for
numeric features, since bins are the baseline's own deciles); `actual%` is the current
window's share of the same bins. Reading it:

| PSI | Status | Meaning |
|---|---|---|
| < 0.10 | **OK** | Distribution has not meaningfully shifted. |
| 0.10 – 0.25 | **WARNING** | Some shift — worth watching, not yet alarming. |
| ≥ 0.25 | **DRIFT** | Substantial shift — the population being scored looks different from what the model was trained on. |

These are the standard PSI bands used across the credit-risk industry, not something
CreditGuard invented. **Empty-bin handling:** an expected or actual bin of exactly 0
would make the `ln(actual/expected)` term blow up (`ln(0)` or a division by zero); both
sides are floored to a small epsilon (`config/monitoring.yaml`'s `drift.epsilon`,
default `1e-6`) before the ratio is taken. There is no universally "correct" way to
handle an empty bin in the PSI literature — flooring is the conventional choice, and
it still penalises the emptied/filled bin heavily rather than silently ignoring it.

**Numeric features** are binned into the baseline's own decile edges (never rebinned
from current data — see the "why a fixed baseline" note above). **Categorical
features** use category-frequency bins instead, and a category with zero baseline mass
is reported separately as "new category" and forces DRIFT regardless of its PSI
contribution (a single small new category can be swamped by chi-square's own p-value
otherwise).

**Prediction-probability PSI is reported first and separately** from every per-feature
finding, because a shift in the *scored population's* predicted-probability
distribution is often the earliest signal — it can move before any single input
feature individually crosses its own DRIFT threshold, since it aggregates the combined
effect of many small feature shifts.

---

## Kolmogorov-Smirnov and chi-square

- **KS** (continuous features): a genuine two-sample test between the baseline's
  persisted reference sample (`config/monitoring.yaml`'s
  `baseline.reference_sample_size`, default 5000 rows — PSI/chi-square only need
  summary statistics, but KS needs an actual second sample) and current values. A
  p-value below `drift.significance_level` (default 0.05) is read as DRIFT.
- **Chi-square** (categorical features): observed current category counts vs. the
  counts implied by baseline frequencies at the current sample size. Also flags a
  brand-new category directly, the same way PSI does.

## Concept drift proxy

The rolling observed default rate among *matured* loans (a `predictions` row whose
`loan_id` has a `loan_outcomes` row with an observed outcome in the window) compared
to the baseline's training-time default rate, with a Wilson-score binomial confidence
interval on the current rate (more reliable than a naive normal approximation at
typical monitoring-window sample sizes). If the baseline rate falls outside that
interval, status is WARNING — this is a proxy for "is the underlying risk of the
population changing", not a replacement for the feature-level drift checks above.

---

## Performance monitoring

For matured loans only (an unmatured `predictions` row — most `/predict` calls, which
never get a real-world outcome recorded — is correctly excluded), the same metric
suite Phase 6 uses at training time (`models.evaluate.full_metric_suite`) is
recomputed and compared against that model's training-time values. A metric is flagged
once it degrades beyond `performance.degradation_tolerance` (default 10%, relative):

- **Higher-is-better** (ROC-AUC, PR-AUC, KS, precision, recall): degraded if
  `current < training * (1 - tolerance)`.
- **Lower-is-better** (Brier score): degraded if `current > training * (1 + tolerance)`.
- **Calibration slope** (target 1.0, not "as high as possible"): degraded if its
  *distance from 1.0* has grown by more than `tolerance` in absolute terms — a relative
  comparison would be hypersensitive whenever training happened to be very close to
  1.0 already.

Below `performance.min_matured_loans` (default 30) matured loans, metrics are still
computed and reported but never used to flag degradation or feed `should_retrain` —
too small a sample to trust.

Operational metrics (prediction volume, approve/review/reject mix, average
score/probability, high-risk share, p95 latency) don't need a label and are always
computed from every `predictions` row in the window.

---

## Data quality monitoring

Re-runs the exact Phase 3 rule engine (`validation.engine`) over recent production
records (not the training dataset) on the same schedule. The **violation rate**
(share of rows flagged by at least one rule) crossing `data_quality.violation_rate_warning`
(2%) or `..._alert` (5%) triggers a WARNING/CRITICAL alert respectively — a rising
violation rate over time usually means an upstream data source changed, not random
noise.

---

## Alerts

Every check above can produce an `Alert`: `severity` (INFO/WARNING/CRITICAL),
`category`, the `triggering_values`, and a `recommended_action`. Alerts fan out to
every configured sink (console always, a log file always, an optional webhook if
`ALERT_WEBHOOK_URL` is set) — a broken sink is logged and skipped, never allowed to
block the others or the monitoring run itself.

| Trigger | Severity |
|---|---|
| A feature at DRIFT status | WARNING |
| Prediction-probability PSI at DRIFT status | **CRITICAL** (earliest signal) |
| Concept drift (baseline rate outside the current CI) | WARNING |
| A performance metric degraded beyond tolerance | WARNING |
| Data-quality violation rate ≥ warning threshold | WARNING |
| Data-quality violation rate ≥ alert threshold | CRITICAL |
| Active model failed to load | CRITICAL |
| A challenger did not beat the champion | WARNING (asks for human review) |

---

## Retraining decision flow

```
                 ┌─────────────────────────────┐
                 │      should_retrain()        │
                 │  (FR-026 -- ANY of:)          │
                 │  - performance degraded       │
                 │  - significant (DRIFT) shift  │
                 │  - enough new labelled loans   │
                 └───────────────┬───────────────┘
                                 │ True
                                 v
                 ┌─────────────────────────────┐
                 │   trigger_retraining()        │
                 │   re-runs Phase 6 training     │
                 │   on the extended dataset       │
                 │   -> registers a NEW version,   │
                 │      inactive (never auto-     │
                 │      promoted)                  │
                 └───────────────┬───────────────┘
                                 v
                 ┌─────────────────────────────┐
                 │  Champion vs. challenger:      │
                 │  challenger PR-AUC >= champion  │
                 │  PR-AUC + margin, AND           │
                 │  |challenger calib. slope - 1|  │
                 │  <= tolerance?                   │
                 └───────┬─────────────┬─────────┘
                     yes │             │ no
                         v             v
              ┌────────────────┐  ┌─────────────────────┐
              │ activate_model  │  │ stays registered,     │
              │ (challenger)    │  │ inactive; WARNING       │
              │ + build its own │  │ alert asks for human    │
              │ baseline        │  │ review                  │
              └────────────────┘  └─────────────────────┘
```

`should_retrain`'s three triggers are evaluated independently (any one is sufficient) —
see `config/monitoring.yaml`'s `retraining` section for `min_new_labeled_loans`
(default 500), `promotion_pr_auc_margin` (default 0.01, absolute PR-AUC), and
`calibration_slope_tolerance` (default 0.15).

The champion-vs-challenger comparison always happens on the **challenger's own
untouched test split** — the champion is scored on it (not retrained), which is a fair
"how would each model do on this data" comparison since the champion was trained
before that data existed. The comparison is logged both to MLflow (a
`champion_vs_challenger` run) and to `reports/models/champion_vs_challenger_*.md`.

`rollback_to_version(model_id)` reactivates any previous version at any time — no
artifact is ever touched (CLAUDE.md hard rule 6: a trained model is never overwritten),
so every prior `.joblib` file stays loadable forever.
