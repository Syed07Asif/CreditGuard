# CreditGuard Model Card

**This is a simulated, educational credit-risk system, not a production
lending model.** All customer data is synthetic. Nothing in this document
validates the model for real lending decisions, and no output of this model
should be used to make, or substitute for, a real credit decision about a
real person. A human underwriter must review and remain accountable for any
decision this model informs — see "Human oversight" below.

## Intended use

Estimates the probability that a loan applicant defaults within 12 months of
disbursement (`default_12m`), at the point the application is submitted
(application-date features only — see `docs/feature_dictionary.md` and
`src/creditguard/features/leakage.py` for the point-in-time discipline this
is built on). The calibrated probability this model outputs is the direct
input to Phase 7's credit score and lending-recommendation conversion — a
miscalibrated probability produces a wrong score even when the model's
*ranking* of applicants is fine, which is why calibration quality (below) is
reported as a first-class result, not an afterthought.

Not intended for: any purpose beyond ranking/scoring 12-month default risk
for this synthetic population; any protected-characteristic-based decision;
any use without the human-review step Phase 7-8 are expected to keep in
place.

## Training data

- Dataset version `ds_20260808_547ecf5a_clean` (Phase 3 output), engineered
  by the Phase 4 pipeline, temporally split 70/15/15 (train/val/test) by
  `application_date` — see `reports/eda/findings.md` for the full population
  characterisation this model was built on.
- **96,749 loans total**: 67,724 train / 14,513 validation / 14,512 test.
  Overall default rate **11.10%** (10,738 of 96,749) — a moderate class
  imbalance, not an extreme one.
- Train: 2022-08-08 – 2024-09-14. Validation: 2024-09-15 – 2025-02-25. Test:
  2025-02-25 – 2025-08-08. The model has never seen the test window's loans
  or their outcomes at any stage of training, calibration, or threshold
  selection.

## Features used

77 features after one-hot/ordinal encoding (see `docs/feature_dictionary.md`
for the full logical-to-encoded mapping): demographic/employment fields,
loan terms, point-in-time financial and credit-bureau snapshots, 11
engineered ratio features (`dti`, `credit_utilization`, `loan_to_income`,
etc.), 7 behavioural features, and 4 quantile/fixed-cutoff band features.
Every feature is available at the loan's `application_date`; none carry
post-decision or outcome information (`assert_no_leakage` enforces this at
both fit and transform time, not only in tests).

## Model selection and search

Logistic regression (L2/elastic-net, `saga` solver), random forest, and
XGBoost were each tuned via `RandomizedSearchCV` (20 iterations,
`StratifiedKFold(5)`, scored on **average precision / PR-AUC — never
accuracy**, enforced in code by `evaluate.select_best_model` raising
`ValueError` if asked to select on accuracy). All three were searched under
the `class_weight` imbalance strategy (`class_weight="balanced"` /
XGBoost's `scale_pos_weight`), per `reports/eda/findings.md`'s "Decisions
for Phase 6" recommendation to prefer weighting over resampling.

**Leaderboard (validation split):**

| Model | PR-AUC | ROC-AUC | Brier | CV score |
|---|---|---|---|---|
| **logistic_regression (selected)** | **0.5464** | **0.8739** | 0.1393 | 0.5582 |
| xgboost | 0.5405 | 0.8723 | 0.1381 | 0.5465 |
| random_forest | 0.5182 | 0.8639 | 0.1074 | 0.5171 |

Logistic regression won on PR-AUC (the search's selection metric) despite
being the simplest of the three families — the interpretable baseline was
kept in the comparison per the Phase 6 brief specifically for this
possibility, and here it wasn't just competitive, it was best. Winning
hyperparameters: `C=0.01`, `penalty=elasticnet`, `l1_ratio=0.7`,
`class_weight=balanced`.

## Imbalance strategy comparison

Full comparison in `reports/models/imbalance_comparison.md` (every model
family x `none`/`class_weight`/`smote`/`random_undersampling`/
`smote_tomek`). **Honest finding, stated plainly because it differs from
the textbook expectation:** on this dataset, resampling and class-weighting
gave only a marginal recall improvement over doing nothing (logistic
regression's recall@precision≥0.5 moved from 0.554 with no adjustment to
0.559 with `class_weight` and 0.563 with SMOTE — a 0.5-0.9 point gain), 
while noticeably worsening raw Brier score (0.068 → 0.139 with
`class_weight`, 0.138 with SMOTE) and, for SMOTE specifically, calibration
slope (1.040 → 0.937). This dataset's 11.1% default rate is a moderate
imbalance rather than the severe (<1%) case where synthetic oversampling
typically earns back its calibration cost — consistent with, and now
directly confirmed by, the EDA's Phase 6 recommendation to prefer
`class_weight` over resampling. The raw calibration damage from
`class_weight` shown in this table is exactly why calibration is a
mandatory separate downstream step (next section), not an optional
nice-to-have: the registered model's actual calibration is excellent, but
only because it was explicitly recalibrated after training, not because the
training-time strategy happened to preserve calibration.

## Calibration

`CalibratedClassifierCV` (via `sklearn.frozen.FrozenEstimator`, fit on the
validation split — never the training split) was tried with both isotonic
and sigmoid methods; **isotonic** won by validation-set Brier score.

| | Before calibration | After calibration |
|---|---|---|
| Brier score | 0.1393 | **0.0675** |
| Calibration slope | 1.010 | **1.000** |
| Calibration intercept | — | 0.0545 |

Brier score roughly halved and calibration slope moved from already-decent
(1.010) to essentially exact (1.000) — well inside the ±0.1 tolerance the
Phase 6 acceptance criteria require. The reliability diagram is at
`reports/figures/models/reliability_diagram.png`. **The calibrated
estimator, not the raw logistic regression, is what's registered and would
be served.**

## Threshold

Not defaulted to 0.5. A cost matrix (false-negative cost 10x false-positive
cost — approving a defaulter is treated as far more expensive than
rejecting a good applicant) drives the chosen operating point:

| Criterion | Threshold | Detail |
|---|---|---|
| **Chosen (min expected cost)** | **0.0840** | expected cost 5,955 (test-set units) |
| Max F1 | 0.2941 | F1 = 0.5312 |
| Max Youden's J | 0.1103 | J = 0.5832 |

The chosen threshold (0.084) is well below both alternatives, which is the
direct, intended consequence of weighting false negatives 10x more heavily
than false positives — the model is deliberately tuned to flag more
applicants as high-risk than a precision/recall- or ROC-optimal threshold
would, because missing an actual defaulter is treated as far costlier than
an unnecessary decline.

## Test-set performance

Untouched test split (2025-02-25 – 2025-08-08), calibrated model, chosen
threshold 0.084:

| Metric | Value |
|---|---|
| ROC-AUC | **0.8770** (≥ 0.75 required) |
| PR-AUC / average precision | **0.5450** (vs. 11.10% base rate — ~4.9x lift) |
| Gini | 0.7541 |
| KS statistic | 0.5916 |
| Brier score | 0.0682 |
| Log loss | 0.2329 |
| Calibration slope | **0.9997** (0.9-1.1 required) |
| Accuracy | 0.7566 |
| Precision | 0.2876 |
| Recall | 0.8332 |
| F1 | 0.4276 |

Accuracy alone would be a misleading headline number here — a model that
never flags anyone as high-risk would score ~89% accuracy on this default
rate while being useless (`FORBIDDEN_SELECTION_METRICS` in `evaluate.py`
exists for exactly this reason). At the cost-minimising threshold, the
model deliberately trades precision for recall: it catches 83.3% of actual
defaulters, at the cost of a 28.8% precision (roughly 7 in 10 applicants it
flags as high-risk do not actually default) — the direct, intended
consequence of the 10:1 false-negative cost weighting above, not a defect.

**Lift/gains** (`reports/models/test_lift_gains.csv`, full table): the top
decile of predicted risk (10% of applicants) captures **49.8%** of all
actual test-set defaults (lift 4.98x); the top 3 deciles (30% of
applicants) capture **82.0%** of defaults. This is the shape a usable risk
ranking should have — heavily front-loaded, not close to the 10%-per-decile
line a non-discriminating model would produce.

## Per-segment performance

Full tables: `reports/models/test_segment_metrics_{loan_type,income_band,age_band}.csv`.

- **By loan type**: ROC-AUC is consistent (0.839-0.853) across
  PERSONAL/AUTO/EDUCATION/BUSINESS/CREDIT_CARD, but **HOME loans stand out
  sharply** — PR-AUC 0.737 (vs. 0.27-0.46 elsewhere) and recall 0.969 at the
  chosen threshold. This mirrors `reports/eda/findings.md` finding 4 (HOME
  loans default at 32% vs. 5.5-11.3% for other types, flagged there as a
  leakage-recheck candidate since `loan_type` isn't documented as a direct
  label input). **The model is visibly exploiting whatever mechanism drives
  that HOME-loan effect.** Before this model is trusted for HOME-loan
  decisions specifically, the underlying generator mechanism should be
  traced and confirmed non-leaky, not just assumed benign because overall
  metrics look strong.
- **By income band**: recall is highest for `Q1` (lowest income, 0.896) and
  falls monotonically to `Q5` (highest income, 0.724) — the model is more
  sensitive at exactly the segment `reports/eda/findings.md` found carries
  the most real risk (finding 3/9: income and credit-utilization risk
  curves), not a segment where it's arbitrarily under- or over-flagging.
- **By age band**: the same pattern — recall 0.912 for `Q1` (youngest)
  falling to 0.648 for `Q5` (oldest), consistent with the age-risk gradient
  EDA finding 8 documented (21.3% → 3.1% default rate across age deciles).
  No segment's ROC-AUC drops below 0.84 — the model isn't *failing* on any
  age group, it's correctly less aggressive where risk is genuinely lower.

No segment examined shows a *materially worse* ROC-AUC than the overall
0.877 (all segments land between 0.84-0.91) — the HOME-loan and
low-income/young-age effects above are differences in operating-point
behaviour (recall/precision at a fixed threshold) driven by real
between-segment default-rate differences, not evidence the model is broken
for any particular group.

## Cross-validated stability (train split)

5-fold `StratifiedKFold` on the winning logistic regression configuration
(mean ± std across folds, `reports/models/train_cv_metrics.csv`) confirms
performance is stable across resampled training folds, not an artifact of
one lucky train/val/test split.

## Known limitations

- **Synthetic data.** Every number in this document describes a generated
  population, not real applicants or real bureau data. Nothing here
  transfers to a real portfolio without re-validation on real data.
- **HOME-loan effect not yet explained.** See "Per-segment performance"
  above — this is the single most important open item before Phase 7/8
  treat this model's HOME-loan predictions as trustworthy.
- **Precision is low at the chosen operating point** (28.8%) — by design,
  given the cost matrix, but it means roughly 7 of 10 applications the
  model flags as high-risk would not actually have defaulted. Any
  downstream UI/process (Phase 7-9) must present this as a risk *signal*
  requiring human judgement, not a verdict.
- **No feature changes were made in this phase** — anything EDA flagged as
  a candidate new feature (Decisions for Phase 6) was deliberately left for
  a future, dedicated Phase 4 revision, not folded in ad hoc here.
- **Multicollinearity was not hand-pruned** for the winning model — the
  elastic-net penalty (`l1_ratio=0.7`) regularises the 19 correlated pairs
  `reports/eda/findings.md` identified rather than any of them being
  manually dropped; the near-zero gender/marital_status coefficients below
  suggest this worked as intended, but it wasn't independently re-verified
  pair by pair.

## Fairness caveat: gender and marital status

**`gender` and `marital_status` are present in the training data and are
not excluded from the feature matrix** — `docs/feature_dictionary.md`
documents `gender` as having "no [direct] effect by design" in the label
generator, and `reports/eda/findings.md` finding 9 measured their
standalone predictive power as near-zero (Information Value 0.0004 for
gender, 0.0001 for marital_status — both in the "useless" band). Checking
the actual fitted model confirms this held after training: the elastic-net
logistic regression's coefficients for every `gender_*`/`marital_status_*`
one-hot column are effectively zero (0.000-0.010), against a maximum
feature coefficient magnitude of 0.83 and a mean of 0.064 across all 77
features — the model learned to essentially ignore them, not because they
were removed, but because the regularised fit found no signal worth
keeping. This is a reassuring result, not a guarantee: a near-zero direct
coefficient does not rule out a smaller indirect proxy effect through
correlated features, and this was not separately tested (e.g. via a formal
disparate-impact analysis by gender/marital-status group). Before any real
deployment, this would need explicit fairness auditing beyond what this
phase performed.

## Human oversight

This model's output is a probability and a threshold-derived flag, not a
decision. It is one input meant to inform a human underwriter's judgement
(via Phase 7's score/recommendation and Phase 8's API), not to replace it.
Given this is a simulated system built on synthetic data with an unresolved
HOME-loan question and no formal fairness audit, no output of this model or
any downstream phase should be treated as a real lending decision without
human review.
