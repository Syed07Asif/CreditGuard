# Scoring methodology (Phase 7)

**This describes a simulated, educational credit-scoring system built on
synthetic data.** The scaling method (points-to-double-odds) is a standard,
widely used technique in real credit scoring, but the specific numbers
below -- the base score, the risk-category bands, the recommendation
thresholds -- are **this project's own configuration choices for this
simulation, not a universal banking standard, a regulatory requirement, or
a real institution's policy.** They live in
[`config/scoring.yaml`](../config/scoring.yaml) and can be changed there;
nothing in `src/creditguard/scoring/` or `src/creditguard/explain/`
hard-codes them (FR-012).

## Why a score at all?

`src/creditguard/models/` (Phase 6) produces a **calibrated probability of
default within 12 months** -- a number between 0 and 1. That's the right
input for further computation, but not what a loan officer wants to read
at a glance. The scorecard transform below turns that probability into a
**credit score** on a fixed, familiar 300-900 scale where **higher score
means lower risk** -- the same convention real bureau scores use, so the
direction is never surprising to read.

## The formula: points-to-double-odds

```
odds   = (1 - p_default) / p_default        # odds of NOT defaulting
score  = OFFSET + FACTOR * ln(odds)
FACTOR = PDO / ln(2)
OFFSET = BASE_SCORE - FACTOR * ln(BASE_ODDS)
```

`p_default` is the calibrated probability from Phase 6.
`BASE_SCORE`, `BASE_ODDS` and `PDO` are the three tunable inputs
(`config/scoring.yaml`'s `scorecard` section); this project's defaults:

| Parameter | Value | Meaning |
|---|---|---|
| `BASE_SCORE` | 600 | The score awarded when the odds of *not* defaulting equal `BASE_ODDS` |
| `BASE_ODDS` | 20.0 | 20:1 odds of not defaulting (a 1-in-21, ≈4.76%, default probability) at `BASE_SCORE` |
| `PDO` | 40.0 | "Points to Double the Odds" -- see below |
| `min_score` / `max_score` | 300 / 900 | The result is clipped to this range and rounded to an integer |

The result is implemented in
[`scoring/scorecard.py`](../src/creditguard/scoring/scorecard.py)'s
`probability_to_score`, which clips `p_default` away from the exact `[0, 1]`
boundary (`PROBABILITY_EPSILON = 1e-9`) before the log transform -- `ln(0)`
and `ln(inf)` are undefined, and no real calibrated probability should ever
land exactly on either boundary anyway.

### What "PDO" means

**PDO (Points to Double the Odds)** is the number of score points added
whenever the odds of *not* defaulting double -- e.g. odds going from 10:1
to 20:1 (default probability roughly halving, from ≈9.1% to ≈4.8%). It's
the scorecard's "sensitivity" dial: a smaller PDO spreads a given range of
probabilities across more score points (a more sensitive, more spread-out
scale); a larger PDO compresses the same range into fewer points. This
project uses **PDO = 40**: every time an applicant's odds of not
defaulting double, their score rises by exactly 40 points, by construction
of the formula (`FACTOR = PDO / ln(2)`, so a doubling of odds means `ln(2)`
more in the log term, which the formula turns into exactly `PDO` points).
`tests/test_scorecard.py::test_doubling_odds_is_worth_exactly_pdo_points`
checks this holds.

### Worked example

Take a calibrated default probability of **p = 0.087** (8.7%):

| Step | Computation | Result |
|---|---|---|
| Odds of not defaulting | `(1 - 0.087) / 0.087` | **10.494253** |
| `ln(odds)` | `ln(10.494253)` | **2.350828** |
| `FACTOR` | `40 / ln(2)` | **57.707802** |
| `OFFSET` | `600 - 57.707802 * ln(20)` | **427.122876** |
| Raw score | `427.122876 + 57.707802 * 2.350828` | **562.783978** |
| Final score | `round(clip(562.783978, 300, 900))` | **563** |

So an 8.7% default probability maps to a credit score of **563** -- solidly
in the `HIGH` risk band (550-649; see below). This exact example is
asserted byte-for-byte in
`tests/test_scorecard.py::test_hand_computed_worked_example`.

### Probability-to-score lookup table

Computed the same way, at `BASE_SCORE=600, BASE_ODDS=20, PDO=40`:

| `p_default` | Credit score |
|---|---|
| 0.01 | 692 |
| 0.02 | 652 |
| 0.05 | 597 |
| 0.10 | 554 |
| 0.20 | 507 |
| 0.30 | 476 |
| 0.50 | 427 |

Reproduced exactly in
`tests/test_scorecard.py::test_hand_computed_lookup_table`.

### The inverse: `score_to_probability`

`scorecard.score_to_probability` inverts the transform algebraically:

```
odds  = exp((score - OFFSET) / FACTOR)
p     = 1 / (1 + odds)
```

Round-tripping `score_to_probability(probability_to_score(p))` recovers `p`
within a small relative tolerance (the only error source is
`probability_to_score` rounding to the nearest *integer* score --
`tests/test_scorecard.py::test_round_trip_within_tolerance`). At the
extreme probabilities `1e-9` and `1 - 1e-9`, the score clips to `900` and
`300` respectively rather than diverging, and `score_to_probability`
applied to those clipped scores recovers the probability *at the clip
boundary*, not the original extreme input -- clipping is lossy by design,
not a bug (`tests/test_scorecard.py::test_clipping_at_extreme_probabilities`).

## Risk categories

**These bands are this project's own rules for this simulation, chosen to
line up with the recommendation policy below -- not an industry standard.**
Defined in `config/scoring.yaml`'s `risk_categories` section and loaded by
[`scoring/categories.py`](../src/creditguard/scoring/categories.py):

| Band | Score range |
|---|---|
| `VERY_LOW` | 750 - 900 |
| `LOW` | 700 - 749 |
| `MODERATE` | 650 - 699 |
| `HIGH` | 550 - 649 |
| `VERY_HIGH` | 300 - 549 |

`categories.load_risk_bands` validates at load time that the configured
bands are contiguous, non-overlapping, and cover `[300, 900]` exactly --
misconfiguring `config/scoring.yaml` (a gap, an overlap, or a range that
doesn't start/end at the scorecard's own min/max) raises `ScoringConfigError`
rather than silently mis-categorising applicants.

## Recommendation policy

[`scoring/recommendation.py`](../src/creditguard/scoring/recommendation.py)
combines the calibrated probability, the credit score, and a handful of
policy rules -- every threshold configurable in `config/scoring.yaml`'s
`recommendation` section, never hard-coded:

- **APPROVE** -- probability below the approve threshold (the *active
  model's own* Phase 6 cost-optimal `chosen_threshold`, read from
  `model_registry`, not a second independently chosen number) **and**
  score ≥ 700 **and** no hard fail.
- **REJECT** -- probability above the reject threshold (a configurable
  multiple of the approve threshold), **or** score ≤ 549, **or** any hard
  fail (`previous_defaults ≥ 2`; `post_loan_dti > 0.60`;
  `credit_utilization > 0.95`; thin-file + high-leverage:
  `employment_years < 0.5` **and** `loan_to_income > 5.0`).
- **REVIEW** -- everything else: the genuine middle band, or any soft
  policy flag (thin credit file `< 12` months; `post_loan_dti` in
  `[0.45, 0.60]`; disposable income after the proposed EMI below a
  configured floor) even when probability/score would otherwise clear the
  approve bar.

Every `Recommendation` carries the exact rule(s) that fired
(`triggered_rules`) -- a decision is always traceable, never a bare label.

## Explainability: what the SHAP numbers mean

[`explain/shap_explainer.py`](../src/creditguard/explain/shap_explainer.py)
explains the **base (pre-calibration) model's decision function** -- not
the final calibrated probability directly. Phase 6's calibration step
(`CalibratedClassifierCV`, isotonic or sigmoid) has no closed-form
coefficients for SHAP to decompose, especially for isotonic calibration.
Calibration is a **monotonic transform** of the base model's output, so a
feature's *direction* (did it push risk up or down for this applicant?)
and *relative importance* survive calibration unchanged, even though the
raw SHAP numbers are in log-odds space rather than probability space. In
practical terms: the ranked list of risk-increasing/risk-reducing factors
you see in a `ScoringResult` is trustworthy exactly as ranked; only the
raw magnitude of each SHAP number isn't itself a probability.

[`explain/reason_codes.py`](../src/creditguard/explain/reason_codes.py)
turns each attributed feature into a sentence with two independently
computed parts: a **factual clause** (the applicant's actual value against
the training-data portfolio median/mode -- always true, regardless of SHAP
sign) and a **direction clause** (this application's actual SHAP sign --
always accurate about which way this factor pushed *this* application). A
reason sentence is never sourced from a global assumption like "higher DTI
always means higher risk"; both halves are computed per application.

## Summary of what "one point" and "one band" mean here

- **40 points = one doubling of the odds of not defaulting** (by
  construction of the PDO transform).
- **A risk band boundary is a policy choice, not a statistical cliff** --
  749 and 750 are one point apart on a continuous risk scale; the
  `LOW`/`VERY_LOW` label attached to each is this project's own cut point.
- **A probability threshold in the recommendation policy is tied to the
  active model's actual cost-optimal operating point** (Phase 6), so it
  moves automatically if a future model promotion changes that operating
  point -- it is not a magic number re-derived by hand each time.
