# Feature dictionary (Phase 4)

One row per feature produced by `creditguard.features.pipeline.build_feature_pipeline`,
at the *logical* level -- i.e. before `OneHotEncoder`/`OrdinalEncoder` expand a
categorical/ordinal column into several numeric columns. The exact expanded
column names (e.g. `gender_MALE`, `gender_FEMALE`) are written to
`features_metadata.json` and `models/artifacts/feature_pipeline_<ver>.json`
by every `creditguard.features.build` run, since one-hot width depends on
which categories clear `min_frequency` in the training split.

"Direction" is the expected sign of the feature's association with 12-month
default risk (`default_12m`), per the label's own design in
[`docs/data_generation.md`](data_generation.md#the-label) where known;
"mixed/context" where the generator doesn't encode a direct effect.

Columns not listed here (`loan_id`, `customer_id`, `application_date`,
`financial_as_of_date`, `credit_as_of_date`, `decision_date`, `status`) are
identifiers or dates carried through the merge for joining/auditing/splitting
and are never selected into the model matrix by `ColumnTransformer`
(`remainder="drop"`), or are excluded from the frame entirely
(`decision_date`, `status` -- see `creditguard.features.leakage`).

## Demographic and employment (source: `customers`)

| Feature | Definition | Source column(s) | Business meaning | Direction | Range |
|---|---|---|---|---|---|
| `age` | Applicant age in years | `customers.age` | Older applicants tend to have more stable finances | − (weak, via `employment_years`/history) | 18-100 |
| `dependents` | Number of financial dependents | `customers.dependents` | More dependents -> more competing claims on income | + (weak) | 0+ |
| `employment_years` | Years in current employment | `customers.employment_years` | Longer tenure signals stability | − | 0+ |
| `annual_income` | Gross annual income | `customers.annual_income` | Higher income -> more capacity to service debt | − (via `log_monthly_income` in the label) | >= 0 |
| `city_tier` | 1 (metro) / 2 / 3 (smaller city) | `customers.city_tier` | Proxy for cost of living / income opportunity | mixed/context | 1-3 |
| `gender` | MALE / FEMALE / OTHER | `customers.gender` | Demographic; not used in the label design | none by design | categorical |
| `marital_status` | SINGLE / MARRIED / DIVORCED / WIDOWED | `customers.marital_status` | Household structure proxy | mixed/context | categorical |
| `education` | HIGH_SCHOOL / GRADUATE / POSTGRADUATE / DOCTORATE | `customers.education` | Correlates with income (via generator), not used directly in the label | mixed/context | categorical |
| `employment_type` | SALARIED / SELF_EMPLOYED / BUSINESS_OWNER / UNEMPLOYED | `customers.employment_type` | Income stability proxy | mixed/context | categorical |

## Loan terms (source: `loan_applications`)

| Feature | Definition | Source column(s) | Business meaning | Direction | Range |
|---|---|---|---|---|---|
| `loan_amount` | Requested principal | `loan_applications.loan_amount` | Larger loans relative to income raise risk (see `loan_to_income`) | + (via `loan_to_income` in the label) | > 0 |
| `loan_tenure_months` | Repayment term | `loan_applications.loan_tenure_months` | Longer tenure lowers EMI but extends exposure | mixed/context | > 0 |
| `interest_rate` | Annual interest rate, % | `loan_applications.interest_rate` | Bureau-risk-priced; a symptom of risk more than a cause | + (correlated with `risk_propensity` at generation time, not itself in the label) | 1-40 |
| `loan_type` | PERSONAL / HOME / AUTO / EDUCATION / BUSINESS / CREDIT_CARD | `loan_applications.loan_type` | Product-type risk profile | mixed/context | categorical |
| `loan_purpose` | e.g. DEBT_CONSOLIDATION, MEDICAL, WEDDING | `loan_applications.loan_purpose` | Stated use of funds | mixed/context | categorical |

## Financial snapshot, point-in-time (source: `financial_profiles`, latest row with `as_of_date <= application_date`)

| Feature | Definition | Source column(s) | Business meaning | Direction | Range |
|---|---|---|---|---|---|
| `monthly_income` | Net monthly income at application time | `financial_profiles.monthly_income` | Debt-servicing capacity | − (via `dti`, `emi_to_income` etc.) | >= 0 |
| `monthly_expenses` | Recurring monthly expenses | `financial_profiles.monthly_expenses` | Reduces disposable income | + (via `dti`) | >= 0 |
| `existing_loan_count` | Number of existing loans | `financial_profiles.existing_loan_count` | Existing debt burden | + | 0+ |
| `existing_loan_amount` | Outstanding balance on existing loans | `financial_profiles.existing_loan_amount` | Existing debt burden | + | >= 0 |
| `monthly_emi` | Current monthly EMI obligation | `financial_profiles.monthly_emi` | Debt-servicing load already committed | + (via `dti`, `emi_to_income`) | >= 0 |
| `savings_balance` | Liquid savings | `financial_profiles.savings_balance` | Buffer against income shocks | − (via `savings_to_income`) | >= 0 |
| `total_assets` | Total assets | `financial_profiles.total_assets` | Overall financial cushion | − (weak) | >= 0 |
| `total_liabilities` | Total liabilities | `financial_profiles.total_liabilities` | Overall debt load | + (weak) | >= 0 |

## Credit bureau snapshot, point-in-time (source: `credit_history`, latest row with `as_of_date <= application_date`)

| Feature | Definition | Source column(s) | Business meaning | Direction | Range |
|---|---|---|---|---|---|
| `credit_history_months` | Length of bureau history | `credit_history.credit_history_months` | Longer history -> more predictable behaviour | − | >= 0 |
| `num_credit_accounts` | Number of bureau-tracked accounts | `credit_history.num_credit_accounts` | Credit experience / exposure | mixed/context | >= 0 |
| `total_credit_limit` | Aggregate credit limit | `credit_history.total_credit_limit` | Denominator of `credit_utilization` | mixed/context (denominator) | >= 0 |
| `total_outstanding` | Aggregate outstanding balance | `credit_history.total_outstanding` | Numerator of `credit_utilization` | + | >= 0 |
| `previous_defaults` | Count of prior defaults | `credit_history.previous_defaults` | Strongest single bureau risk signal | + | 0+ |
| `late_payments_12m` | Late payments, trailing 12 months | `credit_history.late_payments_12m` | Recent repayment behaviour | + | 0+ |
| `missed_payments_12m` | Missed payments, trailing 12 months | `credit_history.missed_payments_12m` | Recent repayment behaviour | + | 0+ |
| `active_loans` | Currently active loan accounts | `credit_history.active_loans` | Current exposure | mixed/context | 0+ |
| `closed_loans` | Closed/paid-off loan accounts | `credit_history.closed_loans` | Track record of completed repayment | − (weak) | 0+ |

## Ratio features (`creditguard.features.ratios`)

Every division is guarded by `safe_divide`: a non-positive denominator maps
to the ratio's documented cap (its worst-case sentinel) instead of inf/NaN,
and every result -- including ordinary in-range divisions -- is clipped into
`[floor, cap]`.

| Feature | Formula | Business meaning | Direction | Range (cap) |
|---|---|---|---|---|
| `dti` | `(monthly_emi + monthly_expenses) / monthly_income` | Total debt-to-income burden | + | [0, 5.0] |
| `emi_to_income` | `monthly_emi / monthly_income` | Existing EMI burden alone | + | [0, 2.0] |
| `credit_utilization` | `total_outstanding / total_credit_limit` | Bureau utilization, recomputed with a safe-divide guard rather than trusted as a raw bureau field | + | [0, 2.0] |
| `loan_to_income` | `loan_amount / annual_income` | Requested loan size relative to income | + | [0, 10.0] |
| `proposed_emi` | Standard amortisation: `P*r*(1+r)^n / ((1+r)^n - 1)`, `r = annual_rate/12/100` (analytic `P/n` limit at `r=0`) | EMI the *new* loan would add | (feeds `post_loan_dti`) | >= 0, uncapped (bounded by `loan_amount`/`loan_tenure_months` themselves) |
| `post_loan_dti` | `(monthly_emi + proposed_emi + monthly_expenses) / monthly_income` | Projected DTI *if this loan is approved* -- the single most direct affordability check | + | [0, 5.0] |
| `savings_to_income` | `savings_balance / monthly_income` | Months of income held in savings | − | [0, 50.0] |
| `net_worth` | `total_assets - total_liabilities` | Overall financial cushion | − | can be negative, uncapped |
| `leverage_ratio` | `total_liabilities / max(total_assets, 1)` | Debt relative to assets | + | >= 0, uncapped |
| `disposable_income` | `monthly_income - monthly_expenses - monthly_emi` | Cash left after fixed obligations | − | can be negative, uncapped |
| `months_of_runway` | `savings_balance / max(monthly_expenses, 1)` | Months the applicant could cover expenses from savings alone if income stopped | − | >= 0, uncapped |

## Behavioural features (`creditguard.features.behavioural`)

| Feature | Formula | Business meaning | Direction | Range |
|---|---|---|---|---|
| `delinquency_rate` | `(late_payments_12m + missed_payments_12m) / max(num_credit_accounts, 1)` | Delinquency incidents per account held | + | >= 0 |
| `has_prior_default` | `previous_defaults > 0` (0/1) | Any default history at all | + | {0, 1} |
| `credit_history_years` | `credit_history_months / 12` | Bureau history in years | − | >= 0 |
| `accounts_per_year` | `num_credit_accounts / max(credit_history_years, 0.5)` | Account-opening pace | mixed/context (very high can signal credit-seeking stress) | >= 0 |
| `active_loan_ratio` | `active_loans / max(active_loans + closed_loans, 1)` | Share of credit history still open | mixed/context | [0, 1] |
| `employment_stability` | `employment_years / max(age - 18, 1)` | Fraction of working-age life in current job | − | [0, 1] typically |
| `income_per_dependent` | `monthly_income / (dependents + 1)` | Income adjusted for household size | − | >= 0 |

## Band features (fixed or quantile-binned, ordinal-encoded)

`utilization_band` uses fixed business cutoffs. `age_band`, `tenure_band` and
`income_band` are quantile bins (5 by default, `Q1`..`Q5`) whose edges are
learned from **training data only**
(`BehaviouralFeatures.fit`) -- see `docs/data_quality.md`-style leakage
notes in `creditguard/features/leakage.py` for why this matters. Values
outside the observed training range still bin into the extreme bucket
(`-inf`/`+inf` edges), never `NaN`.

| Feature | Bins | Business meaning | Direction |
|---|---|---|---|
| `utilization_band` | `0-30` / `30-50` / `50-70` / `70-90` / `90+` (fixed) | Coarse utilization risk tier | higher band -> higher risk |
| `age_band` | `Q1`(youngest) .. `Q5`(oldest), quantiles of `age` on train | Coarse age tier | mixed/context |
| `tenure_band` | `Q1`(shortest) .. `Q5`(longest), quantiles of `employment_years` on train | Coarse employment-stability tier | `Q1` -> higher risk |
| `income_band` | `Q1`(lowest) .. `Q5`(highest), quantiles of `annual_income` on train | Coarse income tier | `Q1` -> higher risk |

## Excluded by design (target/post-decision leakage)

Never selected into the feature frame at all -- either the point-in-time
join simply doesn't select them (`decision_date`, `status`,
`loan_outcomes.*`), or `creditguard.features.leakage.assert_no_leakage`
raises if they ever do reach feature code:

`default_12m`, `outcome_observed_date`, `decision_date`, `status`,
`days_past_due`, `recovery_amount`, `collection_status`, `write_off_amount`,
`future_missed_payments`, and anything matching `^future_`,
`_after_decision$`, `^post_disbursement_`, `^outcome_`, `^actual_`,
`^repayment_`, `^charge_off_`.
