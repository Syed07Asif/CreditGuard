# Session memory

Working notes carried over between Claude Code sessions on this repo. Not part of
the project's build output — read this first when resuming work here, alongside
`CLAUDE.md` (which is the actual project contract: hard rules, stack, phase
checklist).

---

## Project status (as of 2026-08-08)

- **Phase 1** (Foundation, config, database) — done, tests passing, **committed**
  (`0bbdfcc`, "Phase 1: Foundation, configuration and database").
- **Phase 2** (Synthetic data generation and ingestion) — done, all 23 tests
  passing, ruff/black clean, verified end-to-end at full 50k-customer scale
  (generation 2.8s, ingest ~40min, realised default rate 0.1105). **Committed**
  (`f923f94`, "Phase 2: Synthetic data generation and ingestion").
- **Phase 3** (Data validation and cleaning pipeline) — done, all 42 tests
  passing (up from 23; 19 new), ruff/black clean, verified end-to-end against
  the real 50k-customer Phase 2 dataset (`ds_20260808_547ecf5a`): 100%
  detection rate on all 6 injected error types (manifest asked for ≥95%), and
  the cleaned output re-validates with zero ERROR violations. **Committed**
  (`a46bcfc`, "Phase 3: Data validation and cleaning pipeline") — see
  "Validation & cleaning design" below for what it does and one real bug
  found/fixed along the way.
- **Phase 4** (Feature engineering and leakage prevention) — done, all 91
  tests passing (up from 42; 49 new), ruff/black clean, verified end-to-end
  against the real 50k-customer Phase 3 clean dataset
  (`ds_20260808_547ecf5a_clean`): 77 output features, X_train/X_val/X_test =
  (67724,77)/(14513,77)/(14512,77), all finite, zero target-correlation
  offenders, temporal split exactly 70/15/15 with non-overlapping date ranges
  (train 2022-08-08..2024-09-14, val 2024-09-15..2025-02-25, test
  2025-02-25..2025-08-08). **Committed** (`0f85195`, "Phase 4: Feature
  engineering and leakage prevention"). See "Feature engineering design"
  below for the architecture and one real false-positive bug caught by
  testing.
- **Phase 5** (Exploratory data analysis) — done, all 108 tests passing (up
  from 91; 17 new), ruff/black clean, verified end-to-end against the real
  96,749-loan Phase 4 population (`ds_20260808_547ecf5a_clean`,
  train+val+test combined): 73 figures under `reports/figures/eda/` (≥20
  required), every summary table under `reports/eda/tables/`, notebook
  executes top to bottom with zero errors, `reports/eda/findings.md` has 12
  evidenced findings plus the "Decisions for Phase 6" section. **Committed**
  (`804c09f`, "Phase 5: Exploratory data analysis"). See "EDA design" below
  for a real IV-computation bug caught and fixed during this phase (not
  before it).
- **Phase 6** (Model training, imbalance handling and evaluation) — done,
  all 143 tests passing (up from 108; 35 new), ruff/black clean, verified
  end-to-end at real 96,749-loan scale. Leaderboard: logistic_regression
  (PR-AUC 0.5464) beat xgboost (0.5405) and random_forest (0.5182) on the
  validation split — the interpretable baseline won outright, not just
  stayed competitive. Test-set (14,512 loans, calibrated model, threshold
  0.084): ROC-AUC 0.8770 (≥0.75 required), PR-AUC 0.5450 (~4.9x the 11.10%
  base rate), calibration slope 0.9997 (0.9-1.1 required). Registered as
  `logistic_regression` v1.0.0, exactly one `is_active=true` row. **Not yet
  committed** as of this note — check `git status`/`git log` before assuming
  otherwise. See "Model training design" below for two real bugs caught and
  fixed during this phase (an XGBoost/CUDA multiprocessing race, and a
  fresh-Postgres-volume test-database gap), and for the honest imbalance
  comparison result (class-weighting/resampling barely helped recall here
  while visibly hurting raw calibration). **Committed** (`7bd08c6`, "Phase
  6: Model training, imbalance handling and evaluation") — the "not yet
  committed" note above was accurate when first written, not anymore.
- **Phase 7** (Credit scoring engine, risk categories and explainability) —
  done, 260 tests total (117 new), ruff/black clean. `src/creditguard/
  scoring/` (scorecard/categories/recommendation/engine) +
  `src/creditguard/explain/` (shap_explainer/reason_codes) +
  `config/scoring.yaml` + `docs/scoring_methodology.md`. Verified against
  the real registered Phase 6 model: a low-risk fixture scored 900/
  VERY_LOW/APPROVE and a high-risk fixture scored 300/VERY_HIGH/REJECT
  (both p clipped to 0.0/1.0 by isotonic calibration on these
  deliberately extreme synthetic fixtures), both under 500ms per-request
  latency (79ms/31ms; the one-time model-load-and-pipeline-refit cost,
  ~10-25s, is deliberately excluded from `ScoringResult.latency_ms` --
  see "Scoring & explainability design" below). **Committed** (`9c6c820`,
  "Phase 7: Credit scoring engine, risk categories and explainability").
- **Phase 8** (FastAPI real-time scoring service) — done, 305 tests total
  (44 new), ruff/black clean. `src/creditguard/api/` (main/schemas/
  dependencies/middleware/errors/routes/{predict,applications,model,
  health}) + `docker/Dockerfile.api` + `docs/api.md`. Every endpoint from
  the brief implemented; verified end to end against the real registered
  Phase 6/7 model (not just the synthetic fixture the automated tests use)
  -- see "API design" below for the p95 latency number and the two real
  gaps the brief's literal field list had (`city_tier`, `customer_id`/
  `loan_id`) that were filled in rather than silently worked around.
  Docker image built and smoke-tested directly (container starts, `/health`
  200 even with a failed model load, `/health/ready` correctly 503s without
  volumes mounted). **Committed** (`<pending>` -- check `git log`, this
  note was written before any commit was requested for this phase).
- **Phase 9** (Streamlit dashboard) — done, 335 tests total (30 new),
  ruff/black clean. `src/creditguard/dashboard/` (app/api_client/state +
  components/{forms,cards,charts,tables} + pages/1-4) + `.streamlit/
  config.toml` + `docker/Dockerfile.dashboard`. Required a small, additive
  extension to the already-committed Phase 8 API first (new `predictions`
  columns, a new `/model/performance` endpoint, `FactorDetail.
  benchmark_median`) -- see "Dashboard design (Phase 9)" below for why and
  exactly what changed. Verified live in a browser against the real
  registered model and real accumulated prediction traffic (108
  predictions), not just the automated suite: all four pages render, the
  high-risk sample scores 300/VERY_HIGH/REJECT end-to-end through the
  live API, Portfolio Analytics renders real KPIs/segments/charts with
  filters present, Model Performance shows the real backfilled metrics
  matching `reports/models/model_card.md` exactly. **Committed
  (`<pending>` -- check `git log`)**.
- **Next up:** Phase 10 (Monitoring, drift, retraining, Docker and CI), per
  `CLAUDE.md`'s checklist. Phases are strictly sequential — don't start
  Phase 10 work, and don't add stub files for it, until the user actually
  asks.

This is an educational/portfolio simulation (synthetic data only, not a real
lending system), built one phase at a time with the user reviewing each phase's
completion before moving on.

---

## Environment specifics (not visible from git alone)

- **Local Postgres port conflict:** a native Windows `postgres.exe` service
  already listens on port 5432, unrelated to this project. The
  `creditguard_postgres` container is remapped to host port **5433** via
  `DB_PORT=5433` in this repo's `.env` (git-ignored — `.env.example` still shows
  the standard 5432 default for machines without this conflict). If you see
  connection/auth failures against "postgres," check this first — `docker
  compose ps` and `netstat -ano | grep 5432` will show the conflict.
- **`docker-compose.yml`** sets `max_locks_per_transaction=256` on the postgres
  service (Postgres's default is 64) — required for Phase 2's bulk ingest at
  ~100k-row scale. See "Ingest design" below for why this alone wasn't enough.
- **Python 3.11** was missing from this machine when Phase 1 started (only 3.10
  was present); installed via `winget install --id Python.Python.3.11 --source
  winget` at `C:\Users\asiff\AppData\Local\Programs\Python\Python311`, and this
  repo's `.venv` was built from that interpreter. If `.venv` is ever recreated,
  use that same 3.11 interpreter — `CLAUDE.md` fixes the stack at Python 3.11.
- **pgAdmin 4** is installed on this machine
  (`C:\Users\asiff\AppData\Local\Programs\pgAdmin 4`). Connection details (same
  server, two databases): host `localhost`, port `5433`, user `creditguard`,
  password `changeme`, databases `creditguard` (app) and `creditguard_test`
  (pytest only).
- Docker Desktop shows two containers for this project: `creditguard_postgres`
  and `creditguard_mlflow` (mlflow on host port 5000, sqlite backend store).

---

## Ingest design: why per-batch commits, not one transaction

`src/creditguard/data/ingest.py` loads generated parquet data in batches of
`COMMIT_BATCH_SIZE = 2000` rows, each its own committed transaction (bad rows
within a batch are isolated via SAVEPOINT, not by aborting the batch) — **not**
one single transaction for the whole ~400k-row load, even though the original
Phase 2 spec asked for "a single transaction."

**Why:** tried literally as one transaction first. Postgres FK checks take a
row-level lock on the referenced parent row, held until commit. At ~400k rows
across 5 FK-linked tables this first hit `psycopg.errors.OutOfMemory: out of
shared memory / HINT: increase max_locks_per_transaction` (fixed by raising that
setting to 256), but even after that fix, one real run sat for 90+ minutes
without finishing — Postgres's lock manager degrades badly as a single
transaction's held-lock list grows into the tens of thousands, regardless of the
configured ceiling. This was only discovered by actually running ingest at full
50k-customer scale; smaller test-scale runs (2k customers) never hit it.

Also fixed in the same pass: `--truncate` wasn't clearing `data_quality_issues`,
so repeated truncated reloads accumulated duplicate quarantine-log rows; and a
naive "chunk-then-fallback-to-per-row" quarantine strategy degraded to one DB
round trip per row when bad rows are spread uniformly (near-certain at any
reasonable chunk size given 1–5% injection rates) — replaced with recursive
bisection (try bulk, split in half on failure, recurse), roughly O(bad_rows ×
log n) instead of O(n).

**Don't "fix" this back to a single transaction** to match the literal spec
wording — the deviation is deliberate and already documented in the module's own
docstring and in `docs/data_generation.md`. If a later phase needs to bulk-load
large FK-linked tables again, reuse this batched-commit + bisection pattern.

---

## Validation & cleaning design (Phase 3)

`src/creditguard/validation/` splits observation from repair on purpose:
`rules.py`/`engine.py` only report violations (never mutate data); `cleaning.py`
is the only module that transforms data, and every repair it makes is logged.

**Validates the raw parquet dataset, not the database.** Five of Phase 2's six
injected error types violate a DB CHECK/PK/NOT NULL constraint, so
`creditguard.data.ingest` already quarantines those rows *out of* the database
before Phase 3 ever runs. Only `inconsistent_income` survives ingest
unmodified. So `validate`/`clean` both take `--dataset-version` and read
`data/raw/<version>/` directly (like `generator`/`ingest` do) — validating
against the database would silently miss 5 of 6 error types and make the
95%-detection acceptance criterion unmeasurable.

**Severity decides repair vs. drop.** ERROR violations that clip/dedup can fix
(`ImpossibleAgeRule`, `NegativeFinancialRule`, `CreditUtilizationRule`,
`DuplicateRecordRule`) get repaired in `cleaning.py` and are never dropped.
What's left after repair (`OrphanRecordRule`, `TemporalLeakageRule`,
`InvalidDateRule`, `IncomeConsistencyRule` — none of which a clip can fix) gets
quarantined and dropped from the modelling set, cascaded to dependent tables
for referential integrity, but kept in `data_quality_issues`.

**Real bug found and fixed only by testing at real scale (again):** the
`DataCleaner.fit()` step originally computed winsorize percentile bounds and
imputation medians from the *raw* input (including exact-duplicate rows and
unclipped negative values). On a tiny hand-built test frame this visibly
corrupted a legitimate value (a repaired `0` income got winsorized back up,
which then broke income-consistency and got a customer wrongly dropped) — the
fix was to fit statistics on dedup+clip-repaired data instead of raw data
(still never touching `transform`-time data, so still "fit on train only").
Small-scale hand-built frames caught this because percentiles on n≈3-4 samples
are extremely sensitive to outliers; worth remembering for Phase 4+ if fitting
statistics on injected-dirty data again.

**Note on the CreditGuard repo living under OneDrive:** mid-session, a
pre-existing, unrelated file (`src/creditguard/pipeline/.gitkeep`, from the
Phase 1 skeleton commit) was found physically relocated to
`src/creditguard/validation/pipeline/.gitkeep` — `git status` showed it as
deleted+untracked, with no corresponding edit/write by the agent to that path.
Best guess: OneDrive sync/on-demand reconciliation, triggered by the heavy
disk I/O while Docker Desktop was cold-starting, since this repo's path
(`C:\Users\asiff\OneDrive\Documents\CreditGuard`) is inside the OneDrive sync
folder. Restored by hand. If an empty/placeholder file's location looks wrong
and you didn't touch it, check this before assuming you (or a tool) broke it.

## Feature engineering design (Phase 4)

`src/creditguard/features/` builds engineered features as composable
scikit-learn transformers, with `leakage.py` owning both what's forbidden and
how loans get matched to their point-in-time snapshot:

- **`CleaningAndMergeStep` (pipeline.py) reuses Phase 3's `DataCleaner`
  unmodified** as the pipeline's literal first stage, composed with
  `leakage.point_in_time_join` (deterministic, not a fitted statistic) so
  the rest of the pipeline gets one flat per-loan row. `DataCleaner` being
  idempotent means feeding it an already-`_clean` dataset (as `build.py`'s
  `--dataset-version` always is) is a safe no-op — the *same* pipeline object
  also works unmodified on raw input, useful for Phase 8 serving later.
- **`point_in_time_join` uses `pd.merge_asof(..., direction="backward",
  by="customer_id")`** — the right tool for "latest row with as_of_date <=
  application_date, never joined on customer_id alone."
- **Two-phase fit in `build.py`, not one opaque `pipeline.fit_transform`
  call:** the target (`default_12m`) only has a well-defined row order once
  the merge step has run (row count/order can change — `DataCleaner` drops
  quarantined rows), so `build.py` fits `pipeline.named_steps
  ["cleaning_and_merge"]` first, aligns `y` by `loan_id` (not position), then
  fits the remainder via `pipeline[1:]` — which shares the same underlying
  step objects (sklearn Pipeline slicing doesn't clone), so `pipeline` itself
  ends up fully fitted and ready to `joblib.dump`.
- **Real false-positive bug, caught by running the actual pipeline, not by
  inspection:** the spec's literal forbidden-pattern list includes a bare
  `^post_`, meant for `post_disbursement_*`. That collided with this phase's
  *own* required feature, `post_loan_dti` (entirely pre-decision — the
  projected DTI if the loan being applied for is approved), which also starts
  with `post_`. Fixed by narrowing the pattern to `^post_disbursement_`. A
  reminder that an overly broad leakage screen is a real bug (a false
  positive that would silently corrupt or block a legitimate feature), not
  just extra caution — this is exactly why `tests/test_leakage.py` asserts
  BOTH that real offenders are still caught AND that `post_loan_dti` passes
  clean.
- **`RatioFeatures`/`BehaviouralFeatures` implement `get_feature_names_out`
  as deterministic, non-data-dependent name lists** (not by re-deriving them
  from a fitted state) specifically so `Pipeline.get_feature_names_out()`
  chains all the way through — including `CleaningAndMergeStep`, whose
  output columns are fixed by `point_in_time_join`'s hardcoded column
  selection (`leakage.MERGED_FRAME_COLUMNS`), not by what the input data
  happens to contain.

## EDA design (Phase 5)

`src/creditguard/eda/` is a thin, importable analysis layer
(`univariate.py`/`bivariate.py`/`risk_analysis.py`/`plots.py`) plus one
headless orchestrator (`run_eda.py`); `notebooks/01_exploratory_data_
analysis.ipynb` only calls into it — it is a narrative wrapper, not a second
implementation.

- **`run_eda.build_eda_frame` reuses Phase 4's own pipeline stages
  (`CleaningAndMergeStep`, `RatioFeatures`, `BehaviouralFeatures`) but stops
  one stage short of the final `ColumnTransformer`.** EDA wants raw units
  (age in years, income in currency, ratios as ratios), not a fitted
  `StandardScaler`'s z-scores or one-hot columns, so it composes the same
  fit-on-train-only stages by hand instead of calling
  `features.pipeline.build_feature_pipeline` end to end.
- **Where each analysis runs is deliberate:** univariate distributions,
  categorical frequencies and band-level default rates run on the **full**
  train+val+test population (portfolio-level view, also what the Phase 9
  dashboard will want); IV/WOE, the correlation matrix and point-biserial
  correlations run on the **train split only**, since they exist to inform
  what Phase 6 actually fits. The temporal regime-shift check runs on the
  full population — the whole point is to see every month, not just the
  training months. See the module docstring in `run_eda.py` for the full
  reasoning.
- **Real bug caught by running IV/WOE against the actual 96,749-loan
  population, not by inspection:** the first `iv_table` implementation
  quantile-decile-binned every numeric column uniformly. For a zero-inflated
  low-cardinality bureau field like `previous_defaults` (64% of customers at
  0), every decile cut point collapses onto the same value and
  `duplicates="drop"` merges them into a single bin spanning the whole
  population — silently reporting IV = 0.0 ("useless") for a field
  `docs/feature_dictionary.md` documents as "the strongest single bureau
  risk signal." Fixed by routing any numeric column with ≤40 distinct values
  through the same exact-value WOE/IV path used for categoricals instead of
  quantile deciles (`risk_analysis.LOW_CARDINALITY_THRESHOLD`) — this
  recovered `previous_defaults`'s real IV of 0.106 ("medium") with a clean
  monotone default-rate gradient. The 40 cutoff isn't arbitrary: on the real
  data there's a clean gap between count-like fields (2 to 32 distinct
  values) and genuinely continuous ones (age at 52 distinct values and up).
  A reminder that a binning strategy correct for continuous features can be
  silently wrong for discrete ones, and that this class of bug only shows up
  against real data shape, not a synthetic/tiny test fixture. See
  `reports/eda/findings.md` finding 5 for the write-up, and
  `tests/test_eda_risk_analysis.py::test_iv_table_routes_low_cardinality_numeric_through_categorical_binning`
  for the regression test.
- **`plots._save` closes each figure after saving** (`plt.close(fig)`) —
  `run_eda` renders 73 figures in one process and would otherwise trip
  matplotlib's "too many open figures" warning and leak memory. The returned
  `Figure` object stays fully usable afterwards (a notebook cell can still
  display it), since closing only detaches it from pyplot's global registry,
  not from the caller's own reference to it.
- **Two pytest processes must never run against the test DB at the same
  time.** Running the full suite in the background while re-running a subset
  in a second terminal produced a real Postgres `DeadlockDetected` on the
  autouse `TRUNCATE ... RESTART IDENTITY CASCADE` fixture (two truncates
  racing), plus a flaky-looking `test_truncate_reingest_is_idempotent`
  failure in an unrelated Phase 2 test — both vanished when the suite was
  re-run alone. Not a code bug; a reminder specific to this repo's
  autouse-truncate-per-test fixture design (`tests/conftest.py`).

## Model training design (Phase 6)

`src/creditguard/models/` mirrors the same layering discipline as prior
phases: `base.py` defines `BaseCreditModel` so `logistic.py`/
`random_forest.py`/`xgboost_model.py` are interchangeable; `evaluate.py` is
the single metric suite everything else reports through (CV, per-segment,
imbalance comparisons alike); `imbalance.py`/`calibration.py`/`threshold.py`
are independent stages applied to the winning model in that order;
`tracking.py` (MLflow) and `registry.py` (`model_registry`) observe the
process without feeding back into it; `train.py` is the CLI that ties it
together.

- **`train.py` reuses Phase 4's own pipeline components directly** (not
  pre-built `X_*.parquet` files) so it needs only the clean dataset version
  on disk, and deliberately stops one stage before the final
  `ColumnTransformer` so it can keep the pre-encoding frame
  (`train_frame`/`val_frame`/`test_frame`) alongside the numeric matrices,
  row-aligned by construction — `evaluate.per_segment_metrics` needs
  human-readable `loan_type`/`age_band`/`income_band` values that don't
  survive one-hot/ordinal encoding.
- **`evaluate.select_best_model` refuses `metric="accuracy"` in code, not
  just by convention** (FR-010) — with an 11.10% default rate, predicting
  "no default" for everyone scores ~89% accuracy while having zero
  discriminatory power.
- **Real bug, caught only by running the actual 67,724-row training set
  through `RandomizedSearchCV(n_jobs=-1)`, not by smaller-scale testing:**
  XGBoost fits inside separate joblib worker *processes* crashed with
  `XGBoostError: Check failed: err == cudaGetLastError()` on this machine —
  reproducible even with `device="cpu"` and `CUDA_VISIBLE_DEVICES=""` (both
  otherwise-correct fixes that didn't touch the actual cause). Root cause:
  several worker processes probing this machine's broken/partial CUDA stack
  *simultaneously* races, regardless of what device XGBoost is told to use.
  A single process never hit it — confirmed by testing `n_jobs=1` (serial)
  at the same real scale before landing on the actual fix: XGBoost's own
  search now runs with `RandomizedSearchCV(n_jobs=1)` (serial across CV
  trials) but `XGBClassifier(n_jobs=-1)` (multi-threaded *within* that one
  process) — sidesteps the multi-process race entirely, verified safe and
  fast enough (~115s for 25 fits) at full scale. Every other family is
  unaffected (no CUDA calls at all) and keeps process-parallel search. If a
  future phase adds another GPU-capable library, treat "many processes
  touching a shaky GPU driver at once" as a real risk category, not just
  "set device=cpu and move on."
- **Real environment gap, not a code bug: a fresh Postgres volume doesn't
  carry over a manually-created database.** After the Docker Desktop
  reinstall this session (see "Environment specifics" below), the new
  `creditguard_postgres` container came up with a *fresh* named volume.
  `docker-compose.yml`'s `POSTGRES_DB` env var only auto-creates the `creditguard`
  app database on first boot — `creditguard_test` (which `tests/conftest.py`
  needs) had been created by hand in some earlier session against the *old*
  volume and was simply gone. Symptom: every DB-touching test failed with
  `sqlalchemy.exc.OperationalError`, including ones that don't touch the DB
  directly, because `_database`'s autouse fixture is session-scoped. Fixed
  with `CREATE DATABASE creditguard_test OWNER creditguard;` via
  `docker exec creditguard_postgres psql ...`, then
  `python -m creditguard.db.init_db` to apply the schema to the *app* `creditguard`
  DB too (needed for `--register-best`, separate from the test DB). Worth
  checking first, before assuming a code regression, any time Postgres
  itself was recently reinstalled/recreated rather than just restarted.
- **Imbalance comparison result was honestly reported even though it
  contradicts the usual textbook framing** (`reports/models/
  imbalance_comparison.md`, `reports/models/model_card.md`): on this
  dataset's moderate 11.10% imbalance, `class_weight` and resampling gave
  logistic regression only a 0.5-0.9 point recall gain over doing nothing,
  while clearly worsening raw Brier score (0.068 → 0.139) — the "resampling
  trades calibration for recall" story is real but its *effect size* here
  is much smaller on the recall side than the calibration side. This is
  exactly why `calibration.py` is a mandatory separate stage applied to
  whatever the winning strategy was, not an optional step: the final
  registered model's calibration slope (0.9997) is excellent only because
  it was explicitly recalibrated post-training, not because `class_weight`
  (the strategy actually used for the registered model) happened to
  preserve calibration on its own — it didn't.
- **The HOME-loan effect from `reports/eda/findings.md` finding 4 shows up
  directly in per-segment test metrics**: PR-AUC 0.737 and recall 0.969 for
  HOME loans vs. 0.27-0.46 PR-AUC elsewhere. The model is visibly exploiting
  whatever mechanism drives HOME loans' 32% default rate — flagged in
  `model_card.md` as the top open item before Phase 7/8 treat HOME-loan
  predictions as trustworthy, not treated as a good-news metric on its own.
- **Fairness check went one step further than "IV is near zero":** Phase 5
  found `gender`/`marital_status` have near-zero standalone IV, but they're
  still present in the 77-feature training matrix (never excluded). This
  phase checked the actual fitted elastic-net coefficients directly rather
  than assuming the IV finding still held post-training —
  every `gender_*`/`marital_status_*` coefficient came back effectively
  zero (0.000-0.010, vs. a max feature coefficient of 0.83) — a
  confirmatory result worth having actually checked, not just assumed.

## Scoring & explainability design (Phase 7)

`src/creditguard/scoring/` (scorecard -> categories -> recommendation ->
engine) and `src/creditguard/explain/` (shap_explainer, reason_codes) turn
Phase 6's calibrated probability into a score, a risk band, a traceable
lending decision, and a plain-English explanation.

- **`engine.score_application`'s `raw_input` is a flat, already-point-in-time
  dict, not a `customer_id` to look up.** The Phase 7 brief lists "point-in-
  time feature assembly" as its own step, which could have meant a second DB
  lookup inside the engine (by `customer_id`, as of an application date).
  Deliberately scoped narrower: `raw_input` carries the applicant's full
  snapshot directly (demographics, loan terms, financial profile, bureau
  fields), and "assembly" means shaping/validating that snapshot into the
  same column layout `features.leakage.point_in_time_join` produces for
  training -- not re-deriving it from the database. Fetching an *existing*
  customer's latest snapshot from Postgres is left to Phase 8's API layer,
  which is expected to call this function with the resolved snapshot. This
  keeps the engine a plain, DB-lookup-free function of its input (per the
  phase's own "no FastAPI, callable as a plain Python function" scope) and
  matches how a real API request body would look.
- **The feature pipeline is refit from scratch on every process's first
  scoring call, not loaded from a persisted `feature_pipeline_*.joblib`
  artifact.** A Phase 4-era artifact with that name already existed on disk
  (`models/artifacts/feature_pipeline_ds_20260808_547ecf5a_feat.joblib`),
  but it was written by `features.build`'s CLI under a filename tied to
  whatever `--output` directory name was typed by hand, not to the
  dataset_version Phase 6 actually trained the *registered* model on
  (`..._clean`, with the suffix) -- the two don't reliably match, and
  `model_registry` has no column recording which feature-pipeline artifact
  a given model was trained against. Trusting that file would have risked
  silently scoring through statistics (imputer medians, quantile bin edges,
  one-hot categories) that don't match the active model's actual training
  fit. Refitting via `build_feature_pipeline` + `merge_step.fit_transform`
  + `pipeline[1:].fit` on the active model's own registered
  `dataset_version` -- exactly what `models.train.load_training_data` does
  -- guarantees an exact match by construction, at the cost of a one-time
  ~10-25s reload per process (cached after that; see `reload_active_model`).
  If Phase 8 finds this reload too slow for cold starts, the real fix is
  giving `model_registry` a `feature_pipeline_path` column at promotion
  time, not reaching for the stale Phase 4 artifact.
- **SHAP explains the base (pre-calibration) model's log-odds output, not
  the calibrated probability.** `CalibratedClassifierCV` (isotonic here) has
  no closed-form structure for SHAP to decompose. `unwrap_base_estimator`
  reaches through `calibrated_classifiers_[0].estimator.estimator` (the
  `FrozenEstimator` wrapping the original fitted model, exactly one entry
  since the base estimator is frozen rather than cross-validated) to get the
  real `LogisticRegression`/tree object `LinearExplainer`/`TreeExplainer`
  need. Calibration being a monotonic transform means a feature's direction
  and relative importance survive it unchanged even though the raw SHAP
  numbers are in log-odds space -- documented explicitly in
  `docs/scoring_methodology.md` so this isn't a silent gap between what's
  computed and what's shown.
- **Reason-code sentences decouple the factual clause from the direction
  clause on purpose.** Early design used one phrase pair per feature keyed
  to "SHAP said risk-increasing" / "SHAP said risk-reducing" directly, which
  would have made a sentence like "no previous defaults on record" get
  attached to a row that actually *has* defaults, if SHAP's sign for that
  particular row ever disagreed with the feature's usual direction (can't
  happen for *this* linear model, since one global coefficient means sign
  follows value deterministically -- but would be a real bug for a future
  tree-based model with interactions). Fixed by keeping two independent
  computations per sentence: a factual clause from the applicant's actual
  value vs. the training-data portfolio median/mode (always true), and a
  direction clause from this row's real SHAP sign (always accurate about
  *this* application) -- see `explain.reason_codes.render_reason`.
- **Real bug, caught only by running the two required demo fixtures against
  the actual registered model, not by unit tests:** `interest_rate` was
  templated with `format="percent"` like `dti`/`credit_utilization`, which
  multiplies the raw value by 100 before appending "%". Unlike those ratio
  features (stored as 0-1 fractions), `interest_rate` is already stored in
  percentage-point units (`9.5` means 9.5%, per `docs/feature_dictionary.md`'s
  documented 1-40 range) -- so a 9.5% loan was rendered as "950%". No unit
  test caught this because the hand-built test fixtures for `reason_codes.py`
  never exercised `interest_rate` specifically end-to-end. Fixed by adding a
  distinct `rate_percent` format kind (append "%" without rescaling) and a
  regression test (`test_render_reason_rate_percent_does_not_rescale`). A
  reminder that even with full per-feature template coverage, running the
  *actual* acceptance-criteria fixtures through the real pipeline still
  finds bugs synthetic unit fixtures don't.
- **The one-hot-to-source-feature mapping (`shap_explainer.
  map_to_source_feature`) is prefix matching against the known categorical
  column list, not introspection of the fitted `OneHotEncoder`'s internal
  category/infrequent-bucket bookkeeping.** `OneHotEncoder(min_frequency=...)`
  makes the exact output-category count per column data-dependent (some
  categories fold into an `_infrequent_sklearn` bucket), which makes
  reconstructing "how many output columns came from column N" from the
  fitted encoder's own attributes fragile. Matching each encoded name
  against `f"{column}_"` prefixes from `features.yaml`'s already-known
  categorical column list is simpler and doesn't depend on encoder
  internals -- ties broken by longest-prefix-match in case one categorical
  column name is itself a prefix of another's (none currently are, but
  tested anyway).

## API design (Phase 8)

`src/creditguard/api/` is a thin transport layer over `creditguard.
scoring.engine` -- validation (`schemas.py`), auth/rate-limiting
(`dependencies.py`), request-id/logging/metrics (`middleware.py`),
exception-to-HTTP mapping (`errors.py`), and the routes themselves
(`routes/`). No scoring logic lives here; every number in a response comes
from the engine unmodified.

- **The brief's `PredictionRequest` field list was missing two things the
  engine actually needs, filled in rather than silently worked around:**
  `city_tier` (one of the 43 numeric model features, required by
  `engine.RawApplicationInput` with no default -- omitting it would have
  meant either a fabricated default silently biasing every prediction, or
  a confusing 422 the brief never asked for) was added as a real required
  field, documented inline as an addition beyond the literal list.
  `customer_id`/`loan_id` (required by the engine, but absent from the
  brief's field list, and the response schema's explicit "`loan_id`
  (nullable)" note only makes sense if the request doesn't require one)
  were made optional on `PredictionRequest`, with the route layer
  generating a synthetic `customer_id` when omitted for `/predict`/
  `/explain` (stateless what-if scoring, no database row required) --
  `POST /applications` overrides `customer_id` back to required, since
  that endpoint actually persists a customer.
- **`credit_utilization` is accepted and range-checked but never fed to
  the engine.** It's a real `credit_history` table column `POST
  /applications` needs to persist a bureau snapshot, but Phase 4's
  `RatioFeatures` always recomputes credit utilisation from
  `total_outstanding`/`total_credit_limit` (the single source of truth
  established in `creditguard.features.leakage` -- see the Phase 4 section
  above). `routes/predict.build_engine_input` explicitly drops it before
  calling the engine, with a comment explaining why, rather than silently
  ignoring it with no trace.
- **`/explain` needed a richer object than `ScoringResult` (which only
  carries the top-5 factors each direction) -- required a small,
  deliberate refactor of `engine.score_application`,** not a workaround in
  the API layer: its body was split into a private `_score_and_explain`
  (validate -> assemble -> transform -> predict -> score -> categorise ->
  recommend -> explain, returning the `ScoringResult`, the full
  `ShapExplanation`, and the raw feature row) that both `score_application`
  (adds persistence) and the new `explain_application` (returns a
  `DetailedExplanation` with every feature's contribution, not just the
  top-k) call. `score_application`'s existing behaviour, signature and
  tests are unchanged by this -- confirmed by rerunning
  `tests/test_engine.py` before writing anything Phase-8-specific.
  `reason_codes.generate_reason_codes` also gained a `"value"` key in its
  output rows (the applicant's raw value, needed for the API's
  `{feature, value, impact, description}` contract) -- additive, doesn't
  break the existing subset-based test assertions.
- **`POST /applications` writes four tables in one transaction, not
  `BaseRepository.insert_many`'s default one-transaction-per-call.**
  `insert_many` (Phase 1) opens and commits its own session per call, so
  calling it four times in a row for customer/loan/financial/credit rows
  would leave a partial application on disk if, say, the fourth insert hit
  a constraint violation -- a real correctness gap for a multi-table write,
  not just a style preference. `applications._persist_application` uses
  `creditguard.db.engine.get_session()` directly instead, all four
  `session.execute(insert(...))` calls sharing one commit-on-success/
  rollback-on-error transaction.
- **`GET /applications/{loan_id}`'s reconstructed `latest_prediction`
  always has `triggered_rules: []`, honestly, not silently wrong.** The
  `predictions` table (fixed at Phase 1) has no column for the
  recommendation's rule trace -- only the decision, probability, score and
  the two SHAP factor lists. A prediction rebuilt from a stored row can't
  reconstruct rules that were never persisted; only a fresh `/predict` call
  carries them. `model_version` *is* resolvable from a stored row (looked
  up from `model_registry` by the stored `model_id`, which is guaranteed
  to still exist because model versions are never overwritten -- CLAUDE.md
  hard rule 6) -- the two fields aren't handled the same way because one
  is genuinely unrecoverable and the other isn't.
- **`TestClient`'s default `raise_server_exceptions=True` silently
  bypasses the app's own exception handlers -- caught by a real test
  failure, not by reading the docs.** `tests/test_api_predict.py::
  test_unexpected_error_never_leaks_exception_text` initially failed with
  the raw `RuntimeError` propagating all the way out of the test process
  instead of coming back as a 500 response, even though `creditguard.
  api.errors.handle_unexpected_error` was correctly registered. Root
  cause: TestClient's default behaviour re-raises any exception that
  reaches `ServerErrorMiddleware` *in the test process itself*, specifically
  to surface real tracebacks while developing route code -- it does not
  reflect what a real client (or `uvicorn`) would see, where the same
  exception is correctly converted to a safe JSON 500. Fixed by
  constructing the shared `api_client` fixture (`tests/conftest.py`) with
  `TestClient(app, raise_server_exceptions=False)`. Worth remembering for
  Phase 9's dashboard tests too, if they ever hit the API's own error
  paths through a TestClient rather than real HTTP.
- **Measured p95 latency:** 207ms over 50 sequential `/predict` calls
  against the synthetic test fixture model (`tests/test_api_predict.py::
  test_single_prediction_p95_latency_under_2000ms`, printed, not just
  asserted) -- comfortably inside the 2000ms NFR-001 budget. Against the
  real registered Phase 6 model (manual verification, not part of the
  automated suite): single-call latency in the 20-200ms range after the
  one-time ~10-25s startup load, also well inside budget.
- **`config.Settings` gained two Phase 8 fields** (`rate_limit_rpm` default
  100, `cors_origins` default `http://localhost:8501` with a
  `cors_origin_list` computed property splitting it for `CORSMiddleware`)
  and **`db.repository.PredictionRepository` gained `query_predictions`**
  (filtered + paginated, for `GET /predictions`) -- both small, additive
  changes to earlier-phase modules, not new Phase 8-only files, because the
  capability genuinely belongs there (config and repository are exactly
  where a new setting/query method should live), not because of scope
  creep.
- **Docker image built and smoke-tested directly, not just written on
  faith:** `docker build -f docker/Dockerfile.api` succeeds (multi-stage,
  ~280s for the dependency install layer, final image non-root), and a
  real `docker run` (no volumes mounted, matching a fresh-checkout
  scenario) confirmed the intended liveness/readiness split empirically --
  `/health` stayed `200` despite the expected-and-logged startup model-load
  failure (no `data/`/`models/artifacts/` mounted), `/health/ready`
  correctly reported `503` with `model_loaded: false`. `models/artifacts/`
  and `data/processed/<version>/` are never baked into the image (both
  git-ignored generated output, per CLAUDE.md hard rule 1) -- real
  deployment must volume-mount them, documented in the Dockerfile's own
  comments and in `docs/api.md`.

## Dashboard design (Phase 9)

`src/creditguard/dashboard/` is a pure HTTP client of the Phase 8 API
(`api_client.py`) -- pages compose `components/{forms,cards,charts,tables}`
and call the client; no scoring/banding/decisioning logic lives in a page.

- **A real gap between the brief and Phase 8's actual API surface, found
  before writing any dashboard code, required extending the already-
  committed Phase 8 API rather than working around it in the dashboard --
  the user explicitly chose this over the alternatives (an N+1 `/applications`
  join that would mislabel most demo traffic as "Unknown", or silently
  dropping the requested segment breakdowns).** Two extensions, both
  additive/backward-compatible, both covered by the existing Phase 8 test
  suite rerun (305 tests, all still passing) plus new coverage:
  1. **`predictions` gained four nullable columns** (`loan_type`, `age`,
     `annual_income`, `employment_type`) -- Portfolio Analytics' segment
     breakdowns (FR-018/019/020: by loan type/income band/age band/
     employment type) need this data, but the `predictions` table never
     stored it and most scoring traffic goes through the anonymous
     `/predict` endpoint (no `customers`/`loan_applications` row to join
     to at all). `db/schema.sql` uses `ALTER TABLE ... ADD COLUMN IF NOT
     EXISTS` alongside the `CREATE TABLE IF NOT EXISTS` definition so the
     migration is idempotent against an already-existing table, not just a
     fresh one -- applied directly to both the dev and test databases via
     `python -m creditguard.db.init_db` (no data loss; existing rows just
     get `NULL` in the new columns). `scoring.engine.ScoringResult`/
     `_persist_prediction` echo these straight from the already-validated
     request; `GET /predictions` gained a matching `loan_type` filter.
  2. **New `GET /api/v1/model/performance` endpoint** for ROC/PR/
     confusion-matrix/calibration/lift-gains/feature-importance data Page
     3 needs -- none of it was persisted anywhere structured (`model_registry.
     metrics` only ever held scalars). Deliberately **not** computed live
     per request (that would make a "thin transport layer" endpoint refit
     a pipeline over the ~14.5k-row test split on every dashboard page
     view) -- instead a new one-off module, `src/creditguard/models/
     performance.py` (`python -m creditguard.models.performance`), rebuilds
     the model's own test split via the existing `models.train.
     load_training_data` and writes the results into `model_registry.
     metrics.performance` via a new `registry.update_metrics` (merges into
     the JSONB, touches no other column -- enriching a registered model's
     metadata after the fact isn't "overwriting a trained model," CLAUDE.md
     hard rule 6 is about the artifact). Run once against the real active
     model as part of this phase. `ModelVersionSummary` also gained a
     `metrics` field so the version-comparison table needs no extra calls.
  3. **`FactorDetail` gained `benchmark_median`**, populated only on
     `/explain` (not `/predict`'s top-k factor lists, which don't carry
     benchmark context) -- Page 1's "ratios vs. portfolio median" panel
     needs the training-portfolio median `creditguard.explain.
     shap_explainer.load_portfolio_benchmarks` already computes per
     feature, previously only baked into the reason-code sentence text.
  All three are documented in `docs/api.md` alongside the original Phase 8
  contract, not as a separate doc.
- **"Default rate" on Portfolio Analytics always means the *predicted*
  rate** (share of applications at/above the active model's own
  `chosen_threshold`), never an observed 12-month outcome -- a live
  scoring predictions log has no ground truth yet, and the API has no
  endpoint exposing `loan_outcomes` to the dashboard. Labelled explicitly
  as "Predicted default rate" everywhere it appears (KPI tile, segment
  charts, time-series) rather than implying it's a real default rate.
  Age/income bands used for segment charts (`state.age_band`/`income_band`)
  are fixed, human-readable display cutoffs chosen for this synthetic
  population -- not the model's own learned quantile `age_band`/
  `income_band` features (`features.behavioural`), which the dashboard has
  no access to and doesn't need to reproduce.
- **The installed Streamlit version (1.61.1) no longer auto-wires the
  classic `pages/` directory into multipage navigation -- found live in
  the browser, not from docs.** All four pages rendered correctly in
  isolation (confirmed by `streamlit.testing.v1.AppTest` smoke tests
  passing), but the sidebar showed zero navigation and zero custom
  sidebar content in a real running app -- `streamlit.source_util.
  get_pages`, the function the old auto-discovery relied on, no longer
  exists in this version (confirmed directly via `python -c "from
  streamlit.source_util import get_pages"` -> `ImportError`). Fixed by
  rewriting `app.py` as an explicit router using the current `st.
  navigation()`/`st.Page()` API (`st.Page("pages/1_Applicant_Scoring.py",
  ...)` etc., landing content moved into a local `_home()` callable passed
  directly to `st.Page`) -- the four page files keep their brief-mandated
  names/paths under `pages/`, and lost only their now-redundant individual
  `st.set_page_config`/`render_sidebar_status()` calls, which the router
  now owns centrally (also means the sidebar API-status check runs exactly
  once per navigation instead of once per page).
- **`st.cache_data`'s cache is process-global, not per-test** -- a test
  asserting "API unreachable shows a friendly error" passed in isolation
  but failed when run after a test that had already cached a *successful*
  `cached_model_info()` result, since the cache silently served the stale
  success instead of hitting the (now unmocked) endpoint. Fixed with an
  autouse `st.cache_data.clear()` fixture in `tests/
  test_dashboard_components.py`. Same TTL-cache mechanism is why the
  dashboard's own read endpoints (`READ_CACHE_TTL_SECONDS = 30`) don't
  hammer the API on every page rerun -- `/predict`/`/explain`/
  `/applications` are never wrapped in it.
- **Verified live against the real registered model and real accumulated
  prediction traffic, not just the automated suite** (108 predictions
  logged by this point): Applicant Scoring's high-risk sample scored
  300/VERY_HIGH/REJECT end-to-end through the live API with the SHAP
  chart and reason codes rendering; Portfolio Analytics rendered real
  KPIs/segment charts/a 108-row searchable table with filters present;
  Model Performance showed the real backfilled metrics matching
  `reports/models/model_card.md` exactly (ROC-AUC 0.8770, PR-AUC 0.5450,
  etc.) plus all six curve/importance charts and a version-comparison
  table. One incidental finding while doing this: a `uvicorn --reload`
  process not started by this session was found already listening on
  port 8000 (a leftover from independent manual use, matching the
  environment notes in the other memory file about running the API by
  hand) -- it was stopped once by mistake while testing the "API down"
  friendly-error path and immediately restarted with the same flags; the
  automated test (`test_portfolio_analytics_shows_friendly_error_when_api_
  unreachable`) covers that scenario without needing to repeat the live
  disruption.

## How this project likes to be verified

- When a phase's acceptance criteria name a specific scale (row count, customer
  count, etc.), actually run it at that scale before declaring the phase done —
  don't extrapolate from a smaller test run. Small-scale runs here (2k customers)
  passed cleanly but hid a real PostgreSQL scaling failure that only appeared at
  the actual 50k-customer acceptance-criteria scale.
- For long-running background steps, proactively set up a way for the user to
  independently verify progress themselves (e.g. direct DB row-count queries, a
  progress log) rather than only reporting status yourself — this matters more
  when the operation could plausibly hang. Prefer designs where partial progress
  is visibly committed/observable as it happens over designs that are only
  correct-looking once complete and opaque until then.
- Only commit to git when explicitly asked — true for every phase so far
  (each was verified first, committed only once the user asked).
