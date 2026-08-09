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
  while visibly hurting raw calibration).
- **Next up:** Phase 7 (credit scoring engine, risk categories and
  explainability), per `CLAUDE.md`'s checklist. Phases are strictly
  sequential — don't start Phase 7 work, and don't add stub files for it,
  until the user actually asks.

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
