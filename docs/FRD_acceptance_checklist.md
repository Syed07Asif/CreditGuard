# FRD Acceptance Checklist

**A note on sourcing.** No standalone Functional Requirements Document exists as a
file in this repository — only the requirement coverage matrix from the phased build
plan (FR-001 through FR-027, NFR-001 through NFR-005, each mapped to the phase(s) that
deliver it) and each phase's own prompt (which named the FR/NFR IDs it covers in its
header). This checklist is built directly from that matrix and from what each phase
actually shipped — not from a "FRD section 38" text, which was not available when this
checklist was written. Where the Phase 10 brief's own acceptance criteria are the
closest available proxy for that section, they're listed as a supplementary table at
the end, clearly marked as such.

Every mark below reflects what was actually run and observed in this session, not
what the code merely appears to do — see the "Evidence" column.

---

## Functional requirements

| ID | Requirement | Module(s) | Test(s) | Mark | Evidence |
|---|---|---|---|---|---|
| FR-001 | Customer registration | `db/models.py` (`Customer`), `data/generator.py`, `data/ingest.py`, `api/routes/applications.py` | `test_db_models.py`, `test_generator.py`, `test_ingest.py`, `test_api_validation.py` | ✅ PASS | Full suite green (392 passed, 1 slow test deselected, 0 failed — final run of this session); `POST /applications` persists a `customers` row (exercised live in `test_end_to_end.py`). |
| FR-002 | Create loan application | `db/models.py` (`LoanApplication`), same as above | same as above | ✅ PASS | Same. |
| FR-003 | Capture financial information | `db/models.py` (`FinancialProfile`), `features/ratios.py` | `test_db_models.py`, `test_ratios.py` | ✅ PASS | |
| FR-004 | Capture credit information | `db/models.py` (`CreditHistory`) | `test_db_models.py` | ✅ PASS | |
| FR-005 | Generate risk features | `features/ratios.py`, `behavioural.py`, `encoders.py`, `pipeline.py` | `test_ratios.py`, `test_feature_pipeline.py` | ✅ PASS | |
| FR-006 | Validate input data | `validation/rules.py`+`engine.py` (historical), `api/schemas.py` (request-time), `monitoring/data_quality.py` (Phase 10: production traffic, scheduled) | `test_rules.py`, `test_cleaning.py`, `test_api_validation.py`, `test_data_quality.py` | ✅ PASS | Phase 10 extends this to production, not just the training dataset — `run_data_quality_check` re-runs the exact Phase 3 rule engine on live data. |
| FR-007 | Perform EDA | `eda/*` | `test_iv_woe.py`/`test_eda_run.py` (per Phase 5) | ✅ PASS | |
| FR-008 | Train credit risk model | `models/train.py` | `test_models.py` | ✅ PASS | |
| FR-009 | Default prediction target | `db/models.py` (`LoanOutcome`), `data/generator.py`'s label process | `test_generator.py` | ✅ PASS | |
| FR-010 | Evaluate models (never on accuracy alone) | `models/evaluate.py` (`select_best_model` raises `ValueError` on `metric="accuracy"`) | `test_evaluate.py` | ✅ PASS | |
| FR-011 | Handle class imbalance | `models/imbalance.py` | `test_models.py` | ✅ PASS | |
| FR-012 | Generate credit risk score | `scoring/scorecard.py` | `test_scorecard.py` | ✅ PASS | |
| FR-013 | Assign risk category | `scoring/categories.py` | `test_categories.py` | ✅ PASS | |
| FR-014 | Generate recommendation | `scoring/recommendation.py` | `test_recommendation.py` | ✅ PASS | |
| FR-015 | Explain prediction (SHAP) | `explain/shap_explainer.py`, `reason_codes.py` | `test_explainer.py` | ✅ PASS | |
| FR-016 | Risk prediction API | `api/routes/predict.py`, `main.py` | `test_api_predict.py` | ✅ PASS | |
| FR-017 | Applicant risk dashboard | `dashboard/pages/1_Applicant_Scoring.py` | `test_dashboard_components.py` | ✅ PASS | |
| FR-018 | Portfolio analytics | `dashboard/pages/2_Portfolio_Analytics.py` | `test_dashboard_components.py` | ✅ PASS | |
| FR-019 | Segment analytics | `eda/risk_analysis.py`, `dashboard/pages/2_Portfolio_Analytics.py` | `test_dashboard_components.py` | ✅ PASS | |
| FR-020 | Reporting / export | `dashboard` (CSV export, JSON download) | `test_dashboard_components.py` | ✅ PASS | |
| FR-021 | Automated pipeline (single CLI, generate→monitor) | `pipeline/orchestrator.py` | `test_orchestrator.py` | ✅ PASS | **Live-verified**: `python -m creditguard.pipeline.orchestrator run-all --generate --ingest --validate --clean --features --train --register --monitor --n-customers 500 --models logistic` ran the entire chain in one command and finished in 136s, ending with a printed per-stage summary (`generate: ds_20260811_b0346464`, `ingest: {…inserted…}`, `validate: {passed: False, n_violations: 67}` — expected, Phase 2 injects errors on purpose — `clean`, `features: {n_features: 77}`, `train`, `register: {model_id: …}`, `monitor: {…}`). `test_orchestrator.py` additionally covers `orchestrator.main()`'s own control flow (stage ordering, `--from-stage`/`--to-stage`, `--register` requiring `--train`, `PipelineError` naming the failing stage) with the individual stage functions stubbed — 6/6 pass. |
| FR-022 | Prevent future information leakage | `features/leakage.py` | `test_leakage.py` | ✅ PASS | |
| FR-023 | Track model versions | `models/registry.py` (`register_model`/`activate_model`, semantic versioning, never overwritten) | `test_registry.py` | ✅ PASS | Phase 10 additively extends `register_model` with `activate: bool = True` (backward compatible) so `monitoring.retraining` can register a challenger without auto-promoting it — see `models/registry.py`'s own docstring. |
| FR-024 | Monitor model performance | `monitoring/performance.py` | `test_performance_monitor.py` | ✅ PASS | Restricts to matured (outcome-observed) loans only — verified with a real DB-backed test, not just a unit fixture. |
| FR-025 | Detect data drift | `monitoring/drift.py` | `test_drift.py` | ✅ PASS | PSI/KS/chi-square all verified against hand-computed values and constructed shift fixtures; live demonstration below. |
| FR-026 | Model retraining | `monitoring/retraining.py` | `test_retraining.py` | ✅ PASS | `should_retrain` verified True for each of the three FR-026 triggers in isolation and False when none hold; `trigger_retraining` verified live (real training run) to both promote a winning challenger and correctly refuse a losing one; `rollback_to_version` verified. |
| FR-027 | Data protection | `config.py` (secrets only via env/`.env`), `api/middleware.py` (never logs full payloads), `api/dependencies.py` (API-key auth) | `test_config.py`, `test_api_auth.py` | ✅ PASS | Phase 10 follows the same pattern: `ALERT_WEBHOOK_URL` is env-only, alert log lines never carry a raw applicant payload. |

## Non-functional requirements

| ID | Requirement | Module(s) | Test(s) | Mark | Evidence |
|---|---|---|---|---|---|
| NFR-001 | Performance (p95 < 2s per prediction) | `scoring/engine.py`, `api/routes/predict.py` | `test_api_predict.py` | ✅ PASS | Unchanged by Phase 10; Phase 8's own measured p95 was ~107ms, far under budget. |
| NFR-002 | Reliability under invalid input | `api/schemas.py`, `api/errors.py` | `test_api_validation.py` | ✅ PASS | Unchanged by Phase 10. |
| NFR-003 | Scalability | `docker-compose.yml` (independently scalable `api`/`monitoring` containers, shared volumes) | *(none dedicated)* | ⚠️ PARTIAL | Containerisation exists, but there is no load test, and `api/dependencies.py`'s rate limiter and `api/middleware.py`'s metrics store are explicitly documented as in-process/single-worker only ("a real multi-worker deployment would need a shared store instead; out of scope here" — a Phase 8 decision, not new to Phase 10, but one Phase 10 inherits and does not resolve). Honest mark: infrastructure for horizontal deployment exists; the claim "scales" is not evidenced by a test. |
| NFR-004 | Maintainability / modular code | all of `monitoring/`, `pipeline/` (small, single-purpose modules; every function under ~50 lines; no module doing more than one job) | full suite | ✅ PASS | |
| NFR-005 | Reproducibility | `SEED=42` used throughout generation/training/retraining; `monitoring/retraining.py`'s `trigger_retraining` reuses the same seeded search path as `models/train.py` | full suite, `test_retraining.py` | ✅ PASS | |

---

## Phase 10's own acceptance criteria

The closest available proxy for "FRD section 38" — the Phase 10 brief's own 7-item
acceptance list, each verified directly in this session.

| # | Criterion | Mark | Evidence |
|---|---|---|---|
| 1 | `docker compose up` brings up the entire stack; API and dashboard are reachable | ✅ PASS | **Live-verified in this session.** All three new images (`creditguard-api`, `creditguard-dashboard`, `creditguard-monitoring`) built successfully; `docker compose up -d` brought up `postgres`→`mlflow`→`api`→`monitoring` in the documented dependency order, with `api` and `monitoring` both reaching Docker-healthy status (`GET /health` returned `{"status":"ok"}`; `/health/ready` correctly returned `{"status":"not_ready","database":true,"model_loaded":false}` on this genuinely fresh stack — expected, since the named `dataset_data`/`model_artifacts` volumes start empty until the orchestrator seeds them, exactly as `docs/deployment.md` documents). `dashboard`'s own container image was separately confirmed reachable (`GET /_stcore/health` → 200) on an alternate port, since this host machine already had an unrelated process bound to 8501 — a host-environment conflict, not a defect in the compose file. |
| 2 | The orchestrator runs the full chain from generation to monitoring in one command | ✅ PASS | See FR-021 above — live-verified and now covered by `test_orchestrator.py`. |
| 3 | Injecting a shifted feature distribution produces a DRIFT status and an alert — demonstrate this | ✅ PASS | **Live-verified in this session, twice.** First: a dedicated demo registered a model, built its baseline (`annual_income` mean≈₹410,732, std≈₹268,497), then injected 40 production applications with `annual_income=₹5,000,000` (~17 standard deviations out) — the resulting drift check reported `annual_income` PSI = **12.4339** (far past the 0.25 DRIFT threshold) and `any_drift = True`, and `alerts.alerts_for_drift_run` produced a WARNING `Alert` per drifted feature (52 features, PSI **and** KS/chi-square each), all dispatched to the console sink. Second, incidentally: the orchestrator's own `--monitor` stage (see criterion 2) picked up that same leftover shifted data still in the shared dev database and independently reported the identical PSI=12.4339 DRIFT + `should_retrain` recommending retraining for that exact reason — real cross-run evidence, not a rehearsed fixture. |
| 4 | Retraining produces v1.1 while v1.0 remains loadable and inactive | ✅ PASS | `test_trigger_retraining_creates_new_version_and_old_artifact_stays_loadable` passes against a real (small) training run: the challenger is promoted, the old champion's artifact is confirmed still loadable via `joblib.load`. |
| 5 | `test_end_to_end.py` passes | ✅ PASS | See below. |
| 6 | CI is green including the compose smoke test | ⚠️ NOT LIVE-VERIFIED | `.github/workflows/ci.yml` (extended) and `.github/workflows/docker.yml` (new) are written and match the local test run exactly (same `pytest`/`ruff`/`black`/`mypy` commands were run locally and passed), but neither workflow has actually executed on GitHub Actions in this session — that requires a push, which was not requested. |
| 7 | All 14 success criteria in FRD section 38 are satisfied and evidenced here | ⚠️ CANNOT VERIFY | The literal text of "section 38" was never available in this repo or session (see the sourcing note at the top). This table and the FR/NFR table above are the honest substitute. |

---

## Honest summary of partial items

Only two items remain less than a full PASS, and neither is a functional gap:

1. **NFR-003 (scalability):** the deployment topology (independently scalable `api`/
   `monitoring` containers) supports it, but nothing measures it, and two known
   single-process limitations (`api/dependencies.py`'s rate limiter,
   `api/middleware.py`'s metrics store) are explicitly out of scope per Phase 8's own
   design notes, which Phase 10 inherits rather than resolves.
2. **Criterion 6 (CI green on GitHub):** `.github/workflows/ci.yml`/`docker.yml` are
   written to run exactly the same `pytest`/`ruff`/`black`/`mypy` commands (and the
   same `docker build` + `docker compose up` + health-curl sequence) that were run and
   passed locally in this session, but GitHub Actions itself was not exercised — that
   requires a push, which was not requested.

A real bug was found and fixed during this session's live verification, worth noting
here for transparency: `monitoring/drift.py`'s PSI-heatmap plotting crashed on
Postgres `NUMERIC` columns coming back as `decimal.Decimal` (matplotlib can't render an
object-dtype array) — caught only by an actual live drift run, not by any unit test's
synthetic fixtures (which passed Python floats directly). Fixed by casting to `float`
before pivoting/plotting; `test_end_to_end.py`'s monitoring assertions were also
strengthened afterward, since `monitoring/scheduler.py`'s deliberate per-check
exception handling had silently swallowed the original crash.

`mypy src/` (newly added to `ci.yml` by this phase) surfaced 14 pre-existing type
errors in Phase 5/7/8/9 files that had simply never been type-checked before (this
repo had no mypy config or CI step until now) — all 14 were fixed in this session
(small, behaviour-preserving annotation/cast fixes only) since leaving them would make
the new CI gate this phase adds permanently red through no fault of Phase 10's own
code. One pre-existing issue was deliberately **not** touched: `ruff check .` flags 5
long lines inside `notebooks/01_exploratory_data_analysis.ipynb` (Phase 5's committed,
already-reviewed notebook) — likely surfaced by a ruff version upgrade adding
notebook-linting support since that phase was committed, not by anything in this
session. Editing a `.ipynb`'s JSON structure via text replacement risks corrupting it
for a cosmetic lint fix in a file this phase has no reason to touch, so it was left
alone and is flagged here instead: `ruff check .` (as opposed to `ruff check src/
tests/`) will not currently be 100% clean until someone re-runs and re-saves that
notebook.

Nothing above is marked PASS without something in this session actually having been
run and observed passing.
