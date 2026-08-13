# CreditGuard

**An end-to-end credit risk scoring and monitoring platform.** Given a loan
applicant's profile, it estimates their probability of default within 12 months,
turns that into a 300–900 credit score, a risk category, a lending
recommendation (Approve / Review / Reject), and a plain-English explanation of
*why*. It then keeps watching itself in production — tracking data drift, model
performance decay, and data-quality problems — and can retrain and promote a
new model version when the evidence says the old one is falling behind.

> **This is a portfolio / educational simulation, not a production lending
> system.** All customer data is synthetic — no real applicant, bureau, or
> banking data is used anywhere in this repository. Nothing here is validated
> for real lending decisions and would require governance, regulatory
> compliance, fairness testing, and human oversight before any real-world use.
> See [`reports/models/model_card.md`](reports/models/model_card.md) for the
> full disclaimer.

---

## What this project demonstrates

CreditGuard was built to cover the full lifecycle a real ML system needs, not
just a notebook that trains a model once:

1. **Synthetic data generation** with realistic, tunable relationships between
   income, employment, credit history, and default risk — so the whole
   pipeline is reproducible from nothing, with no licensing or privacy concerns.
2. **Data validation and cleaning**, with quarantine logic for bad records and
   a generated data-quality report.
3. **Feature engineering** (ratios, behavioural signals, encodings) with an
   explicit leakage check.
4. **Model training and calibration** (logistic regression / XGBoost, class
   imbalance handling, probability calibration, threshold selection) tracked
   in MLflow.
5. **Explainability**: every prediction comes with SHAP-based feature
   contributions and human-readable reason codes, not just a number.
6. **A real-time scoring API** (FastAPI) and an **analyst-facing dashboard**
   (Streamlit) that only ever talks to that API — never to the model directly.
7. **Production monitoring**: population/feature drift (PSI), performance
   decay, and data-quality checks, running on a schedule, with an automated
   champion-vs-challenger retraining decision.
8. **A fully containerized deployment** (Docker Compose) so the entire stack —
   database, experiment tracker, API, dashboard, and monitoring scheduler —
   comes up with one command on any machine.

## How it works

```
                         ┌─────────────────────────────────────────────┐
                         │              docker compose                  │
                         │                                                │
  applicant / analyst    │   ┌──────────┐        ┌──────────────────┐    │
  ────────────────────►  │   │Dashboard │──HTTP─►│    FastAPI API    │    │
        browser          │   │(Streamlit)│        │  /predict /explain │    │
                         │   └──────────┘        │  /applications      │    │
                         │                        │  /model/* /monitoring/*│
                         │                        └─────────┬──────────┘    │
                         │                                  │                │
                         │        ┌─────────────────────────┼─────────┐      │
                         │        │                          │         │      │
                         │        v                          v         │      │
                         │  ┌───────────┐            ┌───────────────┐ │      │
                         │  │ PostgreSQL │◄──────────►│   MLflow       │ │      │
                         │  │ (state)    │            │ (experiments)  │ │      │
                         │  └─────┬─────┘            └───────────────┘ │      │
                         │        ^                                     │      │
                         │        │ drift_reports / monitoring_metrics /│      │
                         │        │ data_quality_issues                  │      │
                         │  ┌─────┴──────────┐                          │      │
                         │  │  Monitoring     │  drift, performance,     │      │
                         │  │  (scheduler)    │  data-quality checks,     │      │
                         │  │                 │  should_retrain, alerts   │      │
                         │  └─────────────────┘                          │      │
                         └─────────────────────────────────────────────┘
```

The data's own path through the system, end to end:

```
generate → validate → clean → engineer features → train → register →
score (API) → monitor (drift / performance / data quality) → retrain → repeat
```

`src/creditguard/pipeline/orchestrator.py` runs that whole chain as one CLI
command — this is what makes the project reproducible on a fresh machine with
zero shipped data (see Quickstart below).

## Tech stack, and why

| Layer | Choice | Why this, here |
|---|---|---|
| Language | Python 3.11 | The ML ecosystem (scikit-learn, SHAP, pandas) this project depends on lives here. |
| Modeling | scikit-learn, XGBoost | Both are strong, well-understood baselines for tabular credit risk; logistic regression stays interpretable, XGBoost trades some of that for accuracy — the registry can hold either. |
| Explainability | SHAP | Lending decisions need to be justifiable per-applicant, not just accurate in aggregate — SHAP gives a feature-level "why" for every score. |
| Experiment tracking | MLflow | Every training run's parameters, metrics, and artifacts are logged and comparable, not just the final model file. |
| Database | PostgreSQL + SQLAlchemy 2.0 | Applicants, loans, predictions, and monitoring history are relational and need real transactional guarantees, not a flat file. |
| Scoring API | FastAPI | Async, typed request/response models, and free interactive API docs (Swagger UI) out of the box. |
| Dashboard | Streamlit | Fast to build an analyst-facing UI without hand-rolling a separate frontend app — and it's kept as a pure HTTP client of the API, so it can never bypass the same validation/auth the API enforces. |
| Deployment | Docker Compose | Five services (database, experiment tracker, API, dashboard, monitoring scheduler) come up identically on any machine with one command — the same thing this README asks a reader to run. |
| CI / quality | pytest, ruff, black, mypy, GitHub Actions | Tests, linting, formatting, and type-checking run the same way locally and in CI. |

---

## Quickstart (Docker, full stack) — recommended

This is the fastest way to see it running, on your machine or someone else's.

**Requirements:** [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed and running, Git, and ideally 8GB+ of free RAM (see
[Troubleshooting](#troubleshooting--faq) if your machine is tighter than that).

```bash
git clone https://github.com/Syed07Asif/CreditGuard.git
cd CreditGuard
cp .env.example .env          # Windows: copy .env.example .env
# edit .env and set a real API_KEY (any value works for a demo)
docker compose up -d
```

The first run builds three images from scratch and can take a few minutes.
Once all five containers (`postgres`, `mlflow`, `api`, `dashboard`,
`monitoring`) are up, seed a dataset and train the first model — nothing is
shipped pre-trained, it's all generated fresh from code:

```bash
docker compose exec api python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 50000
```

On a memory-constrained machine, use a smaller number instead (see
[Troubleshooting](#troubleshooting--faq)):

```bash
docker compose exec api python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 5000
```

Confirm it worked:

```bash
curl http://localhost:8000/health/ready
# {"status":"ready","database":true,"model_loaded":true}
```

Then open:

- **http://localhost:8501** — the dashboard
- **http://localhost:8000/docs** — the raw API (Swagger UI; Authorize with
  your `API_KEY` from `.env`)

When you're done:

```bash
docker compose down
```

This keeps your generated data and trained model (they live in Docker
volumes, not the containers) — the next `docker compose up -d` starts
instantly, no reseeding needed. See
[`docs/deployment.md`](docs/deployment.md) for the full bring-up sequence,
every environment variable, and how retraining works inside the stack.

## Quickstart (local, no Docker for the app itself)

For active development/iteration — runs the API and dashboard as plain local
processes against Dockerized Postgres/MLflow only.

```bash
python -m venv .venv && .venv\Scripts\activate    # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d postgres mlflow
python -m creditguard.db.init_db
python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 50000
uvicorn creditguard.api.main:app --reload
```

In a second terminal, with `API_BASE_URL`/`API_KEY` set:

```bash
streamlit run src/creditguard/dashboard/app.py --server.port 8501
```

---

## Troubleshooting & FAQ

**"`docker compose up -d` failed with a pipe/API connection error."**
Docker Desktop hadn't finished starting yet. Wait 30–60 seconds after
launching it and run the command again.

**"A container's port is already in use."**
Something else on your machine is using port 8000, 8501, 5433, or 5000. Find
and stop it, or change the host-side port mapping in `docker-compose.yml`
(the left-hand side of `"8501:8501"`, for example).

**"`/health/ready` says `model_loaded: false`."**
This is expected on a completely fresh stack — you haven't run the seeding
command yet (see Quickstart, step "seed a dataset and train the first
model"). Run it, wait for it to finish, then check again. If you *have*
already seeded it and still see this, check `docker compose logs api` for a
`startup_model_load_failed` line — that means the model artifact or its
training dataset went missing from the `model_artifacts`/`dataset_data`
volumes (most likely from a `docker compose down -v` — see below).

**"Training got killed / the `api` container restarted on its own during
`--train`."** This is Docker running out of memory, not a bug in the code.
The `--train` step (fitting the model and building SHAP explanations) is the
most memory-hungry part of the whole pipeline. Two options: give Docker
Desktop more memory (Settings → Resources), or just re-run the seeding
command with a smaller `--n-customers` (5,000 is plenty for a demo; the
project's own acceptance criteria were verified at 50,000).

**"I ran `docker compose down -v` and now nothing works."**
`-v`/`--volumes` deletes the named volumes — your generated dataset, trained
model, and database are gone. This isn't recoverable short of re-seeding.
Plain `docker compose down` (no `-v`) never does this — it's the safe one to
use between sessions.

**"How do I see what's actually going wrong inside a container?"**
```bash
docker compose logs -f api          # follow logs live
docker compose logs api --tail 50   # last 50 lines
docker compose ps                   # container health at a glance
```

**"How do I start completely fresh?"**
```bash
docker compose down -v
docker compose up -d
# then re-run the seeding command from Quickstart
```

**"Can I run this without Docker at all?"** Yes — see "Quickstart (local, no
Docker for the app itself)" above. You'll still need Docker for Postgres and
MLflow specifically, unless you point `DB_HOST`/`MLFLOW_TRACKING_URI` at
your own instances.

---

## What each page of the dashboard does

- **Applicant Scoring** — full input form (or a "load sample" button for a
  low/medium/high-risk demo applicant), returns a 300–900 score with a gauge,
  the default probability, risk category and APPROVE/REVIEW/REJECT
  recommendation, a SHAP-based contribution chart, and reason-code sentences
  comparing the applicant to the portfolio median.
- **Portfolio Analytics** — filterable KPIs (approval/review/reject shares,
  average score, default rate), score/risk-category distributions, default
  rate by loan type/income band/age band/employment type, and a searchable,
  CSV-exportable predictions table.
- **Model Performance** — the active model's card (algorithm, training date,
  chosen threshold), metric tiles (ROC-AUC, PR-AUC, KS, Brier,
  precision/recall/F1), ROC/PR/calibration curves, a confusion matrix, and
  global feature importance.
- **Monitoring** — per-feature drift status and PSI, prediction-probability
  drift (the earliest warning signal), rolling predictive/operational
  performance metrics, and the data-quality violation trend — all read from
  what the scheduled monitoring job already computed, never recomputed on
  page load.

## Project structure

```
creditguard/
├── config/                 YAML domain configuration (generation, validation,
│                            features, model, scoring, monitoring)
├── db/schema.sql            PostgreSQL schema
├── docker/                  Dockerfile.{api,dashboard,monitoring}
├── docs/                    data_generation, data_quality, feature_dictionary,
│                            scoring_methodology, api, monitoring, deployment,
│                            FRD_acceptance_checklist
├── notebooks/                01_exploratory_data_analysis.ipynb
├── reports/                  generated data-quality/EDA/model/monitoring reports
├── src/creditguard/
│   ├── config.py             typed Settings (env vars)
│   ├── db/                   engine, ORM models, repositories, schema init
│   ├── data/                 synthetic generation, ingestion, versioning
│   ├── validation/            rule engine, cleaning, report, CLI
│   ├── features/               ratios, behavioural, encoders, pipeline, leakage
│   ├── eda/                    univariate/bivariate/risk analysis, plots
│   ├── models/                  training, imbalance, evaluation, calibration,
│   │                            threshold, tracking (MLflow), registry
│   ├── scoring/                  scorecard, categories, recommendation, engine
│   ├── explain/                    SHAP explainer, reason codes
│   ├── api/                        FastAPI app, schemas, routes, middleware
│   ├── dashboard/                   Streamlit app, api client, pages, components
│   ├── monitoring/                   baseline, drift, performance, data_quality,
│   │                                 alerts, retraining, scheduler
│   └── pipeline/                      orchestrator (the FR-021 end-to-end CLI)
└── tests/                    one module (roughly) per source module
```

## Documentation

| Doc | Covers |
|---|---|
| [`docs/data_generation.md`](docs/data_generation.md) | The synthetic data-generating process and its coefficients |
| [`docs/data_quality.md`](docs/data_quality.md) | The Phase 3 validation rule set |
| [`docs/feature_dictionary.md`](docs/feature_dictionary.md) | Every engineered feature, formula and expected direction |
| [`docs/scoring_methodology.md`](docs/scoring_methodology.md) | The points-to-double-odds score scaling, worked example |
| [`docs/api.md`](docs/api.md) | Every API endpoint, request/response shapes |
| [`docs/monitoring.md`](docs/monitoring.md) | What's monitored, PSI/KS/chi-square explained, the retraining decision flow |
| [`docs/deployment.md`](docs/deployment.md) | Local/Docker bring-up, environment variables, volumes |
| [`docs/FRD_acceptance_checklist.md`](docs/FRD_acceptance_checklist.md) | Every FR/NFR, its module, its test, and an honest pass/partial mark |

## Development

```bash
pytest -q                 # full suite (a real PostgreSQL test database is required)
pytest -q -m "not slow"   # skip the real-training/real-generation regression tests
ruff check .
black --check .
mypy src/
```

## Limitations

English-language synthetic data only; no fairness/bias audit; the in-process
rate limiter and metrics store are single-worker only (a real multi-worker
deployment would need a shared store); monitoring's per-feature drift needs
real production traffic in the database (via `POST /applications` or
`--ingest`) to say anything beyond prediction-probability drift. See
[`docs/FRD_acceptance_checklist.md`](docs/FRD_acceptance_checklist.md) for the
full, honest accounting of what is and isn't verified.
