# CreditGuard

CreditGuard is an end-to-end credit risk scoring **and monitoring** platform: it
estimates an applicant's probability of default within 12 months, turns that into a
credit score, a risk category, a lending recommendation and a plain-English
explanation — and then watches itself in production for drift, performance
degradation and data-quality problems, retraining and versioning itself when the
evidence says it should.

> **This is a portfolio / educational simulation, not a production lending system.**
> All customer data is synthetic; no real applicant, bureau or banking data is used
> anywhere in this repository. Nothing here is validated for real lending decisions
> and would require governance, regulatory compliance, fairness testing and human
> oversight before any real-world use — see `reports/models/model_card.md` for the
> full disclaimer.

## Architecture

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

`src/creditguard/pipeline/orchestrator.py` runs that whole chain as one CLI command.

## Quickstart (Docker, full stack)

```bash
git clone <this repo> && cd CreditGuard
copy .env.example .env                 # macOS/Linux: cp .env.example .env
# edit .env: set a real API_KEY
docker compose up -d
docker compose exec api python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 50000
```

Then open **http://localhost:8501** for the dashboard, or **http://localhost:8000/docs**
for the raw API. See [`docs/deployment.md`](docs/deployment.md) for the full bring-up
sequence, environment variable reference, and how retraining works inside the stack.

## Quickstart (local, no Docker for the app itself)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
docker compose up -d postgres mlflow
python -m creditguard.db.init_db
python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 50000
uvicorn creditguard.api.main:app --reload
```

## What each page of the dashboard does

- **Applicant Scoring** — full input form (or a "load sample" button for a
  low/medium/high-risk demo applicant), returns a 300–900 score with a gauge, the
  default probability, risk category and APPROVE/REVIEW/REJECT recommendation, a
  SHAP-based contribution chart, and reason-code sentences comparing the applicant to
  the portfolio median.
- **Portfolio Analytics** — filterable KPIs (approval/review/reject shares, average
  score, default rate), score/risk-category distributions, default rate by loan type/
  income band/age band/employment type, and a searchable, CSV-exportable predictions
  table.
- **Model Performance** — the active model's card (algorithm, training date, chosen
  threshold), metric tiles (ROC-AUC, PR-AUC, KS, Brier, precision/recall/F1), ROC/PR/
  calibration curves, a confusion matrix, and global feature importance.
- **Monitoring** *(Phase 10)* — per-feature drift status and PSI, prediction-
  probability drift (the earliest warning signal), rolling predictive/operational
  performance metrics, and the data-quality violation trend — all read from what the
  scheduled monitoring job already computed, never recomputed on page load.

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

## Stack

Python 3.11 · PostgreSQL · SQLAlchemy 2.0 · scikit-learn · XGBoost · SHAP · MLflow ·
FastAPI · Streamlit · Matplotlib/Seaborn · pytest · Docker · GitHub Actions.
See [`CLAUDE.md`](CLAUDE.md) for the full stack, repository layout and hard rules
every phase followed.

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

English-language synthetic data only; no fairness/bias audit; the in-process rate
limiter and metrics store are single-worker only (a real multi-worker deployment
would need a shared store); monitoring's per-feature drift needs real production
traffic in the database (via `POST /applications` or `--ingest`) to say anything
beyond prediction-probability drift. See
[`docs/FRD_acceptance_checklist.md`](docs/FRD_acceptance_checklist.md) for the full,
honest accounting of what is and isn't verified.
