# CLAUDE.md — CreditGuard Project Contract

This file is the shared context for every session on this repository. Read it first,
follow it exactly, and tick the phase checklist as work is completed.

---

## 1. Project purpose

CreditGuard is an end-to-end credit risk scoring and monitoring platform that estimates
an applicant's probability of default within 12 months and turns it into a credit score,
a risk category, a lending recommendation and a plain-English explanation.
It covers the full lifecycle: data generation, validation, feature engineering, model
training, calibrated scoring, a REST API, a dashboard, and production monitoring.
This is a **portfolio / educational simulation, not a production lending system**.
All customer data is **synthetic**; no real applicant, bureau or banking data is used.
Nothing here is validated for real lending decisions.

---

## 2. Fixed stack

Do not substitute, add heavyweight alternatives, or swap libraries without being asked.

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Database | PostgreSQL |
| ORM / DB access | SQLAlchemy 2.0 |
| Machine learning | scikit-learn |
| Gradient boosting | XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow |
| API | FastAPI + Pydantic v2 |
| Dashboard | Streamlit |
| Visualisation | Matplotlib / Seaborn |
| Testing | pytest |
| Lint / format | Ruff, Black |
| Containerisation | Docker, docker-compose |
| CI | GitHub Actions |

---

## 3. Repository layout

```
creditguard/
├── CLAUDE.md
├── .gitignore
├── .env.example                    # Phase 1 — never commit a real .env
├── pyproject.toml                  # Phase 1
├── Makefile                        # Phase 1
├── docker-compose.yml              # Phase 1, extended in Phase 10
├── README.md                       # Phase 1, rewritten in Phase 10
│
├── .vscode/                        # settings.json, extensions.json
├── .github/workflows/              # ci.yml (Phase 1), docker.yml (Phase 10)
├── .streamlit/                     # config.toml (Phase 9)
│
├── config/                         # YAML configuration, not Python
│   ├── data_generation.yaml        # Phase 2
│   ├── validation_rules.yaml       # Phase 3
│   ├── features.yaml               # Phase 4
│   ├── model_config.yaml           # Phase 6
│   ├── scoring.yaml                # Phase 7
│   └── monitoring.yaml             # Phase 10
│
├── db/
│   └── schema.sql                  # Phase 1
│
├── docker/                         # Dockerfile.api, Dockerfile.dashboard,
│                                   # Dockerfile.monitoring
├── docs/                           # data_generation.md, data_quality.md,
│                                   # feature_dictionary.md, scoring_methodology.md,
│                                   # api.md, monitoring.md, deployment.md,
│                                   # FRD_acceptance_checklist.md
├── notebooks/                      # 01_exploratory_data_analysis.ipynb (Phase 5)
├── tests/                          # pytest suite, one module per source module
│
├── data/                           # generated artefacts, git-ignored
│   ├── raw/<dataset_version>/
│   ├── processed/
│   └── features/<version>/
│
├── models/
│   └── artifacts/                  # feature pipeline + model files, git-ignored
│
├── reports/
│   ├── data_quality/
│   ├── eda/
│   ├── figures/
│   └── models/
│
└── src/creditguard/
    ├── __init__.py
    ├── config.py                   # single module, NOT a package
    ├── api/
    │   ├── main.py  schemas.py  dependencies.py  middleware.py  errors.py
    │   └── routes/                 # predict.py, applications.py, model.py, health.py
    ├── dashboard/
    │   ├── app.py  api_client.py  state.py
    │   ├── components/             # forms.py, cards.py, charts.py, tables.py
    │   └── pages/                  # 1_Applicant_Scoring.py … 4_Monitoring.py
    ├── data/                       # generator.py, distributions.py, ingest.py,
    │                               # versioning.py
    ├── db/                         # engine.py, models.py, repository.py, init_db.py
    ├── eda/                        # univariate.py, bivariate.py, risk_analysis.py,
    │                               # plots.py, run_eda.py
    ├── explain/                    # shap_explainer.py, reason_codes.py
    ├── features/                   # ratios.py, behavioural.py, encoders.py,
    │                               # pipeline.py, leakage.py, build.py
    ├── models/                     # base.py, logistic.py, random_forest.py,
    │                               # xgboost_model.py, imbalance.py, evaluate.py,
    │                               # calibration.py, threshold.py, tracking.py,
    │                               # registry.py, train.py
    ├── monitoring/                 # baseline.py, drift.py, performance.py,
    │                               # data_quality.py, alerts.py, retraining.py,
    │                               # scheduler.py
    ├── pipeline/                   # orchestrator.py
    ├── scoring/                    # scorecard.py, categories.py,
    │                               # recommendation.py, engine.py
    └── validation/                 # rules.py, engine.py, report.py, cleaning.py,
                                    # cli.py
```

**Layout notes**

- Configuration is split deliberately: `src/creditguard/config.py` reads environment
  variables into a typed `Settings` object; `config/*.yaml` holds tunable domain
  parameters (thresholds, bands, search spaces). Secrets only ever live in the former.
- `data/` and `models/artifacts/` are generated output and are git-ignored.
  Never commit datasets or model binaries.
- Every package directory gets an `__init__.py`.

---

## 4. Hard rules — these apply to every phase

1. **No secrets in code.** All credentials, connection strings and API keys come from
   environment variables or `.env`. `.env` is git-ignored; only `.env.example` is committed.
2. **Type hints and docstrings** on every module, class and public function.
3. **Every phase ships pytest tests.** `pytest` must pass before a phase is done.
   A phase with failing tests is not finished, regardless of how complete the code looks.
4. **Fixed random seed.** `SEED = 42` everywhere randomness occurs — data generation,
   train/test splits, cross-validation, model initialisation, sampling, SHAP background
   samples.
5. **No future information.** No feature may use data dated after the loan application /
   decision date. Point-in-time correctness is mandatory, not aspirational. The forbidden
   feature registry in `features/leakage.py` is enforced at fit and transform time, not
   only in tests.
6. **Never overwrite a trained model.** Retraining always produces a new version;
   previous versions stay registered and loadable, and rollback must remain possible.
7. **Model selection is never based on accuracy alone.** With an imbalanced default rate
   accuracy is misleading. Select on PR-AUC, with ROC-AUC, KS and calibration considered.
8. **Fit on training data only.** Imputers, scalers, encoders and quantile bins are fitted
   inside the pipeline on the training fold, never on the full dataset.
9. **Prefer small, testable modules over large scripts.** If a file is doing three jobs,
   split it.
10. **Stay inside the current phase.** Do not create stub files or scaffolding for later
    phases; each phase's prompt states what is out of scope.

---

## 5. Local development

- Python 3.11 in a `.venv` at the repo root.
- After Phase 1 creates `pyproject.toml`, run `pip install -e .` so the `src/` layout
  resolves at runtime as well as in the editor.
- `docker compose up -d postgres mlflow` for the database and tracking server.
- `pytest -q` before every commit; `ruff check .` and `black --check .` must be clean.
- Commit once per phase, with the phase name in the message.

---

## 6. Build phases

- [x] **Phase 1** — Foundation, configuration and database
- [x] **Phase 2** — Synthetic data generation and ingestion
- [x] **Phase 3** — Data validation and cleaning pipeline
- [x] **Phase 4** — Feature engineering and leakage prevention
- [x] **Phase 5** — Exploratory data analysis
- [ ] **Phase 6** — Model training, imbalance handling and evaluation
- [ ] **Phase 7** — Credit scoring engine, risk categories and explainability
- [ ] **Phase 8** — FastAPI real-time scoring service
- [ ] **Phase 9** — Streamlit dashboard
- [ ] **Phase 10** — Monitoring, drift, retraining, Docker and CI

Phases are delivered one at a time and in order. Do not start a later phase until the
current one is complete and its tests pass. Tick the box above when a phase is finished.
