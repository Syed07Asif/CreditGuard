# Deployment

Three ways to run CreditGuard: local (Python + Docker-only Postgres/MLflow, what every
earlier phase's development used), full `docker compose up` (the whole stack), and a
plain environment-variable reference for either.

This is an educational simulation on synthetic data — see the README's disclaimer.
Nothing here is a production deployment guide for a real lending system.

---

## Local development

Already the setup every earlier phase used; unchanged by Phase 10.

```bash
python -m venv .venv
.venv\Scripts\activate           # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
copy .env.example .env           # macOS/Linux: cp .env.example .env
# edit .env: API_KEY at minimum
docker compose up -d postgres mlflow
python -m creditguard.db.init_db
pytest -q
```

Run the API and dashboard directly (no Docker) for iteration:

```bash
uvicorn creditguard.api.main:app --reload --host 127.0.0.1 --port 8000
# in a second terminal, with API_BASE_URL/API_KEY set:
streamlit run src/creditguard/dashboard/app.py --server.port 8501
```

Run the monitoring scheduler once, without waiting for its interval (useful for
checking it works, or for a cron-style external scheduler instead of the container):

```bash
python -m creditguard.monitoring.scheduler --once
```

---

## Full stack: `docker compose up`

```bash
copy .env.example .env      # macOS/Linux: cp .env.example .env
# edit .env: set a real API_KEY at minimum
docker compose up -d
```

This starts five services: `postgres`, `mlflow`, `api`, `dashboard`, `monitoring`.
`api`/`dashboard`/`monitoring` `depends_on` `postgres`/`mlflow` being **healthy**, not
just started — `docker compose up` will wait for the healthchecks before starting the
dependents. Bring-up order:

1. `postgres` becomes healthy (`pg_isready`).
2. `mlflow` becomes healthy (a plain HTTP probe against its own root).
3. `api` starts. Its own healthcheck hits `GET /health` (liveness — always 200 once the
   process is up, regardless of database/model state). `GET /health/ready` (not part of
   the Docker healthcheck, but worth checking by hand) is 503 until **both** the
   database is reachable **and** a model has been loaded — on a completely fresh stack,
   with nothing trained yet, it stays 503 until you train and register a first model
   (below). This is expected, not a bug.
4. `dashboard` starts once `api` is healthy. It never talks to the database or a model
   artifact directly — only to `api` over HTTP.
5. `monitoring` starts once `postgres`/`mlflow` are healthy. Its own first cycle will
   log "no active model registered" and do nothing else until a model exists (below).

At this point the schema exists (the API's own startup applies it — see
`db/init_db.py`) but there is **no data and no trained model yet**.

### Seeding data and training the first model inside the stack

The pipeline orchestrator (FR-021) is the one-command way to do this — run it **inside**
the `api` (or `monitoring`) container, since that's where the package and its
dependencies already live, sharing the same `model_artifacts`/`dataset_data` volumes
the `api`/`monitoring` services mount:

```bash
docker compose exec api python -m creditguard.pipeline.orchestrator run-all \
    --generate --ingest --validate --clean --features --train --register --monitor \
    --n-customers 50000
```

This: generates a synthetic dataset, loads it into Postgres (`--ingest` — so the API's
`POST /applications`-style production traffic and the monitoring checks below have real
customer/loan rows to look at, not just the model artifact), validates and cleans it,
builds the feature pipeline, trains and registers the champion model, builds its
baseline profile, and runs one monitoring cycle. It writes into the `dataset_data` and
`model_artifacts` volumes — both containers see the result immediately, no restart
needed.

Once it finishes:

```bash
curl http://localhost:8000/health/ready   # now {"status": "ready", ...}
```

Open `http://localhost:8501` — the dashboard's sidebar should show 🟢 Connected with
the active model version, and the Applicant Scoring page can score a real application.

### Retraining inside the stack

```bash
docker compose exec api python -m creditguard.monitoring.retraining check
docker compose exec api python -m creditguard.monitoring.retraining trigger \
    --dataset-version <a newer, extended clean dataset_version>
docker compose exec api python -m creditguard.monitoring.retraining rollback \
    --model-id <a previous model_id>
```

After a promotion, restart `api` (or call `POST` to any endpoint that reloads —
currently a restart is the reliable way) so its in-process model cache picks up the
newly active version:

```bash
docker compose restart api
```

---

## Environment variable reference

All configuration is environment-variable driven (`.env`, git-ignored;
`.env.example` documents every variable with safe non-secret defaults). No secret
lives in `docker-compose.yml` — the few values overridden there
(`DB_HOST`/`DB_PORT`/`MLFLOW_TRACKING_URI`/`API_BASE_URL`) are in-network hostnames,
not credentials.

| Variable | Used by | Default | Notes |
|---|---|---|---|
| `DB_HOST`/`DB_PORT`/`DB_NAME`/`DB_USER`/`DB_PASSWORD`/`DB_SCHEMA` | all | `localhost`/`5432`/... | Docker Compose overrides `DB_HOST=postgres`/`DB_PORT=5432` for in-network services; `.env`'s own values stay host-oriented for local, non-Docker development. |
| `MLFLOW_TRACKING_URI` | api, monitoring, training CLIs | `http://localhost:5000` | Compose overrides to `http://mlflow:5000`. |
| `API_KEY` | api (auth), dashboard/monitoring (as a client) | — (required) | The single shared secret; set a real value before `docker compose up`. |
| `ENV` | all | `dev` | `dev`\|`test`\|`prod`. |
| `SEED` | data generation, training | `42` | CLAUDE.md hard rule: fixed everywhere randomness is used. |
| `MODEL_DIR`/`REPORTS_DIR`/`DATA_DIR` | all | `models/artifacts`/`reports`/`data` | Generated output, git-ignored; mounted as Docker volumes for the full stack. |
| `RATE_LIMIT_RPM` | api | `100` | Phase 8. |
| `CORS_ORIGINS` | api | `http://localhost:8501` | Phase 8. |
| `MONITORING_INTERVAL_MINUTES` | monitoring | `60` | How often the scheduler container runs a full monitoring cycle. |
| `ALERT_WEBHOOK_URL` | monitoring (alerts) | unset | Optional; e.g. a Slack incoming webhook. Alerts still go to console + `reports/monitoring/alerts.log` either way. |
| `API_HOST`/`API_PORT`/`API_WORKERS`/`UVICORN_WORKERS` | api container | `0.0.0.0`/`8000`/`1`/`1` | `UVICORN_WORKERS` is what the Dockerfile's `CMD` actually reads. |
| `API_BASE_URL` | dashboard (as an HTTP client) | `http://localhost:8000` | Compose overrides to `http://api:8000`. |

Domain-tunable thresholds (PSI bands, performance-degradation tolerance, retraining
triggers, data-quality violation-rate bands) are **not** environment variables — they
live in `config/monitoring.yaml`, following the same split every earlier phase's
`config/*.yaml` already uses (env vars for secrets/deployment context, YAML for
tunable domain parameters). See `docs/monitoring.md` for what each one means.

---

## Volumes

| Volume | Mounted by | Contents |
|---|---|---|
| `postgres_data` | postgres | The database itself. |
| `mlflow_data` | mlflow | SQLite backend store + artifact root. |
| `model_artifacts` | api, monitoring | Trained model `.joblib` files, feature pipelines, SHAP background samples, baseline profiles — everything under `models/artifacts/`. |
| `dataset_data` | api, monitoring | Generated dataset parquet under `data/` (raw/processed/features) — the API needs it to refit each active model's feature pipeline on its own training data; monitoring needs it for the same reason plus retraining. |
| `reports_data` | api, monitoring | Generated reports (drift reports, alerts log, model cards, ...) under `reports/`. |

`dashboard` mounts none of these — it only ever talks to `api` over HTTP (Phase 9's
rule holds in Docker too).
