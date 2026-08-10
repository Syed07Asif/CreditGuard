# CreditGuard API (Phase 8)

**Simulated, educational credit-risk system, not a real lending service.**
All data is synthetic; no output of this API should inform a real credit
decision. See `docs/scoring_methodology.md` for what the score/category/
recommendation actually mean and their limitations.

The API is a thin transport layer over `creditguard.scoring.engine`
(Phase 7): validation, authentication, rate limiting, error handling,
observability and persistence of the request live here. **No scoring
logic lives in this layer** -- every number in a response comes from the
engine, unmodified.

Run locally:

```bash
uvicorn creditguard.api.main:app --host 0.0.0.0 --port 8000
```

Interactive docs (Swagger UI): `http://localhost:8000/docs`. OpenAPI JSON:
`http://localhost:8000/openapi.json`.

## Authentication

Every endpoint except `/health`, `/health/ready` requires an `X-API-Key`
header matching `settings.api_key` (the `API_KEY` environment variable).
Missing or wrong key -> `401`.

```bash
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/v1/model/info
```

## Rate limiting

A simple in-memory, per-API-key, fixed-window-per-minute limiter
(`creditguard.api.dependencies.RateLimiter`), default **100 requests/
minute**, configurable via `RATE_LIMIT_RPM`. Exceeding it returns `429`.
Not distributed -- each `uvicorn` worker process tracks its own counts (see
`docker/Dockerfile.api`'s note on multi-worker deployments).

## CORS

Configured via `CORS_ORIGINS` (comma-separated), default
`http://localhost:8501` (Streamlit's default local port, for Phase 9).

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/v1/predict` | yes | Score one application |
| POST | `/api/v1/predict/batch` | yes | Score up to 1000 applications |
| POST | `/api/v1/explain` | yes | Full per-feature SHAP breakdown for one application |
| POST | `/api/v1/applications` | yes | Persist a customer + loan application, then score it |
| GET | `/api/v1/applications/{loan_id}` | yes | Retrieve a stored application + its latest prediction |
| GET | `/api/v1/predictions` | yes | Paginated, filterable prediction log |
| GET | `/api/v1/model/info` | yes | Active model's id, version, metrics, threshold |
| GET | `/api/v1/model/versions` | yes | Every registered model version, with its scalar metrics |
| GET | `/api/v1/model/performance` | yes | ROC/PR/confusion/calibration/lift-gains/feature-importance for the active model (Phase 9) |
| GET | `/health` | no | Liveness |
| GET | `/health/ready` | no | Readiness: database reachable AND model loaded |
| GET | `/metrics` | yes | Request count, error count, p50/p95/p99 latency |

### `POST /api/v1/predict` vs `POST /api/v1/applications`

`/predict` (and `/explain`) are **stateless what-if scoring**: nothing is
written to `customers`/`loan_applications`/`financial_profiles`/
`credit_history` -- only a `predictions` log row, for `/predict` (not
`/explain`). `customer_id`/`loan_id` are optional; if omitted, the engine
uses a synthetic identifier internally but the response's `loan_id` stays
`null` (it only ever echoes what you actually sent).

`/applications` **persists a real application** -- all four tables, in one
transaction -- then scores it the same way `/predict` does. `customer_id`
is required; `loan_id` is generated if you don't supply one.

### Request schema (`PredictionRequest`)

See `/docs` for the full field list, types and the worked example. Notable
points:

- `credit_utilization` is accepted and range-checked (`[0, 2]`, matching
  the database's own `CHECK` constraint) because `/applications` needs it
  to populate `credit_history.credit_utilization` -- but **the engine never
  uses the submitted value for scoring**. `creditguard.features.ratios.
  RatioFeatures` always recomputes credit utilisation from
  `total_outstanding`/`total_credit_limit` (the single source of truth
  established in Phase 4; see `creditguard.features.leakage`). Submitting
  an inconsistent `credit_utilization` will not change the score.
- Four cross-field checks run after the individual field constraints,
  each producing `422` if violated:
  - `annual_income` must be within 5% of `12 * monthly_income`.
  - `employment_years <= age - 16`.
  - `total_outstanding <= 1.5 * total_credit_limit`.
  - `monthly_expenses + monthly_emi <= 2 * monthly_income`.

### Response shape (`PredictionResponse`)

```json
{
  "request_id": "…",
  "loan_id": null,
  "credit_score": 724,
  "default_probability": 0.0058,
  "risk_category": "LOW",
  "recommendation": "APPROVE",
  "triggered_rules": ["probability (0.0058) < approve threshold …"],
  "top_risk_factors": [
    {"feature": "savings_to_income", "value": 2.0, "impact": 0.3688, "description": "…"}
  ],
  "top_positive_factors": [ ... ],
  "model_id": "logistic_regression_…",
  "model_version": "1.0.0",
  "latency_ms": 42,
  "scored_at": "2026-08-10T07:17:40.340218Z"
}
```

`top_risk_factors`/`top_positive_factors` are the top 5 (configurable via
`config/scoring.yaml`'s `explainability.top_k_factors`) SHAP contributors
in each direction, already aggregated from one-hot-encoded columns back to
their logical source feature and rendered as a human-readable sentence
(`creditguard.explain.reason_codes`). `/explain` returns the same
prediction fields plus **every** logical feature's contribution
(`feature_contributions`, typically 53 entries), not just the top 5.
`/explain`'s `feature_contributions[*].benchmark_median` (Phase 9 addition)
is the training-portfolio median for that feature -- `null` on `/predict`'s
`top_risk_factors`/`top_positive_factors`, which don't carry benchmark
context.

### `GET /api/v1/predictions` segment fields (Phase 9 addition)

`PredictionListItem` also carries `loan_type`, `age`, `annual_income` and
`employment_type` -- echoed from the scored request and persisted onto the
`predictions` row itself (not joined from `customers`/`loan_applications`,
which most predictions -- anonymous `/predict` calls -- have no row in at
all). `NULL` for any prediction logged before this column existed. `GET
/predictions` also accepts a `loan_type` query filter alongside the
existing ones. Added so Phase 9's dashboard can build real segment
breakdowns without the dashboard touching the database directly.

### `GET /api/v1/model/performance` (Phase 9 addition)

Curve/table data `ModelInfoResponse`'s scalar `metrics` doesn't carry: ROC
curve points, PR curve points, a confusion matrix at the model's own
`chosen_threshold`, a calibration/reliability curve, a lift/gains table by
decile, and global feature importance. Computed **once, offline**, by
`python -m creditguard.models.performance` (see that module's docstring)
and persisted onto `model_registry.metrics.performance` -- this endpoint
only reads it back, never recomputes it, so it stays a thin transport
layer even though the underlying computation needs the full test split.
`503` if that backfill hasn't been run yet for the active model.

### `GET /api/v1/applications/{loan_id}`'s `latest_prediction` limitation

The `predictions` table (fixed at Phase 1) doesn't have a column for the
recommendation's rule trace or the model version -- only the decision,
probability, score and SHAP factor breakdowns. So a prediction
*reconstructed from storage* (this endpoint) always has
`triggered_rules: []`; only a fresh `/predict` call returns the real rule
trace. `model_version` is resolved from `model_registry` by the stored
`model_id` (always possible, since a model version is never overwritten --
CLAUDE.md hard rule 6).

## Errors

Every error response has the shape:

```json
{"request_id": "…", "error": "…", "detail": [ ... ] | null}
```

| Status | When |
|---|---|
| 401 | Missing/wrong `X-API-Key` |
| 404 | Unknown `loan_id` (`GET /applications/{loan_id}`) |
| 422 | Request validation failed (field-level `detail`), or a duplicate identifier on `POST /applications` |
| 429 | Rate limit exceeded |
| 503 | The active model/feature pipeline/SHAP explainer isn't ready |
| 500 | Anything unexpected -- logged server-side with a traceback, never echoed to the client |

No error response ever contains a stack trace, a database driver's raw
error text, or any other internal detail -- verified directly in
`tests/test_api_predict.py`/`test_api_validation.py`
(`test_no_error_response_leaks_stack_trace_or_db_error_text`,
`test_unexpected_error_never_leaks_exception_text`, etc.).

## Observability

Every request gets a UUID `request_id` (or reuses an inbound
`X-Request-ID`), returned in the `X-Request-ID` response header and in
every response body (including errors). Structured JSON logs (one line per
request: `request_id`, `method`, `path`, `status_code`, `latency_ms`) go to
stdout -- **never the request payload itself**, per the "don't log
applicant data" rule. `GET /metrics` reports request/error counts and
p50/p95/p99 latency over the most recent 2000 requests
(`creditguard.api.middleware.MetricsStore`).

## Startup and readiness

`main.py`'s `lifespan` eagerly loads the active model, its refit feature
pipeline and SHAP explainer once at startup (10-25s the first time, cached
after that -- see `creditguard.scoring.engine`'s module docstring for why
the pipeline is refit rather than loaded from a stale artifact). A failed
load does **not** stop the app from starting: `/health` stays `200`
either way (liveness), but `/health/ready` reports `503` until a model is
loaded and the database is reachable (readiness) -- the standard
liveness/readiness split for an orchestrator that shouldn't route traffic
to a not-yet-ready instance but also shouldn't restart a container that's
merely still warming up.

## Docker

```bash
docker build -f docker/Dockerfile.api -t creditguard-api .
docker run -p 8000:8000 --env-file .env \
  -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
  creditguard-api
```

`data/` and `models/artifacts/` are generated output (git-ignored, per
CLAUDE.md) and must be mounted, not baked into the image -- see the
Dockerfile's own comments.
