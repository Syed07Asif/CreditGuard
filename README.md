# CreditGuard

CreditGuard is an end-to-end credit risk scoring and monitoring platform that estimates
an applicant's probability of default within 12 months and turns it into a credit score,
a risk category, a lending recommendation and a plain-English explanation.

**This is a portfolio / educational simulation, not a production lending system.**
All customer data is synthetic; no real applicant, bureau or banking data is used.
Nothing here is validated for real lending decisions.

This README covers **Phase 1: Foundation, configuration and database**. It will be
rewritten as later phases (data generation, feature engineering, modelling, the API,
the dashboard and monitoring) are built out.

## Stack

Python 3.11, PostgreSQL, SQLAlchemy 2.0, pydantic-settings, pandas, MLflow, Docker.
See [CLAUDE.md](CLAUDE.md) for the full stack and repository layout.

## Setup

1. **Create a virtual environment and install the package in editable mode:**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   make install
   ```

2. **Copy the environment template and fill in local values:**

   ```bash
   copy .env.example .env
   ```

3. **Start PostgreSQL and MLflow:**

   ```bash
   make db-up
   ```

4. **Initialise the database schema** (safe to re-run; it is idempotent):

   ```bash
   make db-init
   ```

## Development

```bash
make test      # run pytest
make lint      # ruff check
make format    # black .
```

Tests run against a real PostgreSQL database (`creditguard_test` by default) and apply
`db/schema.sql` automatically via a session fixture before the first test runs.

## Configuration

All configuration is environment-variable driven and validated at startup by
`creditguard.config.Settings` (see `.env.example` for the full list). Missing required
variables raise a `ConfigurationError` with a clear message rather than failing silently.

## Database schema

`db/schema.sql` defines the full CreditGuard schema: customers, loan applications,
financial profiles, credit history, loan outcomes (kept separate from applications to
prevent label leakage into features), the model registry, predictions, data quality
issues, and monitoring/drift tables. `src/creditguard/db/models.py` mirrors the schema
as SQLAlchemy 2.0 declarative models, and `src/creditguard/db/repository.py` provides a
thin repository per table for inserts, lookups, filtered/raw-SQL reads (as pandas
DataFrames), and upserts.

## Out of scope for this phase

No data generation, no feature code, no models, no API, no dashboard. These are covered
in later phases — see the build-phase checklist in [CLAUDE.md](CLAUDE.md).
