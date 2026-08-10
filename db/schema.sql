-- CreditGuard database schema.
-- Idempotent: safe to re-run against an already-initialised database.

CREATE TABLE IF NOT EXISTS customers (
    customer_id        TEXT PRIMARY KEY,
    age                INT NOT NULL CHECK (age BETWEEN 18 AND 100),
    gender              TEXT NOT NULL,
    marital_status      TEXT NOT NULL,
    dependents          INT NOT NULL CHECK (dependents >= 0),
    education           TEXT NOT NULL,
    employment_type     TEXT NOT NULL,
    employment_years    NUMERIC NOT NULL CHECK (employment_years >= 0),
    annual_income       NUMERIC NOT NULL CHECK (annual_income >= 0),
    monthly_income      NUMERIC NOT NULL CHECK (monthly_income >= 0),
    city_tier           INT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS loan_applications (
    loan_id             TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers (customer_id),
    loan_type           TEXT NOT NULL CHECK (loan_type IN (
                            'PERSONAL', 'HOME', 'AUTO', 'EDUCATION',
                            'BUSINESS', 'CREDIT_CARD'
                         )),
    loan_amount         NUMERIC NOT NULL CHECK (loan_amount > 0),
    loan_tenure_months  INT NOT NULL CHECK (loan_tenure_months > 0),
    interest_rate       NUMERIC NOT NULL CHECK (interest_rate >= 0),
    loan_purpose        TEXT NOT NULL,
    application_date    DATE NOT NULL,
    decision_date       DATE,
    status               TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_loan_applications_customer_id
    ON loan_applications (customer_id);
CREATE INDEX IF NOT EXISTS ix_loan_applications_application_date
    ON loan_applications (application_date);
CREATE INDEX IF NOT EXISTS ix_loan_applications_decision_date
    ON loan_applications (decision_date);

CREATE TABLE IF NOT EXISTS financial_profiles (
    profile_id           SERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers (customer_id),
    as_of_date           DATE NOT NULL,
    monthly_income       NUMERIC NOT NULL,
    monthly_expenses     NUMERIC NOT NULL,
    existing_loan_count  INT NOT NULL,
    existing_loan_amount NUMERIC NOT NULL,
    monthly_emi          NUMERIC NOT NULL,
    savings_balance      NUMERIC NOT NULL,
    total_assets         NUMERIC NOT NULL,
    total_liabilities    NUMERIC NOT NULL,
    UNIQUE (customer_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS ix_financial_profiles_customer_id
    ON financial_profiles (customer_id);
CREATE INDEX IF NOT EXISTS ix_financial_profiles_as_of_date
    ON financial_profiles (as_of_date);

CREATE TABLE IF NOT EXISTS credit_history (
    credit_id            SERIAL PRIMARY KEY,
    customer_id          TEXT NOT NULL REFERENCES customers (customer_id),
    as_of_date           DATE NOT NULL,
    credit_history_months INT NOT NULL,
    num_credit_accounts  INT NOT NULL,
    total_credit_limit   NUMERIC NOT NULL,
    total_outstanding    NUMERIC NOT NULL,
    credit_utilization   NUMERIC NOT NULL CHECK (credit_utilization BETWEEN 0 AND 2),
    previous_defaults    INT NOT NULL,
    late_payments_12m    INT NOT NULL,
    missed_payments_12m  INT NOT NULL,
    active_loans         INT NOT NULL,
    closed_loans         INT NOT NULL,
    UNIQUE (customer_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS ix_credit_history_customer_id
    ON credit_history (customer_id);
CREATE INDEX IF NOT EXISTS ix_credit_history_as_of_date
    ON credit_history (as_of_date);

-- Label table, deliberately separate from loan_applications so that feature
-- code cannot accidentally join outcome information into the feature set.
CREATE TABLE IF NOT EXISTS loan_outcomes (
    loan_id              TEXT PRIMARY KEY REFERENCES loan_applications (loan_id),
    default_12m          INT NOT NULL CHECK (default_12m IN (0, 1)),
    outcome_observed_date DATE NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_loan_outcomes_outcome_observed_date
    ON loan_outcomes (outcome_observed_date);

CREATE TABLE IF NOT EXISTS model_registry (
    model_id             TEXT PRIMARY KEY,
    model_version        TEXT NOT NULL,
    algorithm            TEXT NOT NULL,
    training_date        TIMESTAMPTZ NOT NULL,
    dataset_version      TEXT NOT NULL,
    feature_list         JSONB NOT NULL,
    hyperparameters       JSONB NOT NULL,
    metrics               JSONB NOT NULL,
    mlflow_run_id        TEXT NOT NULL,
    artifact_path         TEXT NOT NULL,
    is_active             BOOLEAN NOT NULL DEFAULT false
);

CREATE INDEX IF NOT EXISTS ix_model_registry_training_date
    ON model_registry (training_date);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         SERIAL PRIMARY KEY,
    loan_id               TEXT NOT NULL,
    customer_id           TEXT NOT NULL,
    model_id              TEXT NOT NULL REFERENCES model_registry (model_id),
    default_probability   NUMERIC NOT NULL,
    credit_score           INT NOT NULL CHECK (credit_score BETWEEN 300 AND 900),
    risk_category          TEXT NOT NULL,
    recommendation         TEXT NOT NULL CHECK (recommendation IN ('APPROVE', 'REVIEW', 'REJECT')),
    top_risk_factors       JSONB NOT NULL,
    top_positive_factors   JSONB NOT NULL,
    latency_ms             INT NOT NULL,
    request_source         TEXT NOT NULL,
    -- Nullable: added in Phase 9 so the dashboard's Portfolio Analytics page
    -- can segment real prediction traffic by loan type / age / income /
    -- employment type without joining to customers/loan_applications (which
    -- most predictions -- anonymous /predict calls -- have no row in at
    -- all). NULL for any prediction logged before this column existed.
    loan_type              TEXT,
    age                    INT,
    annual_income          NUMERIC,
    employment_type        TEXT,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Idempotent patch for a predictions table created before the Phase 9
-- columns above existed (CREATE TABLE IF NOT EXISTS above is a no-op on an
-- already-existing table, so these are needed to bring it up to date too).
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS loan_type TEXT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS age INT;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS annual_income NUMERIC;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS employment_type TEXT;

CREATE INDEX IF NOT EXISTS ix_predictions_model_id
    ON predictions (model_id);
CREATE INDEX IF NOT EXISTS ix_predictions_created_at
    ON predictions (created_at);
CREATE INDEX IF NOT EXISTS ix_predictions_loan_type
    ON predictions (loan_type);

CREATE TABLE IF NOT EXISTS data_quality_issues (
    issue_id      SERIAL PRIMARY KEY,
    table_name    TEXT NOT NULL,
    record_key    TEXT NOT NULL,
    rule_name     TEXT NOT NULL,
    severity      TEXT NOT NULL,
    detail        TEXT NOT NULL,
    detected_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_data_quality_issues_detected_at
    ON data_quality_issues (detected_at);

CREATE TABLE IF NOT EXISTS monitoring_metrics (
    metric_id     SERIAL PRIMARY KEY,
    model_id      TEXT NOT NULL,
    metric_name   TEXT NOT NULL,
    metric_value  NUMERIC NOT NULL,
    window_start  DATE NOT NULL,
    window_end    DATE NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_monitoring_metrics_window_start
    ON monitoring_metrics (window_start);
CREATE INDEX IF NOT EXISTS ix_monitoring_metrics_window_end
    ON monitoring_metrics (window_end);

CREATE TABLE IF NOT EXISTS drift_reports (
    report_id       SERIAL PRIMARY KEY,
    model_id        TEXT NOT NULL,
    feature_name    TEXT NOT NULL,
    method          TEXT NOT NULL,
    baseline_stat   NUMERIC NOT NULL,
    current_stat    NUMERIC NOT NULL,
    drift_score     NUMERIC NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('OK', 'WARNING', 'DRIFT')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_drift_reports_created_at
    ON drift_reports (created_at);
