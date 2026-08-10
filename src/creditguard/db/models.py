"""SQLAlchemy 2.0 declarative models mirroring db/schema.sql exactly."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all CreditGuard ORM models."""


class Customer(Base):
    """A loan applicant's demographic and employment profile."""

    __tablename__ = "customers"
    __table_args__ = (
        CheckConstraint("age BETWEEN 18 AND 100", name="ck_customers_age"),
        CheckConstraint("dependents >= 0", name="ck_customers_dependents"),
        CheckConstraint("employment_years >= 0", name="ck_customers_employment_years"),
        CheckConstraint("annual_income >= 0", name="ck_customers_annual_income"),
        CheckConstraint("monthly_income >= 0", name="ck_customers_monthly_income"),
    )

    customer_id: Mapped[str] = mapped_column(Text, primary_key=True)
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(Text)
    marital_status: Mapped[str] = mapped_column(Text)
    dependents: Mapped[int] = mapped_column(Integer)
    education: Mapped[str] = mapped_column(Text)
    employment_type: Mapped[str] = mapped_column(Text)
    employment_years: Mapped[float] = mapped_column(Numeric)
    annual_income: Mapped[float] = mapped_column(Numeric)
    monthly_income: Mapped[float] = mapped_column(Numeric)
    city_tier: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )


class LoanApplication(Base):
    """A single loan application submitted by a customer."""

    __tablename__ = "loan_applications"
    __table_args__ = (
        CheckConstraint(
            "loan_type IN "
            "('PERSONAL','HOME','AUTO','EDUCATION','BUSINESS','CREDIT_CARD')",
            name="ck_loan_applications_loan_type",
        ),
        CheckConstraint("loan_amount > 0", name="ck_loan_applications_loan_amount"),
        CheckConstraint(
            "loan_tenure_months > 0", name="ck_loan_applications_loan_tenure_months"
        ),
        CheckConstraint(
            "interest_rate >= 0", name="ck_loan_applications_interest_rate"
        ),
    )

    loan_id: Mapped[str] = mapped_column(Text, primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    loan_type: Mapped[str] = mapped_column(Text)
    loan_amount: Mapped[float] = mapped_column(Numeric)
    loan_tenure_months: Mapped[int] = mapped_column(Integer)
    interest_rate: Mapped[float] = mapped_column(Numeric)
    loan_purpose: Mapped[str] = mapped_column(Text)
    application_date: Mapped[date] = mapped_column(Date)
    decision_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )


class FinancialProfile(Base):
    """A point-in-time snapshot of a customer's income, expenses and balance sheet."""

    __tablename__ = "financial_profiles"
    __table_args__ = (UniqueConstraint("customer_id", "as_of_date"),)

    profile_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    as_of_date: Mapped[date] = mapped_column(Date)
    monthly_income: Mapped[float] = mapped_column(Numeric)
    monthly_expenses: Mapped[float] = mapped_column(Numeric)
    existing_loan_count: Mapped[int] = mapped_column(Integer)
    existing_loan_amount: Mapped[float] = mapped_column(Numeric)
    monthly_emi: Mapped[float] = mapped_column(Numeric)
    savings_balance: Mapped[float] = mapped_column(Numeric)
    total_assets: Mapped[float] = mapped_column(Numeric)
    total_liabilities: Mapped[float] = mapped_column(Numeric)


class CreditHistory(Base):
    """A point-in-time snapshot of a customer's bureau credit history."""

    __tablename__ = "credit_history"
    __table_args__ = (
        CheckConstraint(
            "credit_utilization BETWEEN 0 AND 2",
            name="ck_credit_history_credit_utilization",
        ),
        UniqueConstraint("customer_id", "as_of_date"),
    )

    credit_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    as_of_date: Mapped[date] = mapped_column(Date)
    credit_history_months: Mapped[int] = mapped_column(Integer)
    num_credit_accounts: Mapped[int] = mapped_column(Integer)
    total_credit_limit: Mapped[float] = mapped_column(Numeric)
    total_outstanding: Mapped[float] = mapped_column(Numeric)
    credit_utilization: Mapped[float] = mapped_column(Numeric)
    previous_defaults: Mapped[int] = mapped_column(Integer)
    late_payments_12m: Mapped[int] = mapped_column(Integer)
    missed_payments_12m: Mapped[int] = mapped_column(Integer)
    active_loans: Mapped[int] = mapped_column(Integer)
    closed_loans: Mapped[int] = mapped_column(Integer)


class LoanOutcome(Base):
    """The observed 12-month default label for a loan.

    Deliberately separate from LoanApplication so that feature code cannot
    accidentally join outcome information into the feature set.
    """

    __tablename__ = "loan_outcomes"
    __table_args__ = (
        CheckConstraint("default_12m IN (0, 1)", name="ck_loan_outcomes_default_12m"),
    )

    loan_id: Mapped[str] = mapped_column(
        ForeignKey("loan_applications.loan_id"), primary_key=True
    )
    default_12m: Mapped[int] = mapped_column(Integer)
    outcome_observed_date: Mapped[date] = mapped_column(Date)


class ModelRegistry(Base):
    """A registered, trained model version and its training metadata."""

    __tablename__ = "model_registry"

    model_id: Mapped[str] = mapped_column(Text, primary_key=True)
    model_version: Mapped[str] = mapped_column(Text)
    algorithm: Mapped[str] = mapped_column(Text)
    training_date: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True))
    dataset_version: Mapped[str] = mapped_column(Text)
    feature_list: Mapped[Any] = mapped_column(JSONB)
    hyperparameters: Mapped[Any] = mapped_column(JSONB)
    metrics: Mapped[Any] = mapped_column(JSONB)
    mlflow_run_id: Mapped[str] = mapped_column(Text)
    artifact_path: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, server_default=text("false"))


class Prediction(Base):
    """A single model prediction logged for a loan application."""

    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint(
            "credit_score BETWEEN 300 AND 900", name="ck_predictions_credit_score"
        ),
        CheckConstraint(
            "recommendation IN ('APPROVE','REVIEW','REJECT')",
            name="ck_predictions_recommendation",
        ),
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    loan_id: Mapped[str] = mapped_column(Text)
    customer_id: Mapped[str] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(ForeignKey("model_registry.model_id"))
    default_probability: Mapped[float] = mapped_column(Numeric)
    credit_score: Mapped[int] = mapped_column(Integer)
    risk_category: Mapped[str] = mapped_column(Text)
    recommendation: Mapped[str] = mapped_column(Text)
    top_risk_factors: Mapped[Any] = mapped_column(JSONB)
    top_positive_factors: Mapped[Any] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(Integer)
    request_source: Mapped[str] = mapped_column(Text)
    # Nullable: NULL for predictions logged before Phase 9 added these
    # columns for dashboard segment analysis (see db/schema.sql).
    loan_type: Mapped[str | None] = mapped_column(Text)
    age: Mapped[int | None] = mapped_column(Integer)
    annual_income: Mapped[float | None] = mapped_column(Numeric)
    employment_type: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )


class DataQualityIssue(Base):
    """A single data quality rule violation detected on a source table."""

    __tablename__ = "data_quality_issues"

    issue_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    table_name: Mapped[str] = mapped_column(Text)
    record_key: Mapped[str] = mapped_column(Text)
    rule_name: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(Text)
    detail: Mapped[str] = mapped_column(Text)
    detected_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )


class MonitoringMetric(Base):
    """A single monitoring metric value computed for a model over a time window."""

    __tablename__ = "monitoring_metrics"

    metric_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    model_id: Mapped[str] = mapped_column(Text)
    metric_name: Mapped[str] = mapped_column(Text)
    metric_value: Mapped[float] = mapped_column(Numeric)
    window_start: Mapped[date] = mapped_column(Date)
    window_end: Mapped[date] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )


class DriftReport(Base):
    """A single feature drift measurement comparing a baseline to a current window."""

    __tablename__ = "drift_reports"
    __table_args__ = (
        CheckConstraint(
            "status IN ('OK','WARNING','DRIFT')", name="ck_drift_reports_status"
        ),
    )

    report_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    model_id: Mapped[str] = mapped_column(Text)
    feature_name: Mapped[str] = mapped_column(Text)
    method: Mapped[str] = mapped_column(Text)
    baseline_stat: Mapped[float] = mapped_column(Numeric)
    current_stat: Mapped[float] = mapped_column(Numeric)
    drift_score: Mapped[float] = mapped_column(Numeric)
    status: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=text("now()")
    )
