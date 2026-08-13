"""Pydantic v2 request/response models for the Phase 8 API.

`PredictionRequest`'s field constraints intentionally mirror (and, where the
brief's field list was incomplete relative to what `creditguard.scoring.
engine.RawApplicationInput` actually needs -- `city_tier` -- extend) the
Phase 7 engine's own input contract, so a request that passes this schema's
validation is guaranteed to also pass the engine's. `credit_utilization` is
accepted and range-checked here (needed to persist a `credit_history` row
in `POST /applications`) but is never fed into the engine as an override --
`creditguard.features.ratios.RatioFeatures` always recomputes it from
`total_outstanding`/`total_credit_limit`, the single source of truth
established in Phase 4 (see `creditguard.features.leakage`).
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# -- constrained enums ------------------------------------------------------


class Gender(StrEnum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class MaritalStatus(StrEnum):
    SINGLE = "SINGLE"
    MARRIED = "MARRIED"
    DIVORCED = "DIVORCED"
    WIDOWED = "WIDOWED"


class Education(StrEnum):
    HIGH_SCHOOL = "HIGH_SCHOOL"
    GRADUATE = "GRADUATE"
    POSTGRADUATE = "POSTGRADUATE"
    DOCTORATE = "DOCTORATE"


class EmploymentType(StrEnum):
    SALARIED = "SALARIED"
    SELF_EMPLOYED = "SELF_EMPLOYED"
    BUSINESS_OWNER = "BUSINESS_OWNER"
    UNEMPLOYED = "UNEMPLOYED"


class LoanType(StrEnum):
    PERSONAL = "PERSONAL"
    HOME = "HOME"
    AUTO = "AUTO"
    EDUCATION = "EDUCATION"
    BUSINESS = "BUSINESS"
    CREDIT_CARD = "CREDIT_CARD"


class RiskCategory(StrEnum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class Recommendation(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


_EXAMPLE_PAYLOAD: dict[str, Any] = {
    "customer_id": "CUST-DEMO-001",
    "age": 35,
    "gender": "MALE",
    "marital_status": "MARRIED",
    "education": "GRADUATE",
    "employment_type": "SALARIED",
    "dependents": 1,
    "employment_years": 8.0,
    "annual_income": 600000.0,
    "monthly_income": 50000.0,
    "city_tier": 1,
    "monthly_expenses": 15000.0,
    "existing_loan_count": 0,
    "existing_loan_amount": 0.0,
    "monthly_emi": 0.0,
    "savings_balance": 100000.0,
    "total_assets": 500000.0,
    "total_liabilities": 0.0,
    "credit_history_months": 60,
    "num_credit_accounts": 3,
    "total_credit_limit": 300000.0,
    "total_outstanding": 50000.0,
    "credit_utilization": 0.17,
    "previous_defaults": 0,
    "late_payments_12m": 0,
    "missed_payments_12m": 0,
    "active_loans": 1,
    "closed_loans": 2,
    "loan_type": "PERSONAL",
    "loan_amount": 200000.0,
    "loan_tenure_months": 36,
    "interest_rate": 12.0,
    "loan_purpose": "DEBT_CONSOLIDATION",
}


class PredictionRequest(BaseModel):
    """One loan application's full point-in-time snapshot, as a client
    (Phase 9's dashboard, or any other caller) would submit it.

    `customer_id`/`loan_id` are optional here: `/predict` and `/explain`
    are stateless what-if scoring (no database row is required to exist or
    is created), so the API layer fills in a synthetic identifier when
    omitted rather than requiring one. `POST /applications` overrides
    `customer_id` to be required, since that endpoint actually persists a
    customer/loan application.
    """

    model_config = ConfigDict(json_schema_extra={"example": _EXAMPLE_PAYLOAD})

    customer_id: str | None = Field(default=None, max_length=64)
    loan_id: str | None = Field(default=None, max_length=64)

    age: int = Field(ge=18, le=100)
    gender: Gender
    marital_status: MaritalStatus
    education: Education
    employment_type: EmploymentType
    dependents: int = Field(ge=0, le=20)
    employment_years: float = Field(ge=0, le=60)
    annual_income: float = Field(gt=0)
    monthly_income: float = Field(gt=0)
    # Not in the Phase 8 brief's literal field list, but required by
    # creditguard.scoring.engine.RawApplicationInput -- one of the 43
    # numeric model features (see docs/feature_dictionary.md).
    city_tier: int = Field(ge=1, le=3)

    monthly_expenses: float = Field(ge=0)
    existing_loan_count: int = Field(ge=0)
    existing_loan_amount: float = Field(ge=0)
    monthly_emi: float = Field(ge=0)
    savings_balance: float = Field(ge=0)
    total_assets: float = Field(ge=0)
    total_liabilities: float = Field(ge=0)

    credit_history_months: int = Field(ge=0)
    num_credit_accounts: int = Field(ge=0)
    total_credit_limit: float = Field(ge=0)
    total_outstanding: float = Field(ge=0)
    credit_utilization: float = Field(ge=0, le=2)
    previous_defaults: int = Field(ge=0)
    late_payments_12m: int = Field(ge=0)
    missed_payments_12m: int = Field(ge=0)
    active_loans: int = Field(ge=0)
    closed_loans: int = Field(ge=0)

    loan_type: LoanType
    loan_amount: float = Field(gt=0)
    loan_tenure_months: int = Field(gt=0)
    interest_rate: float = Field(ge=0, le=100)
    loan_purpose: str = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _check_cross_field_consistency(self) -> PredictionRequest:
        errors: list[str] = []

        implied_annual_income = 12.0 * self.monthly_income
        tolerance = 0.05 * self.annual_income
        if abs(self.annual_income - implied_annual_income) > tolerance:
            errors.append(
                "annual_income is inconsistent with monthly_income (must be "
                "within 5% of 12 * monthly_income)"
            )

        if self.employment_years > self.age - 16:
            errors.append("employment_years cannot exceed age - 16")

        if self.total_outstanding > self.total_credit_limit * 1.5:
            errors.append("total_outstanding cannot exceed 1.5x total_credit_limit")

        if self.monthly_expenses + self.monthly_emi > self.monthly_income * 2:
            errors.append(
                "monthly_expenses + monthly_emi cannot exceed 2x monthly_income"
            )

        if errors:
            raise ValueError("; ".join(errors))
        return self


class ApplicationCreateRequest(PredictionRequest):
    """`PredictionRequest` plus the identifiers `POST /applications` needs
    to actually persist `customers`/`loan_applications`/`financial_profiles`/
    `credit_history` rows.
    """

    customer_id: str = Field(max_length=64)
    application_date: date | None = None


# -- response models ---------------------------------------------------------


class FactorDetail(BaseModel):
    """One feature's contribution to a scored application, in the shape the
    Phase 8 API contract specifies (`{feature, value, impact, description}`)
    -- reshaped from `creditguard.explain.reason_codes.generate_reason_codes`'s
    `{feature, value, contribution, reason}` dicts at the API boundary, not
    recomputed.
    """

    feature: str
    value: Any
    impact: float
    description: str
    # Phase 9 addition, `/explain` only: the training-portfolio median for
    # this feature, so the dashboard can show "applicant vs. portfolio
    # median" without recomputing it (it's never fed by /predict's factor
    # lists, which don't carry benchmark context -- see docs/api.md).
    benchmark_median: float | None = None


class PredictionResponse(BaseModel):
    request_id: str
    loan_id: str | None
    credit_score: int = Field(ge=300, le=900)
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_category: RiskCategory
    recommendation: Recommendation
    triggered_rules: list[str]
    top_risk_factors: list[FactorDetail]
    top_positive_factors: list[FactorDetail]
    model_id: str
    model_version: str
    latency_ms: int
    scored_at: datetime


class ExplainResponse(BaseModel):
    """The detailed SHAP breakdown `POST /api/v1/explain` returns: every
    logical feature's contribution (not just the top-k `PredictionResponse`
    carries), ordered by absolute impact.
    """

    request_id: str
    loan_id: str | None
    credit_score: int = Field(ge=300, le=900)
    default_probability: float = Field(ge=0.0, le=1.0)
    risk_category: RiskCategory
    recommendation: Recommendation
    shap_base_value: float
    feature_contributions: list[FactorDetail]
    model_id: str
    model_version: str
    latency_ms: int
    scored_at: datetime


class ApplicationResponse(BaseModel):
    loan_id: str
    customer_id: str
    loan_type: LoanType
    loan_amount: float
    loan_tenure_months: int
    interest_rate: float
    loan_purpose: str
    application_date: date
    status: str
    latest_prediction: PredictionResponse | None = None


class PredictionListItem(BaseModel):
    prediction_id: int
    loan_id: str
    customer_id: str
    model_id: str
    default_probability: float
    credit_score: int
    risk_category: str
    recommendation: str
    latency_ms: int
    request_source: str
    # Phase 9 addition: NULL for predictions logged before these columns
    # existed on `predictions` (see db/schema.sql). Lets the dashboard
    # segment real prediction traffic without joining to other tables.
    loan_type: str | None = None
    age: int | None = None
    annual_income: float | None = None
    employment_type: str | None = None
    created_at: datetime


class PredictionListResponse(BaseModel):
    items: list[PredictionListItem]
    total: int
    page: int
    page_size: int


class ModelInfoResponse(BaseModel):
    model_id: str
    model_version: str
    algorithm: str
    training_date: datetime
    dataset_version: str
    feature_count: int
    metrics: dict[str, float]
    chosen_threshold: float
    is_active: bool


class ModelVersionSummary(BaseModel):
    model_id: str
    model_version: str
    algorithm: str
    training_date: datetime
    is_active: bool
    metrics: dict[str, float] = Field(default_factory=dict)


class ModelVersionsResponse(BaseModel):
    versions: list[ModelVersionSummary]


class ConfusionMatrix(BaseModel):
    tn: int
    fp: int
    fn: int
    tp: int


class LiftGainsRow(BaseModel):
    decile: int
    n: int
    n_positive: int
    default_rate: float
    cum_n: int
    cum_positive: int
    cum_gain: float
    lift: float
    cum_lift: float


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class ModelPerformanceResponse(BaseModel):
    """ROC/PR/confusion/calibration/lift-gains/feature-importance data for
    one model, computed once by `creditguard.models.performance`'s backfill
    (see that module's docstring) and persisted onto the model's own
    `model_registry.metrics` row -- this endpoint only reads it back.
    """

    model_id: str
    model_version: str
    roc_curve: dict[str, list[float]]
    pr_curve: dict[str, list[float]]
    confusion_matrix: ConfusionMatrix
    calibration_curve: dict[str, list[float]]
    lift_gains: list[LiftGainsRow]
    feature_importance: list[FeatureImportanceItem]


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: bool
    model_loaded: bool


class MetricsResponse(BaseModel):
    request_count: int
    error_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class ErrorDetailItem(BaseModel):
    loc: list[str] = Field(default_factory=list)
    msg: str
    type: str


class ErrorResponse(BaseModel):
    request_id: str | None = None
    error: str
    detail: list[ErrorDetailItem] | None = None


# -- Phase 10: monitoring -----------------------------------------------------
# Every response below is read back from drift_reports/monitoring_metrics/
# data_quality_issues -- rows already written by monitoring/scheduler.py's
# periodic cycle or the pipeline orchestrator's --monitor stage -- never
# recomputed on request, the same "thin transport layer" rule
# ModelPerformanceResponse above already follows.


class DriftFindingItem(BaseModel):
    feature_name: str
    method: str
    baseline_stat: float
    current_stat: float
    drift_score: float
    status: str
    created_at: datetime


class MonitoringDriftResponse(BaseModel):
    model_id: str
    findings: list[DriftFindingItem]
    n_ok: int
    n_warning: int
    n_drift: int
    latest_run_at: datetime | None = None


class MonitoringMetricItem(BaseModel):
    metric_name: str
    metric_value: float
    window_start: date
    window_end: date
    created_at: datetime


class MonitoringPerformanceResponse(BaseModel):
    model_id: str
    metrics: list[MonitoringMetricItem]


class DataQualityTrendItem(BaseModel):
    rule_name: str
    severity: str
    day: date
    n: int


class MonitoringDataQualityResponse(BaseModel):
    trend: list[DataQualityTrendItem]
    total_issues: int
