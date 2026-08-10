"""`POST /applications` (persist a customer + loan application, then score
it) and `GET /applications/{loan_id}` (retrieve a stored application and
its latest prediction).

Unlike `/predict`, this endpoint actually writes `customers`/
`loan_applications`/`financial_profiles`/`credit_history` rows -- all four
inserts share one database transaction (`creditguard.db.engine.
get_session`'s commit-on-success/rollback-on-error), so a failure partway
through (e.g. a duplicate `customer_id`) never leaves an orphaned partial
application.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from fastapi import APIRouter, Depends, Request
from sqlalchemy import insert

from creditguard.api.dependencies import enforce_rate_limit
from creditguard.api.errors import LoanNotFoundError
from creditguard.api.routes.predict import (
    build_engine_input,
    build_prediction_response,
    to_factor_details,
)
from creditguard.api.schemas import (
    ApplicationCreateRequest,
    ApplicationResponse,
    PredictionResponse,
)
from creditguard.db.engine import get_session
from creditguard.db.models import (
    CreditHistory,
    Customer,
    FinancialProfile,
    LoanApplication,
)
from creditguard.db.repository import (
    LoanApplicationRepository,
    ModelRegistryRepository,
    PredictionRepository,
)
from creditguard.scoring import engine

router = APIRouter(tags=["applications"], dependencies=[Depends(enforce_rate_limit)])


def _persist_application(
    payload: ApplicationCreateRequest, loan_id: str, application_date: date
) -> None:
    """Insert the customer, loan application, financial-profile and
    credit-history rows implied by `payload`, all in one transaction.
    """
    data = payload.model_dump(mode="json")
    with get_session() as session:
        session.execute(
            insert(Customer),
            [
                {
                    "customer_id": payload.customer_id,
                    "age": data["age"],
                    "gender": data["gender"],
                    "marital_status": data["marital_status"],
                    "dependents": data["dependents"],
                    "education": data["education"],
                    "employment_type": data["employment_type"],
                    "employment_years": data["employment_years"],
                    "annual_income": data["annual_income"],
                    "monthly_income": data["monthly_income"],
                    "city_tier": data["city_tier"],
                }
            ],
        )
        session.execute(
            insert(LoanApplication),
            [
                {
                    "loan_id": loan_id,
                    "customer_id": payload.customer_id,
                    "loan_type": data["loan_type"],
                    "loan_amount": data["loan_amount"],
                    "loan_tenure_months": data["loan_tenure_months"],
                    "interest_rate": data["interest_rate"],
                    "loan_purpose": data["loan_purpose"],
                    "application_date": application_date,
                    "status": "PENDING",
                }
            ],
        )
        session.execute(
            insert(FinancialProfile),
            [
                {
                    "customer_id": payload.customer_id,
                    "as_of_date": application_date,
                    "monthly_income": data["monthly_income"],
                    "monthly_expenses": data["monthly_expenses"],
                    "existing_loan_count": data["existing_loan_count"],
                    "existing_loan_amount": data["existing_loan_amount"],
                    "monthly_emi": data["monthly_emi"],
                    "savings_balance": data["savings_balance"],
                    "total_assets": data["total_assets"],
                    "total_liabilities": data["total_liabilities"],
                }
            ],
        )
        session.execute(
            insert(CreditHistory),
            [
                {
                    "customer_id": payload.customer_id,
                    "as_of_date": application_date,
                    "credit_history_months": data["credit_history_months"],
                    "num_credit_accounts": data["num_credit_accounts"],
                    "total_credit_limit": data["total_credit_limit"],
                    "total_outstanding": data["total_outstanding"],
                    "credit_utilization": data["credit_utilization"],
                    "previous_defaults": data["previous_defaults"],
                    "late_payments_12m": data["late_payments_12m"],
                    "missed_payments_12m": data["missed_payments_12m"],
                    "active_loans": data["active_loans"],
                    "closed_loans": data["closed_loans"],
                }
            ],
        )


@router.post("/applications", response_model=ApplicationResponse, status_code=201)
def create_application(
    payload: ApplicationCreateRequest, request: Request
) -> ApplicationResponse:
    """Persist a customer + loan application (and its financial/credit
    snapshot), then score it.
    """
    loan_id = payload.loan_id or f"LOAN-{uuid.uuid4().hex[:12]}"
    application_date = payload.application_date or date.today()
    _persist_application(payload, loan_id, application_date)

    raw_input = build_engine_input(payload)
    raw_input["loan_id"] = loan_id
    raw_input["customer_id"] = payload.customer_id
    result = engine.score_application(raw_input, persist=True, request_source="api")
    prediction_response = build_prediction_response(
        result, request.state.request_id, loan_id
    )

    return ApplicationResponse(
        loan_id=loan_id,
        customer_id=payload.customer_id,
        loan_type=payload.loan_type,
        loan_amount=payload.loan_amount,
        loan_tenure_months=payload.loan_tenure_months,
        interest_rate=payload.interest_rate,
        loan_purpose=payload.loan_purpose,
        application_date=application_date,
        status="PENDING",
        latest_prediction=prediction_response,
    )


def _latest_prediction_response(
    loan_id: str, request_id: str
) -> PredictionResponse | None:
    """The most recent `predictions` row for `loan_id`, reshaped into a
    `PredictionResponse` -- or `None` if this loan has never been scored.

    `triggered_rules` comes back empty: the `predictions` table doesn't
    persist the recommendation's rule trace (only the decision and the
    SHAP factor breakdowns), so a *stored* prediction can't reconstruct it
    -- only a fresh `/predict` call can. `model_version` is looked up from
    `model_registry` by the stored `model_id` (never overwritten -- CLAUDE.md
    hard rule 6 -- so this is always resolvable for any prediction ever made).
    """
    predictions = PredictionRepository().fetch_dataframe(filters={"loan_id": loan_id})
    if predictions.empty:
        return None

    row: dict[str, Any] = (
        predictions.sort_values("created_at", ascending=False).iloc[0].to_dict()
    )
    model_row = ModelRegistryRepository().get_by_id(row["model_id"])
    model_version = model_row["model_version"] if model_row is not None else ""

    return PredictionResponse(
        request_id=request_id,
        loan_id=row["loan_id"],
        credit_score=int(row["credit_score"]),
        default_probability=round(float(row["default_probability"]), 4),
        risk_category=row["risk_category"],
        recommendation=row["recommendation"],
        triggered_rules=[],
        top_risk_factors=to_factor_details(row["top_risk_factors"] or []),
        top_positive_factors=to_factor_details(row["top_positive_factors"] or []),
        model_id=row["model_id"],
        model_version=model_version,
        latency_ms=int(row["latency_ms"]),
        scored_at=row["created_at"],
    )


@router.get("/applications/{loan_id}", response_model=ApplicationResponse)
def get_application(loan_id: str, request: Request) -> ApplicationResponse:
    """A stored application and its latest prediction (if any)."""
    application = LoanApplicationRepository().get_by_id(loan_id)
    if application is None:
        raise LoanNotFoundError(loan_id)

    return ApplicationResponse(
        loan_id=application["loan_id"],
        customer_id=application["customer_id"],
        loan_type=application["loan_type"],
        loan_amount=float(application["loan_amount"]),
        loan_tenure_months=int(application["loan_tenure_months"]),
        interest_rate=float(application["interest_rate"]),
        loan_purpose=application["loan_purpose"],
        application_date=application["application_date"],
        status=application["status"],
        latest_prediction=_latest_prediction_response(
            loan_id, request.state.request_id
        ),
    )
