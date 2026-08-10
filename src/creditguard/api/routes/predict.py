"""`POST /predict`, `POST /predict/batch`, `POST /explain`, `GET /predictions`.

Every handler here is a thin transport layer: shape the validated request
into the dict `creditguard.scoring.engine` expects, call the engine (or a
repository query), reshape the result into a response schema. No scoring
logic lives here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request

from creditguard.api.dependencies import enforce_rate_limit
from creditguard.api.schemas import (
    ExplainResponse,
    FactorDetail,
    PredictionListItem,
    PredictionListResponse,
    PredictionRequest,
    PredictionResponse,
)
from creditguard.db.repository import PredictionRepository
from creditguard.explain.reason_codes import render_reason
from creditguard.scoring import engine
from creditguard.scoring.engine import ScoringResult

router = APIRouter(tags=["predict"], dependencies=[Depends(enforce_rate_limit)])


def build_engine_input(payload: PredictionRequest) -> dict[str, Any]:
    """Shape a validated API request into the flat dict
    `creditguard.scoring.engine.score_application` expects.

    `credit_utilization` is deliberately dropped: the engine's feature
    pipeline always recomputes it from `total_outstanding`/
    `total_credit_limit` (Phase 4's single source of truth, see
    `creditguard.features.leakage`) -- passing the client-submitted value
    through would be a silent no-op at best.

    `mode="json"` (rather than the default `mode="python"`) so enum fields
    serialize to their plain string values -- what
    `creditguard.scoring.engine.RawApplicationInput`'s `Literal[...]`
    fields expect.
    """
    data = payload.model_dump(mode="json", exclude={"credit_utilization"})
    if not data.get("customer_id"):
        data["customer_id"] = f"ANON-{uuid.uuid4().hex[:12]}"
    return data


def to_factor_details(rows: list[dict[str, Any]]) -> list[FactorDetail]:
    return [
        FactorDetail(
            feature=row["feature"],
            value=row.get("value"),
            impact=row["contribution"],
            description=row["reason"],
        )
        for row in rows
    ]


def build_prediction_response(
    result: ScoringResult, request_id: str, echo_loan_id: str | None
) -> PredictionResponse:
    """`echo_loan_id` is the *client-supplied* `loan_id` (possibly `None`),
    not `result.loan_id` -- the engine always assigns a synthetic one
    internally for its own `predictions` logging even when the caller
    didn't provide one, but the API contract's `loan_id` is nullable
    specifically to reflect "no loan_id was given," not to leak that
    internal placeholder.
    """
    return PredictionResponse(
        request_id=request_id,
        loan_id=echo_loan_id,
        credit_score=result.credit_score,
        default_probability=round(result.default_probability, 4),
        risk_category=result.risk_category,
        recommendation=result.recommendation,
        triggered_rules=result.triggered_rules,
        top_risk_factors=to_factor_details(result.top_risk_factors),
        top_positive_factors=to_factor_details(result.top_positive_factors),
        model_id=result.model_id,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        scored_at=result.scored_at,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Score one application and persist the prediction."""
    result = engine.score_application(
        build_engine_input(payload), persist=True, request_source="api"
    )
    return build_prediction_response(result, request.state.request_id, payload.loan_id)


@router.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(
    payload: Annotated[list[PredictionRequest], Body(min_length=1, max_length=1000)],
    request: Request,
) -> list[PredictionResponse]:
    """Score up to 1000 applications, returned in the original order."""
    return [
        build_prediction_response(
            engine.score_application(
                build_engine_input(item), persist=True, request_source="api"
            ),
            request.state.request_id,
            item.loan_id,
        )
        for item in payload
    ]


@router.post("/explain", response_model=ExplainResponse)
def explain(payload: PredictionRequest, request: Request) -> ExplainResponse:
    """Full per-feature SHAP breakdown for one application. Does not
    persist a `predictions` row (see `creditguard.scoring.engine.
    explain_application`).
    """
    detailed = engine.explain_application(build_engine_input(payload))
    ranked = sorted(
        detailed.contributions_by_feature.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    feature_contributions = [
        FactorDetail(
            feature=feature,
            value=detailed.raw_features.get(feature),
            impact=contribution,
            description=render_reason(
                feature,
                detailed.raw_features.get(feature),
                contribution,
                detailed.benchmarks.get(feature, {}),
            ),
            benchmark_median=detailed.benchmarks.get(feature, {}).get("median"),
        )
        for feature, contribution in ranked
    ]
    result = detailed.result
    return ExplainResponse(
        request_id=request.state.request_id,
        loan_id=payload.loan_id,
        credit_score=result.credit_score,
        default_probability=round(result.default_probability, 4),
        risk_category=result.risk_category,
        recommendation=result.recommendation,
        shap_base_value=detailed.shap_base_value,
        feature_contributions=feature_contributions,
        model_id=result.model_id,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
        scored_at=result.scored_at,
    )


@router.get("/predictions", response_model=PredictionListResponse)
def list_predictions(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    date_from: Annotated[datetime | None, Query()] = None,
    date_to: Annotated[datetime | None, Query()] = None,
    risk_category: Annotated[str | None, Query()] = None,
    recommendation: Annotated[str | None, Query()] = None,
    model_id: Annotated[str | None, Query()] = None,
    loan_type: Annotated[str | None, Query()] = None,
) -> PredictionListResponse:
    """Paginated, filterable prediction log, newest first."""
    rows, total = PredictionRepository().query_predictions(
        date_from=date_from,
        date_to=date_to,
        risk_category=risk_category,
        recommendation=recommendation,
        model_id=model_id,
        loan_type=loan_type,
        page=page,
        page_size=page_size,
    )
    items = [
        PredictionListItem(
            prediction_id=row["prediction_id"],
            loan_id=row["loan_id"],
            customer_id=row["customer_id"],
            model_id=row["model_id"],
            default_probability=float(row["default_probability"]),
            credit_score=int(row["credit_score"]),
            risk_category=row["risk_category"],
            recommendation=row["recommendation"],
            latency_ms=row["latency_ms"],
            request_source=row["request_source"],
            loan_type=row.get("loan_type"),
            age=row.get("age"),
            annual_income=(
                float(row["annual_income"])
                if row.get("annual_income") is not None
                else None
            ),
            employment_type=row.get("employment_type"),
            created_at=row["created_at"],
        )
        for row in rows
    ]
    return PredictionListResponse(
        items=items, total=total, page=page, page_size=page_size
    )
