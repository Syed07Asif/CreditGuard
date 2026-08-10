"""The Phase 7 scoring engine: `score_application` is the single entry
point everything else (Phase 8's API, Phase 9's dashboard, this phase's own
fixture demo) calls.

`raw_input` carries one applicant's full point-in-time snapshot as a flat
dict -- demographics, loan terms, financial profile and credit-bureau
fields, all already resolved to "as of now" values by the caller. Fetching
that snapshot for an *existing* customer from the database (by
`customer_id`, as of an application date) is Phase 8's concern, not this
function's: `score_application` validates and shapes the given snapshot
into the same flat-row layout `creditguard.features.leakage.
point_in_time_join` produces for the multi-table training pipeline, then
runs it through the *same* ratios/behavioural/`ColumnTransformer` stages,
so scoring and training are guaranteed to treat a feature identically.

Steps: validate input -> assemble the point-in-time feature row -> feature
pipeline transform -> active (calibrated) model `predict_proba` -> score ->
category -> recommendation -> SHAP explanation -> persist to `predictions`
-> return.

The active model, its refit feature pipeline, the scoring config and the
SHAP explainer are all loaded once into a module-level singleton
(`_get_loaded_model`) and reused across calls; `reload_active_model()` is
the explicit way to force a reload (call it after promoting a new model).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from creditguard.config import get_settings
from creditguard.data.versioning import read_dataset_tables
from creditguard.db.repository import PredictionRepository
from creditguard.explain import reason_codes, shap_explainer
from creditguard.features.build import align_target, filter_tables, temporal_split
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.models import registry
from creditguard.scoring import categories, recommendation, scorecard
from creditguard.validation.engine import load_rule_config

DEFAULT_FEATURES_CONFIG_PATH = "config/features.yaml"
DEFAULT_CLEANING_CONFIG_PATH = "config/validation_rules.yaml"
DEFAULT_SCORING_CONFIG_PATH = "config/scoring.yaml"

LOAN_TYPES = ("PERSONAL", "HOME", "AUTO", "EDUCATION", "BUSINESS", "CREDIT_CARD")


class ScoringInputError(ValueError):
    """Raised when `raw_input` fails validation."""


class ScoringEngineError(RuntimeError):
    """Raised for engine setup failures (e.g. no active model registered)."""


class RawApplicationInput(BaseModel):
    """One applicant's point-in-time snapshot: demographics, loan terms,
    financial profile and credit-bureau fields -- everything
    `creditguard.features.ratios`/`behavioural`/`encoders` need, at the
    logical (pre-engineering) level.
    """

    model_config = ConfigDict(extra="ignore")

    customer_id: str
    loan_id: str | None = None

    age: int = Field(ge=18, le=100)
    gender: str
    marital_status: str
    dependents: int = Field(ge=0)
    education: str
    employment_type: str
    employment_years: float = Field(ge=0)
    annual_income: float = Field(ge=0)
    city_tier: int

    loan_type: Literal[LOAN_TYPES]  # type: ignore[valid-type]
    loan_amount: float = Field(gt=0)
    loan_tenure_months: int = Field(gt=0)
    interest_rate: float = Field(ge=0)
    loan_purpose: str

    monthly_income: float = Field(ge=0)
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
    previous_defaults: int = Field(ge=0)
    late_payments_12m: int = Field(ge=0)
    missed_payments_12m: int = Field(ge=0)
    active_loans: int = Field(ge=0)
    closed_loans: int = Field(ge=0)


# The subset of RawApplicationInput's fields that feed the feature pipeline
# (everything except the identifiers, which are metadata for the result/
# predictions row, not model input).
_FRAME_COLUMNS: tuple[str, ...] = (
    "age",
    "gender",
    "marital_status",
    "dependents",
    "education",
    "employment_type",
    "employment_years",
    "annual_income",
    "city_tier",
    "loan_type",
    "loan_amount",
    "loan_tenure_months",
    "interest_rate",
    "loan_purpose",
    "monthly_income",
    "monthly_expenses",
    "existing_loan_count",
    "existing_loan_amount",
    "monthly_emi",
    "savings_balance",
    "total_assets",
    "total_liabilities",
    "credit_history_months",
    "num_credit_accounts",
    "total_credit_limit",
    "total_outstanding",
    "previous_defaults",
    "late_payments_12m",
    "missed_payments_12m",
    "active_loans",
    "closed_loans",
)


def _validate_input(raw_input: dict[str, Any]) -> RawApplicationInput:
    try:
        return RawApplicationInput.model_validate(raw_input)
    except ValidationError as exc:
        raise ScoringInputError(f"Invalid scoring input: {exc}") from exc


def _build_point_in_time_frame(validated: RawApplicationInput) -> pd.DataFrame:
    """Shape a validated snapshot into the one-row layout the ratios/
    behavioural/`ColumnTransformer` stages expect (the same column set
    `creditguard.features.leakage.point_in_time_join` produces, minus the
    identifier/date columns that aren't model input).
    """
    payload = validated.model_dump()
    return pd.DataFrame([{column: payload[column] for column in _FRAME_COLUMNS}])


@dataclass(frozen=True)
class ScoringResult:
    """Everything a loan officer (or Phase 8's API response) needs for one
    scored application.
    """

    loan_id: str
    customer_id: str
    model_id: str
    model_version: str
    default_probability: float
    credit_score: int
    risk_category: str
    recommendation: str
    triggered_rules: list[str]
    top_risk_factors: list[dict[str, Any]]
    top_positive_factors: list[dict[str, Any]]
    latency_ms: int
    scored_at: datetime


@dataclass
class _LoadedModel:
    """Everything `score_application` needs, loaded once and cached."""

    model_id: str
    model_version: str
    algorithm: str
    calibrated_model: Any
    ratios_behavioural: (
        Any  # fitted pipeline[1:-1]: RatioFeatures -> BehaviouralFeatures
    )
    preprocess: Any  # fitted pipeline.named_steps["preprocess"]: ColumnTransformer
    feature_names: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    ordinal_columns: list[str]
    scorecard_config: scorecard.ScorecardConfig
    risk_bands: list[categories.RiskBand]
    recommendation_policy: recommendation.RecommendationPolicy
    explainer: Any
    benchmarks: dict[str, dict[str, Any]]
    top_k: int


_MODEL_CACHE: _LoadedModel | None = None


def _load_active_model(
    *,
    features_config_path: str | Path = DEFAULT_FEATURES_CONFIG_PATH,
    cleaning_config_path: str | Path = DEFAULT_CLEANING_CONFIG_PATH,
    scoring_config_path: str | Path = DEFAULT_SCORING_CONFIG_PATH,
    data_dir: Path | None = None,
) -> _LoadedModel:
    """Load the active `model_registry` row, refit its feature pipeline on
    its own training data (guaranteeing an exact match to how that model
    was trained -- see the module docstring), and build/load its SHAP
    explainer and scoring config. Meant to run once per process; use
    `reload_active_model()` to force a rebuild.
    """
    settings = get_settings()
    data_root = data_dir if data_dir is not None else settings.data_dir / "processed"

    model_row = registry.get_active_model()
    if model_row is None:
        raise ScoringEngineError(
            "No active model registered -- run `python -m creditguard.models.train "
            "--register-best` first (see creditguard.models.registry.register_model)."
        )
    model_id = model_row["model_id"]
    dataset_version = model_row["dataset_version"]
    algorithm = model_row["algorithm"]
    chosen_threshold = float(model_row["metrics"]["chosen_threshold"])

    calibrated_model = joblib.load(model_row["artifact_path"])
    base_estimator = shap_explainer.unwrap_base_estimator(calibrated_model)

    with open(features_config_path, encoding="utf-8") as f:
        features_config = yaml.safe_load(f)
    cleaning_config = load_rule_config(cleaning_config_path)
    with open(scoring_config_path, encoding="utf-8") as f:
        scoring_config = yaml.safe_load(f)

    pipeline = build_feature_pipeline(features_config, cleaning_config)
    tables = read_dataset_tables(dataset_version, data_root)
    train_ids, _val_ids, _test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    train_merged = merge_step.fit_transform(train_tables)
    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    pipeline[1:].fit(train_merged, y_train)

    preprocess = pipeline.named_steps["preprocess"]
    feature_names = list(preprocess.get_feature_names_out())

    try:
        background = shap_explainer.load_background_sample(model_id, settings.model_dir)
        benchmarks = shap_explainer.load_portfolio_benchmarks(
            model_id, settings.model_dir
        )
    except shap_explainer.ExplainabilityError:
        # Not built yet at training-promotion time -- build once now and
        # persist, so every later load (this process or the next) reuses it.
        n_background = int(scoring_config["explainability"]["background_sample_size"])
        seed = int(scoring_config.get("random_seed", 42))
        background, benchmarks = shap_explainer.build_training_artifacts(
            dataset_version,
            pipeline,
            features_config,
            data_root,
            n_background=n_background,
            seed=seed,
        )
        shap_explainer.save_background_sample(background, model_id, settings.model_dir)
        shap_explainer.save_portfolio_benchmarks(
            benchmarks, model_id, settings.model_dir
        )

    explainer = shap_explainer.get_cached_explainer(
        model_id, algorithm, base_estimator, background
    )

    scorecard_config = scorecard.ScorecardConfig.from_config(scoring_config)
    risk_bands = categories.load_risk_bands(scoring_config)
    recommendation_policy = recommendation.RecommendationPolicy.from_config(
        scoring_config, chosen_threshold
    )
    columns = features_config["feature_columns"]

    return _LoadedModel(
        model_id=model_id,
        model_version=model_row["model_version"],
        algorithm=algorithm,
        calibrated_model=calibrated_model,
        ratios_behavioural=pipeline[1:-1],
        preprocess=preprocess,
        feature_names=feature_names,
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"],
        ordinal_columns=columns["ordinal"],
        scorecard_config=scorecard_config,
        risk_bands=risk_bands,
        recommendation_policy=recommendation_policy,
        explainer=explainer,
        benchmarks=benchmarks,
        top_k=int(scoring_config["explainability"]["top_k_factors"]),
    )


def _get_loaded_model() -> _LoadedModel:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = _load_active_model()
    return _MODEL_CACHE


def reload_active_model() -> None:
    """Force the next `score_application` call to reload the active model,
    its refit feature pipeline, scoring config and SHAP explainer from
    scratch. Call after promoting a new model (`creditguard.models.registry.
    register_model`).
    """
    global _MODEL_CACHE
    _MODEL_CACHE = None
    shap_explainer.clear_explainer_cache()


def is_model_loaded() -> bool:
    """Whether the active model singleton is currently warm -- Phase 8's
    `/health/ready` readiness check reads this rather than reaching into
    `_MODEL_CACHE` directly.
    """
    return _MODEL_CACHE is not None


def warm_up() -> None:
    """Eagerly load the active model, its refit feature pipeline, scoring
    config and SHAP explainer, so the first real request doesn't pay that
    one-time cost. Call once at process startup (Phase 8's FastAPI
    `lifespan`); raises whatever `_load_active_model` raises (e.g.
    `ScoringEngineError` if no model is registered) so the caller can decide
    how to handle a failed startup load.
    """
    _get_loaded_model()


def _persist_prediction(result: ScoringResult, request_source: str) -> None:
    PredictionRepository().insert_many(
        [
            {
                "loan_id": result.loan_id,
                "customer_id": result.customer_id,
                "model_id": result.model_id,
                "default_probability": result.default_probability,
                "credit_score": result.credit_score,
                "risk_category": result.risk_category,
                "recommendation": result.recommendation,
                "top_risk_factors": result.top_risk_factors,
                "top_positive_factors": result.top_positive_factors,
                "latency_ms": result.latency_ms,
                "request_source": request_source,
            }
        ]
    )


def _score_and_explain(
    raw_input: dict[str, Any],
) -> tuple[ScoringResult, shap_explainer.ShapExplanation, dict[str, Any]]:
    """Shared body of `score_application`/`explain_application`: validate ->
    assemble the point-in-time feature row -> feature pipeline transform ->
    the active calibrated model's `predict_proba` -> credit score -> risk
    category -> lending recommendation -> SHAP explanation. Returns the
    `ScoringResult`, the full `ShapExplanation` (every feature, not just the
    top-k `ScoringResult` carries), and the human-readable raw feature row
    (for callers building a fuller explanation than `ScoringResult` exposes,
    e.g. Phase 8's `/explain` endpoint). Never persists -- callers decide
    that.

    `ScoringResult.latency_ms` measures this call's own work (feature
    assembly onward), not the one-time cost of loading the active model on
    the very first call in a process -- `_get_loaded_model()` caches that.
    """
    validated = _validate_input(raw_input)
    loaded = _get_loaded_model()

    t0 = time.perf_counter()

    row = _build_point_in_time_frame(validated)
    human_frame = loaded.ratios_behavioural.transform(row)
    X = loaded.preprocess.transform(human_frame)
    X_df = pd.DataFrame(X, columns=loaded.feature_names)

    p_default = float(loaded.calibrated_model.predict_proba(X_df)[:, 1][0])
    credit_score = scorecard.probability_to_score(p_default, loaded.scorecard_config)
    risk_category = categories.categorize(credit_score, loaded.risk_bands)

    raw_features = human_frame.iloc[0].to_dict()
    decision = recommendation.recommend(
        probability=p_default,
        credit_score=credit_score,
        features=raw_features,
        policy=loaded.recommendation_policy,
    )

    explanation = shap_explainer.explain(
        X_df,
        loaded.explainer,
        loaded.feature_names,
        loaded.numeric_columns,
        loaded.categorical_columns,
        loaded.ordinal_columns,
        top_k=loaded.top_k,
    )
    risk_factors, positive_factors = reason_codes.generate_reason_codes(
        explanation.top_positive_factors,
        explanation.top_negative_factors,
        raw_features,
        loaded.benchmarks,
    )

    latency_ms = int((time.perf_counter() - t0) * 1000)
    loan_id = validated.loan_id or f"SIM-{uuid.uuid4().hex[:12]}"

    result = ScoringResult(
        loan_id=loan_id,
        customer_id=validated.customer_id,
        model_id=loaded.model_id,
        model_version=loaded.model_version,
        default_probability=p_default,
        credit_score=credit_score,
        risk_category=risk_category,
        recommendation=decision.decision,
        triggered_rules=decision.triggered_rules,
        top_risk_factors=risk_factors,
        top_positive_factors=positive_factors,
        latency_ms=latency_ms,
        scored_at=datetime.now(UTC),
    )
    return result, explanation, raw_features


def score_application(
    raw_input: dict[str, Any],
    *,
    persist: bool = True,
    request_source: str = "scoring_engine",
) -> ScoringResult:
    """Score one loan application end to end and (optionally) persist the
    result to `predictions`. See `_score_and_explain` for the pipeline
    itself; this wrapper just adds persistence.
    """
    result, _explanation, _raw_features = _score_and_explain(raw_input)
    if persist:
        _persist_prediction(result, request_source)
    return result


@dataclass(frozen=True)
class DetailedExplanation:
    """Everything Phase 8's `/explain` endpoint needs beyond what
    `ScoringResult` carries: a per-feature SHAP contribution for *every*
    logical feature (not just the top-k risk/positive factors), plus the
    raw applicant values and portfolio benchmarks needed to render a reason
    code for any of them on demand.
    """

    result: ScoringResult
    shap_base_value: float
    contributions_by_feature: dict[str, float]
    raw_features: dict[str, Any]
    benchmarks: dict[str, dict[str, Any]]


def explain_application(raw_input: dict[str, Any]) -> DetailedExplanation:
    """Full per-feature SHAP breakdown for one application. Does not
    persist a `predictions` row -- explaining isn't itself a scoring
    decision event; callers that also want the decision logged should call
    `score_application` separately (or persist `.result` themselves).
    """
    result, explanation, raw_features = _score_and_explain(raw_input)
    loaded = _get_loaded_model()
    return DetailedExplanation(
        result=result,
        shap_base_value=explanation.base_value,
        contributions_by_feature=explanation.contributions_by_source_feature,
        raw_features=raw_features,
        benchmarks=loaded.benchmarks,
    )
