"""Tests for creditguard.explain.shap_explainer: explainer selection,
one-hot aggregation, SHAP additivity, and training-artifact persistence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from creditguard.explain import shap_explainer
from creditguard.explain.shap_explainer import (
    ExplainabilityError,
    ShapExplanation,
    aggregate_by_source,
    build_explainer,
    build_portfolio_benchmarks,
    build_training_artifacts,
    clear_explainer_cache,
    explain,
    get_cached_explainer,
    load_background_sample,
    load_portfolio_benchmarks,
    map_to_source_feature,
    save_background_sample,
    save_portfolio_benchmarks,
    unwrap_base_estimator,
)
from creditguard.features.build import align_target, filter_tables, temporal_split
from creditguard.features.pipeline import build_feature_pipeline
from creditguard.validation.engine import load_rule_config


def _linear_fixture(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
        }
    )
    logits = 2.0 * X["num_a"] - 1.0 * X["num_b"]
    p = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.random(n) < p).astype(int))
    return X, y


def _calibrated_logistic(X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    base = LogisticRegression().fit(X, y)
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    calibrated.fit(X, y)
    return calibrated, base


@pytest.fixture(autouse=True)
def _reset_explainer_cache():
    clear_explainer_cache()
    yield
    clear_explainer_cache()


# --- unwrap_base_estimator -------------------------------------------------


def test_unwrap_base_estimator_recovers_the_original_fitted_model() -> None:
    X, y = _linear_fixture()
    calibrated, base = _calibrated_logistic(X, y)
    unwrapped = unwrap_base_estimator(calibrated)
    assert unwrapped is base
    np.testing.assert_allclose(unwrapped.coef_, base.coef_)


def test_unwrap_base_estimator_raises_explainability_error_for_wrong_shape() -> None:
    with pytest.raises(ExplainabilityError):
        unwrap_base_estimator(object())


# --- build_explainer / caching ---------------------------------------------


def test_build_explainer_selects_linear_for_logistic_regression() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    explainer = build_explainer("logistic_regression", base, X.head(50))
    assert type(explainer).__name__ == "LinearExplainer"


def test_build_explainer_selects_tree_for_random_forest() -> None:
    X, y = _linear_fixture()
    forest = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(
        X, y
    )
    explainer = build_explainer("random_forest", forest, X.head(50))
    assert type(explainer).__name__ == "TreeExplainer"


def test_build_explainer_raises_for_unknown_algorithm() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    with pytest.raises(ExplainabilityError, match="No SHAP explainer strategy"):
        build_explainer("some_future_algorithm", base, X.head(50))


def test_get_cached_explainer_reuses_the_same_instance() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    first = get_cached_explainer("model-1", "logistic_regression", base, X.head(50))
    second = get_cached_explainer("model-1", "logistic_regression", base, X.head(50))
    assert first is second


def test_clear_explainer_cache_forces_a_rebuild() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    first = get_cached_explainer("model-1", "logistic_regression", base, X.head(50))
    clear_explainer_cache()
    second = get_cached_explainer("model-1", "logistic_regression", base, X.head(50))
    assert first is not second


# --- map_to_source_feature / aggregate_by_source ----------------------------


NUMERIC = ["age", "dti"]
CATEGORICAL = ["gender", "employment_type"]
ORDINAL = ["income_band"]


@pytest.mark.parametrize(
    "encoded_name,expected_source",
    [
        ("age", "age"),
        ("dti", "dti"),
        ("income_band", "income_band"),
        ("gender_MALE", "gender"),
        ("gender_FEMALE", "gender"),
        ("employment_type_SELF_EMPLOYED", "employment_type"),
        ("employment_type_infrequent_sklearn", "employment_type"),
    ],
)
def test_map_to_source_feature(encoded_name: str, expected_source: str) -> None:
    assert (
        map_to_source_feature(encoded_name, NUMERIC, CATEGORICAL, ORDINAL)
        == expected_source
    )


def test_map_to_source_feature_prefers_longest_prefix_match() -> None:
    # "loan" is a prefix of "loan_type"; an encoded "loan_type_HOME" column
    # must map to "loan_type", not the shorter "loan".
    categorical = ["loan", "loan_type"]
    assert map_to_source_feature("loan_type_HOME", [], categorical, []) == "loan_type"


def test_aggregate_by_source_sums_one_hot_contributions() -> None:
    values = {
        "age": 0.10,
        "gender_MALE": 0.03,
        "gender_FEMALE": -0.01,
        "income_band": -0.05,
    }
    aggregated = aggregate_by_source(values, NUMERIC, CATEGORICAL, ORDINAL)
    assert aggregated["age"] == pytest.approx(0.10)
    assert aggregated["gender"] == pytest.approx(0.03 + -0.01)
    assert aggregated["income_band"] == pytest.approx(-0.05)
    assert set(aggregated) == {"age", "gender", "income_band"}


# --- explain() ---------------------------------------------------------------


def test_explain_shap_values_sum_to_prediction_minus_base_value() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    explainer = build_explainer("logistic_regression", base, X.head(100))
    row = X.iloc[[0]]

    explanation = explain(
        row, explainer, list(X.columns), NUMERIC, CATEGORICAL, ORDINAL
    )
    assert isinstance(explanation, ShapExplanation)

    margin = base.decision_function(row)[0]
    assert explanation.model_output_value == pytest.approx(margin, abs=1e-6)
    reconstructed = explanation.base_value + sum(
        explanation.contributions_by_encoded_feature.values()
    )
    assert reconstructed == pytest.approx(explanation.model_output_value, abs=1e-9)


def test_explain_one_hot_contributions_aggregate_to_parent_feature() -> None:
    rng = np.random.default_rng(7)
    n = 200
    numeric = rng.normal(size=n)
    category = rng.choice(["A", "B"], size=n)
    onehot_a = (category == "A").astype(float)
    onehot_b = (category == "B").astype(float)
    logits = 1.5 * numeric + 2.0 * onehot_a - 0.5 * onehot_b
    p = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.random(n) < p).astype(int))

    X = pd.DataFrame({"num_a": numeric, "cat_A": onehot_a, "cat_B": onehot_b})
    base = LogisticRegression().fit(X, y)
    explainer = build_explainer("logistic_regression", base, X.head(100))

    row = X.iloc[[0]]
    explanation = explain(
        row,
        explainer,
        list(X.columns),
        numeric_columns=["num_a"],
        categorical_columns=["cat"],
        ordinal_columns=[],
    )
    expected_cat_contribution = (
        explanation.contributions_by_encoded_feature["cat_A"]
        + explanation.contributions_by_encoded_feature["cat_B"]
    )
    assert explanation.contributions_by_source_feature["cat"] == pytest.approx(
        expected_cat_contribution
    )
    assert explanation.contributions_by_source_feature["num_a"] == pytest.approx(
        explanation.contributions_by_encoded_feature["num_a"]
    )
    assert set(explanation.contributions_by_source_feature) == {"num_a", "cat"}


def test_explain_top_factors_split_by_sign_and_ordered_by_magnitude() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    explainer = build_explainer("logistic_regression", base, X.head(100))
    row = X.iloc[[0]]

    explanation = explain(
        row, explainer, list(X.columns), NUMERIC, CATEGORICAL, ORDINAL, top_k=5
    )
    assert all(value > 0 for _name, value in explanation.top_positive_factors)
    assert all(value < 0 for _name, value in explanation.top_negative_factors)

    pos_magnitudes = [abs(value) for _name, value in explanation.top_positive_factors]
    assert pos_magnitudes == sorted(pos_magnitudes, reverse=True)
    neg_magnitudes = [abs(value) for _name, value in explanation.top_negative_factors]
    assert neg_magnitudes == sorted(neg_magnitudes, reverse=True)


def test_explain_raises_for_multi_row_input() -> None:
    X, y = _linear_fixture()
    _calibrated, base = _calibrated_logistic(X, y)
    explainer = build_explainer("logistic_regression", base, X.head(50))
    with pytest.raises(ValueError, match="exactly one row"):
        explain(X.iloc[:2], explainer, list(X.columns), NUMERIC, CATEGORICAL, ORDINAL)


def test_tree_explainer_additivity_does_not_raise() -> None:
    """shap.TreeExplainer validates additivity internally (check_additivity)
    -- a successful call already proves the sum-to-prediction property for
    the tree family.
    """
    X, y = _linear_fixture()
    forest = RandomForestClassifier(n_estimators=20, max_depth=4, random_state=42).fit(
        X, y
    )
    explainer = build_explainer("random_forest", forest, X.head(100))
    row = X.iloc[[0]]
    explanation = explain(
        row, explainer, list(X.columns), NUMERIC, CATEGORICAL, ORDINAL
    )
    assert isinstance(explanation, ShapExplanation)


# --- background sample / benchmark persistence ------------------------------


def test_save_and_load_background_sample_round_trip(tmp_path) -> None:
    background = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    save_background_sample(background, "model-x", tmp_path)
    loaded = load_background_sample("model-x", tmp_path)
    pd.testing.assert_frame_equal(loaded, background)


def test_load_background_sample_missing_raises(tmp_path) -> None:
    with pytest.raises(ExplainabilityError, match="No SHAP background sample"):
        load_background_sample("missing-model", tmp_path)


def test_build_portfolio_benchmarks_numeric_median_and_categorical_mode() -> None:
    frame = pd.DataFrame(
        {
            "dti": [0.1, 0.2, 0.3, 0.4, 0.5],
            "gender": ["MALE", "MALE", "FEMALE", "MALE", "FEMALE"],
            "income_band": ["Q1", "Q2", "Q1", "Q1", "Q3"],
        }
    )
    benchmarks = build_portfolio_benchmarks(frame, ["dti"], ["gender"], ["income_band"])
    assert benchmarks["dti"] == {"type": "numeric", "median": 0.3}
    assert benchmarks["gender"] == {"type": "categorical", "mode": "MALE"}
    assert benchmarks["income_band"] == {"type": "categorical", "mode": "Q1"}


def test_save_and_load_portfolio_benchmarks_round_trip(tmp_path) -> None:
    benchmarks = {"dti": {"type": "numeric", "median": 0.5}}
    save_portfolio_benchmarks(benchmarks, "model-x", tmp_path)
    loaded = load_portfolio_benchmarks("model-x", tmp_path)
    assert loaded == benchmarks


def test_load_portfolio_benchmarks_missing_raises(tmp_path) -> None:
    with pytest.raises(ExplainabilityError, match="No portfolio benchmark"):
        load_portfolio_benchmarks("missing-model", tmp_path)


# --- build_training_artifacts (end to end on a small fixture) --------------


def _features_config() -> dict:
    return {
        "behavioural": {"n_quantile_bins": 5},
        "preprocessing": {"min_frequency": 0.01},
        "feature_columns": {
            "numeric": [
                "age",
                "dependents",
                "employment_years",
                "annual_income",
                "city_tier",
                "loan_amount",
                "loan_tenure_months",
                "interest_rate",
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
                "dti",
                "emi_to_income",
                "credit_utilization",
                "loan_to_income",
                "proposed_emi",
                "post_loan_dti",
                "savings_to_income",
                "net_worth",
                "leverage_ratio",
                "disposable_income",
                "months_of_runway",
                "delinquency_rate",
                "has_prior_default",
                "credit_history_years",
                "accounts_per_year",
                "active_loan_ratio",
                "employment_stability",
                "income_per_dependent",
            ],
            "categorical": [
                "gender",
                "marital_status",
                "education",
                "employment_type",
                "loan_type",
                "loan_purpose",
            ],
            "ordinal": ["utilization_band", "age_band", "tenure_band", "income_band"],
        },
    }


def _multitable_fixture(n_customers: int = 20) -> dict[str, pd.DataFrame]:
    ages = [22 + i for i in range(n_customers)]
    employment_years = [1.0 + 0.5 * i for i in range(n_customers)]
    annual_income = [300000.0 + 20000 * i for i in range(n_customers)]
    customer_ids = [f"C{i:03d}" for i in range(n_customers)]
    loan_ids = [f"L{i:03d}" for i in range(n_customers)]
    monthly_income = [a / 12 for a in annual_income]

    customers = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "age": ages,
            "gender": ["MALE", "FEMALE"] * (n_customers // 2),
            "marital_status": ["SINGLE", "MARRIED"] * (n_customers // 2),
            "dependents": [0, 1] * (n_customers // 2),
            "education": ["GRADUATE"] * n_customers,
            "employment_type": ["SALARIED"] * n_customers,
            "employment_years": employment_years,
            "annual_income": annual_income,
            "monthly_income": monthly_income,
            "city_tier": [1, 2, 3] * (n_customers // 3) + [1] * (n_customers % 3),
        }
    )
    application_dates = [
        pd.Timestamp("2024-01-01") + pd.Timedelta(days=5 * i)
        for i in range(n_customers)
    ]
    loan_applications = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "customer_id": customer_ids,
            "loan_type": ["PERSONAL"] * n_customers,
            "loan_amount": [100000.0 + 5000 * i for i in range(n_customers)],
            "loan_tenure_months": [24] * n_customers,
            "interest_rate": [12.0] * n_customers,
            "loan_purpose": ["OTHER"] * n_customers,
            "application_date": application_dates,
            "decision_date": [d + pd.Timedelta(days=3) for d in application_dates],
            "status": ["APPROVED"] * n_customers,
        }
    )
    as_of_dates = [d - pd.Timedelta(days=5) for d in application_dates]
    financial_profiles = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "monthly_income": monthly_income,
            "monthly_expenses": [m * 0.3 for m in monthly_income],
            "existing_loan_count": [0] * n_customers,
            "existing_loan_amount": [0.0] * n_customers,
            "monthly_emi": [m * 0.1 for m in monthly_income],
            "savings_balance": [m * 6 for m in monthly_income],
            "total_assets": [a * 0.5 for a in annual_income],
            "total_liabilities": [0.0] * n_customers,
        }
    )
    credit_history = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "as_of_date": as_of_dates,
            "credit_history_months": [24 + 6 * i for i in range(n_customers)],
            "num_credit_accounts": [2] * n_customers,
            "total_credit_limit": [200000.0] * n_customers,
            "total_outstanding": [50000.0] * n_customers,
            "credit_utilization": [0.25] * n_customers,
            "previous_defaults": [0] * n_customers,
            "late_payments_12m": [0] * n_customers,
            "missed_payments_12m": [0] * n_customers,
            "active_loans": [1] * n_customers,
            "closed_loans": [1] * n_customers,
        }
    )
    loan_outcomes = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "default_12m": [i % 2 for i in range(n_customers)],
            "outcome_observed_date": [
                d + pd.DateOffset(months=12) for d in application_dates
            ],
        }
    )
    return {
        "customers": customers,
        "loan_applications": loan_applications,
        "financial_profiles": financial_profiles,
        "credit_history": credit_history,
        "loan_outcomes": loan_outcomes,
    }


def test_build_training_artifacts_end_to_end(tmp_path, monkeypatch) -> None:
    tables = _multitable_fixture()
    features_config = _features_config()
    cleaning_config = load_rule_config("config/validation_rules.yaml")

    pipeline = build_feature_pipeline(features_config, cleaning_config)
    train_ids, _val_ids, _test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)
    merge_step = pipeline.named_steps["cleaning_and_merge"]
    train_merged = merge_step.fit_transform(train_tables)
    y_train = align_target(train_merged, train_tables["loan_outcomes"])
    pipeline[1:].fit(train_merged, y_train)

    monkeypatch.setattr(
        shap_explainer, "read_dataset_tables", lambda dataset_version, data_root: tables
    )
    background, benchmarks = build_training_artifacts(
        "fixture-version",
        pipeline,
        features_config,
        tmp_path,
        n_background=200,
        seed=42,
    )

    assert background.shape[0] <= 200
    assert background.shape[0] > 0
    assert list(background.columns) == list(
        pipeline.named_steps["preprocess"].get_feature_names_out()
    )
    n_expected_benchmarks = (
        len(features_config["feature_columns"]["numeric"])
        + len(features_config["feature_columns"]["categorical"])
        + len(features_config["feature_columns"]["ordinal"])
    )
    assert len(benchmarks) == n_expected_benchmarks
    assert benchmarks["dti"]["type"] == "numeric"
    assert benchmarks["gender"]["type"] == "categorical"
