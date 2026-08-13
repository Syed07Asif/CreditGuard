"""SHAP-based explainability for the active CreditGuard model.

Explains the **base (pre-calibration) estimator's decision function**, not
the final calibrated probability directly: `creditguard.models.calibration`
always wraps the winning model in `CalibratedClassifierCV` via
`sklearn.frozen.FrozenEstimator`, and isotonic calibration in particular has
no closed-form gradient/coefficients for SHAP to decompose. Calibration is a
monotonic transform of the base model's output, so a feature's *direction*
and *relative importance* as a risk driver survive it unchanged even though
the raw SHAP numbers are in log-odds space, not probability space -- see
`docs/scoring_methodology.md` for the same note aimed at a non-technical
reader.

`TreeExplainer` is selected for tree-ensemble algorithms, `LinearExplainer`
for logistic regression -- chosen automatically from the model's registered
`algorithm` name, never hard-coded to one family. Both need a background/
reference sample of training data; `build_training_artifacts` computes one
(200 stratified training rows, by default) once, at training-promotion
time, and `save_background_sample`/`load_background_sample` persist and
reload it so scoring requests never recompute it. `build_training_artifacts`
also computes the portfolio benchmark statistics `creditguard.explain.
reason_codes` needs, from that same training-data reload.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from creditguard.data.versioning import read_dataset_tables
from creditguard.features.build import align_target, filter_tables, temporal_split

# Which SHAP explainer strategy applies to which registered `algorithm`
# name (`model_registry.algorithm` / `creditguard.models.imbalance.
# MODEL_CLASSES` keys) -- extend here, not with an isinstance check
# scattered through the codebase, if a future phase registers a new family.
TREE_ALGORITHMS: frozenset[str] = frozenset({"random_forest", "xgboost"})
LINEAR_ALGORITHMS: frozenset[str] = frozenset({"logistic_regression"})


class ExplainabilityError(RuntimeError):
    """Raised for explainability setup failures: an unrecognised algorithm,
    or a missing persisted artifact.
    """


def unwrap_base_estimator(calibrated_model: Any) -> Any:
    """The raw fitted sklearn/xgboost estimator inside a
    `CalibratedClassifierCV` produced by `creditguard.models.calibration.
    calibrate_model` -- which always wraps the base model in a
    `sklearn.frozen.FrozenEstimator` before fitting the calibrator on top,
    with a single (never cross-validated) calibrated classifier since the
    base estimator is frozen, not refit. This is what SHAP needs direct
    access to (`coef_` / tree structure) -- the calibrator itself has no
    such structure to explain.
    """
    try:
        frozen_estimator = calibrated_model.calibrated_classifiers_[0].estimator
        return frozen_estimator.estimator
    except (AttributeError, IndexError) as exc:
        raise ExplainabilityError(
            "Could not unwrap a base estimator from the calibrated model -- "
            "expected a CalibratedClassifierCV built from a FrozenEstimator, "
            "as creditguard.models.calibration.calibrate_model always builds."
        ) from exc


def build_explainer(
    algorithm: str, base_estimator: Any, background: pd.DataFrame
) -> Any:
    """Construct the appropriate SHAP explainer for `algorithm`, backed by
    `background` (the persisted reference sample).
    """
    if algorithm in TREE_ALGORITHMS:
        return shap.TreeExplainer(
            base_estimator, data=background, feature_perturbation="interventional"
        )
    if algorithm in LINEAR_ALGORITHMS:
        return shap.LinearExplainer(base_estimator, background)
    raise ExplainabilityError(
        f"No SHAP explainer strategy registered for algorithm {algorithm!r} "
        f"-- known tree algorithms: {sorted(TREE_ALGORITHMS)}, "
        f"known linear algorithms: {sorted(LINEAR_ALGORITHMS)}"
    )


# Cache keyed by model_id: SHAP explainer construction reads the full
# background sample and (for LinearExplainer) the estimator's covariance,
# so it's done once per active model and reused, never rebuilt per request.
_EXPLAINER_CACHE: dict[str, Any] = {}


def get_cached_explainer(
    model_id: str, algorithm: str, base_estimator: Any, background: pd.DataFrame
) -> Any:
    """Return the cached explainer for `model_id`, building and caching it
    on first use.
    """
    if model_id not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[model_id] = build_explainer(
            algorithm, base_estimator, background
        )
    return _EXPLAINER_CACHE[model_id]


def clear_explainer_cache() -> None:
    """Drop every cached explainer -- call after promoting a new model."""
    _EXPLAINER_CACHE.clear()


def map_to_source_feature(
    encoded_name: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
) -> str:
    """The logical/source feature an encoded model-input column came from.

    Numeric and ordinal columns pass through the `ColumnTransformer`
    unchanged (1:1), so they map to themselves. Categorical columns are
    one-hot expanded by `creditguard.features.encoders.build_preprocessor`
    into `f"{column}_{category}"` (`verbose_feature_names_out=False`), so an
    encoded name maps back to whichever categorical column is a matching
    `f"{column}_"` prefix -- the longest match, in case one categorical
    column name is itself a prefix of another's.
    """
    if encoded_name in numeric_columns or encoded_name in ordinal_columns:
        return encoded_name
    candidates = [
        column
        for column in categorical_columns
        if encoded_name.startswith(f"{column}_")
    ]
    if candidates:
        return max(candidates, key=len)
    return encoded_name


def aggregate_by_source(
    values: dict[str, float],
    numeric_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
) -> dict[str, float]:
    """Sum `values` (keyed by encoded model-input column) back onto their
    logical source feature, so a one-hot-expanded categorical contributes
    one number, not one per category.
    """
    aggregated: dict[str, float] = {}
    for encoded_name, value in values.items():
        source = map_to_source_feature(
            encoded_name, numeric_columns, categorical_columns, ordinal_columns
        )
        aggregated[source] = aggregated.get(source, 0.0) + value
    return aggregated


@dataclass(frozen=True)
class ShapExplanation:
    """One row's SHAP explanation, both at the encoded-model-input level
    and aggregated back to logical/source feature names.
    """

    base_value: float
    model_output_value: float
    contributions_by_encoded_feature: dict[str, float]
    contributions_by_source_feature: dict[str, float]
    top_positive_factors: list[tuple[str, float]]
    top_negative_factors: list[tuple[str, float]]


def explain(
    X_row: pd.DataFrame,
    explainer: Any,
    feature_names: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
    top_k: int = 5,
) -> ShapExplanation:
    """Explain a single row already in the model's encoded feature space
    (exactly one row, columns == `feature_names`, matching what the base
    estimator's `predict_proba`/`decision_function` consumes).
    """
    if len(X_row) != 1:
        raise ValueError(f"explain() takes exactly one row, got {len(X_row)}")

    raw_explanation = explainer(X_row)
    values_arr = np.asarray(raw_explanation.values, dtype=float)
    base_arr = np.asarray(raw_explanation.base_values, dtype=float)
    if values_arr.ndim == 3:
        # Some explainers (e.g. TreeExplainer on a binary-classifier
        # RandomForest) return one contribution per class,
        # shape (1, n_features, n_classes) -- take the positive
        # (default == 1) class, index 1, consistently with
        # `predict_proba(...)[:, 1]` everywhere else in this project.
        values = values_arr[0, :, 1]
        base_value = float(base_arr[0, 1])
    else:
        values = values_arr[0]
        base_value = float(base_arr.reshape(-1)[0])

    by_encoded = dict(zip(feature_names, values.tolist(), strict=True))
    by_source = aggregate_by_source(
        by_encoded, numeric_columns, categorical_columns, ordinal_columns
    )

    ranked = sorted(by_source.items(), key=lambda item: abs(item[1]), reverse=True)
    top_positive = [(name, value) for name, value in ranked if value > 0][:top_k]
    top_negative = [(name, value) for name, value in ranked if value < 0][:top_k]

    return ShapExplanation(
        base_value=base_value,
        model_output_value=base_value + float(values.sum()),
        contributions_by_encoded_feature=by_encoded,
        contributions_by_source_feature=by_source,
        top_positive_factors=top_positive,
        top_negative_factors=top_negative,
    )


def background_sample_path(model_id: str, model_dir: str | Path) -> Path:
    """Where `model_id`'s persisted SHAP background sample lives."""
    return Path(model_dir) / f"shap_background_{model_id}.parquet"


def save_background_sample(
    background: pd.DataFrame, model_id: str, model_dir: str | Path
) -> Path:
    """Persist `background` as this model's SHAP reference sample artifact."""
    path = background_sample_path(model_id, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    background.reset_index(drop=True).to_parquet(path, index=False)
    return path


def load_background_sample(model_id: str, model_dir: str | Path) -> pd.DataFrame:
    """Load `model_id`'s persisted SHAP background sample.

    Raises:
        ExplainabilityError: if no artifact has been built yet for this
            model (call `build_training_artifacts` + `save_background_sample`
            first, normally at training-promotion time).
    """
    path = background_sample_path(model_id, model_dir)
    if not path.exists():
        raise ExplainabilityError(
            f"No SHAP background sample artifact at {path}. Build one with "
            "build_training_artifacts(...) and save_background_sample(...) "
            "before scoring -- this should happen once, at training-"
            "promotion time, not per request."
        )
    return pd.read_parquet(path)


def portfolio_benchmarks_path(model_id: str, model_dir: str | Path) -> Path:
    """Where `model_id`'s persisted portfolio benchmark statistics live."""
    return Path(model_dir) / f"portfolio_benchmarks_{model_id}.json"


def save_portfolio_benchmarks(
    benchmarks: dict[str, dict[str, Any]], model_id: str, model_dir: str | Path
) -> Path:
    """Persist `benchmarks` (see `build_portfolio_benchmarks`) as this
    model's reason-code comparison artifact.
    """
    path = portfolio_benchmarks_path(model_id, model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(benchmarks, indent=2), encoding="utf-8")
    return path


def load_portfolio_benchmarks(
    model_id: str, model_dir: str | Path
) -> dict[str, dict[str, Any]]:
    """Load `model_id`'s persisted portfolio benchmark statistics.

    Raises:
        ExplainabilityError: if no artifact has been built yet for this model.
    """
    path = portfolio_benchmarks_path(model_id, model_dir)
    if not path.exists():
        raise ExplainabilityError(
            f"No portfolio benchmark artifact at {path}. Build one with "
            "build_training_artifacts(...) and save_portfolio_benchmarks(...) "
            "before scoring."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_portfolio_benchmarks(
    train_frame: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
) -> dict[str, dict[str, Any]]:
    """Per-feature portfolio benchmark: the training-split median for
    numeric features, the training-split mode for categorical/ordinal ones
    -- what `creditguard.explain.reason_codes` compares an applicant's raw
    value against.
    """
    benchmarks: dict[str, dict[str, Any]] = {}
    for column in numeric_columns:
        benchmarks[column] = {
            "type": "numeric",
            "median": float(pd.to_numeric(train_frame[column]).median()),
        }
    for column in list(categorical_columns) + list(ordinal_columns):
        modes = train_frame[column].mode(dropna=True)
        benchmarks[column] = {
            "type": "categorical",
            "mode": str(modes.iloc[0]) if not modes.empty else None,
        }
    return benchmarks


def build_training_artifacts(
    dataset_version: str,
    feature_pipeline: Pipeline,
    features_config: dict[str, Any],
    data_dir: str | Path,
    n_background: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Rebuild the training split through the already-fitted
    `feature_pipeline` (never re-fits anything -- reusing a fitted
    pipeline's `.transform` on its own training tables is leakage-free by
    construction) and derive both artifacts a scoring engine needs from
    training data: a class-stratified SHAP background sample (encoded
    feature space) and the portfolio benchmark statistics (human-readable,
    pre-encoding feature space) `creditguard.explain.reason_codes` compares
    applicants against.

    Meant to run once, at training-promotion time (or whenever it's missing
    -- `creditguard.scoring.engine`'s model cache calls this lazily on first
    use if the artifacts aren't already on disk), not per scoring request.
    """
    tables = read_dataset_tables(dataset_version, Path(data_dir))
    train_ids, _val_ids, _test_ids = temporal_split(tables["loan_applications"])
    train_tables = filter_tables(tables, train_ids)

    merge_step = feature_pipeline.named_steps["cleaning_and_merge"]
    train_merged = merge_step.transform(train_tables)
    y_train = align_target(train_merged, train_tables["loan_outcomes"])

    ratios_behavioural = feature_pipeline[1:-1]
    train_frame = ratios_behavioural.transform(train_merged).reset_index(drop=True)

    preprocess = feature_pipeline.named_steps["preprocess"]
    X_train = preprocess.transform(train_frame)
    feature_names = list(preprocess.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    sample_size = min(n_background, len(X_train_df))
    if sample_size >= len(X_train_df):
        # Nothing to subsample -- the whole training split IS the sample.
        background = X_train_df
    else:
        background, _ = train_test_split(
            X_train_df, train_size=sample_size, stratify=y_train, random_state=seed
        )

    columns = features_config["feature_columns"]
    benchmarks = build_portfolio_benchmarks(
        train_frame, columns["numeric"], columns["categorical"], columns["ordinal"]
    )

    return background.reset_index(drop=True), benchmarks
