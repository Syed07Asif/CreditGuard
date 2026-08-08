"""Assembles the full Phase 4 feature pipeline: Phase 3 cleaning + point-in-time
merge -> ratio features -> behavioural features -> `ColumnTransformer`.

`CleaningAndMergeStep` reuses `creditguard.validation.cleaning.DataCleaner`
unmodified rather than reimplementing cleaning here: it operates on the same
`dict[str, DataFrame]` shape `DataCleaner` already expects, and composes the
point-in-time join (a deterministic lookup, not a fitted statistic) so the
rest of the pipeline can work on one flat per-loan frame. Because `DataCleaner`
is idempotent, feeding it an already-`_clean` dataset version (as
`creditguard.features.build`'s CLI does) is a safe no-op -- the same pipeline
object also works unmodified on genuinely raw input at serving time.

See `creditguard.features.build` for why this pipeline is *not* fit with one
opaque `pipeline.fit_transform(tables, y)` call: the target (`default_12m`)
only has a well-defined row order once the merge step has run, so the build
CLI fits the `cleaning_and_merge` step first (via `pipeline.named_steps`),
aligns `y` by `loan_id`, then fits the remainder (via `pipeline[1:]`, which
shares the same underlying step objects) -- leaving `pipeline` itself fully
fitted and ready to `joblib.dump`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from creditguard.features.behavioural import BehaviouralFeatures
from creditguard.features.encoders import build_preprocessor
from creditguard.features.leakage import (
    MERGED_FRAME_COLUMNS,
    assert_no_leakage,
    point_in_time_join,
)
from creditguard.features.ratios import RatioFeatures
from creditguard.validation.cleaning import DataCleaner

Tables = dict[str, pd.DataFrame]


class CleaningAndMergeStep(BaseEstimator, TransformerMixin):
    """Cleans the raw multi-table dataset (Phase 3's `DataCleaner`) then
    assembles one point-in-time-joined row per loan
    (`creditguard.features.leakage.point_in_time_join`).

    This is the pipeline's "cleaning transformer (Phase 3)" stage, composed
    with the merge because every downstream stage (ratios, behavioural,
    `ColumnTransformer`) needs a single flat per-loan frame, while
    `DataCleaner` operates on the dict-of-tables shape.
    """

    def __init__(self, cleaning_config: dict[str, Any]) -> None:
        self.cleaning_config = cleaning_config

    def fit(self, X: Tables, y: Any = None) -> CleaningAndMergeStep:
        """Fit `DataCleaner` on `X` (must be the training fold's tables only)."""
        self.cleaner_ = DataCleaner(self.cleaning_config)
        self.cleaner_.fit(X)
        return self

    def transform(self, X: Tables) -> pd.DataFrame:
        cleaned = self.cleaner_.transform(X)
        merged = point_in_time_join(
            cleaned["customers"],
            cleaned["loan_applications"],
            cleaned["financial_profiles"],
            cleaned["credit_history"],
        )
        assert_no_leakage(merged.columns)
        return merged

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """The fixed, deterministic column set `point_in_time_join` produces."""
        return np.asarray(MERGED_FRAME_COLUMNS, dtype=object)


def build_feature_pipeline(
    features_config: dict[str, Any], cleaning_config: dict[str, Any]
) -> Pipeline:
    """Build the full Phase 4 pipeline: cleaning+merge -> ratios -> behavioural
    -> `ColumnTransformer`, parameterised from `config/features.yaml`
    (`features_config`) and `config/validation_rules.yaml` (`cleaning_config`).
    """
    columns = features_config["feature_columns"]
    n_bins = features_config.get("behavioural", {}).get("n_quantile_bins", 5)
    min_frequency = features_config.get("preprocessing", {}).get("min_frequency", 0.01)

    preprocessor = build_preprocessor(
        numeric_columns=columns["numeric"],
        categorical_columns=columns["categorical"],
        ordinal_columns=columns["ordinal"],
        min_frequency=min_frequency,
    )

    return Pipeline(
        steps=[
            ("cleaning_and_merge", CleaningAndMergeStep(cleaning_config)),
            ("ratios", RatioFeatures()),
            ("behavioural", BehaviouralFeatures(n_bins=n_bins)),
            ("preprocess", preprocessor),
        ]
    )
