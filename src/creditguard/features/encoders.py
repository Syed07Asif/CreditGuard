"""Explicit ordering for the ordinal-encoded band columns, and the
`ColumnTransformer` assembly used by `creditguard.features.pipeline`.

`OrdinalEncoder` sorts categories alphabetically by default, which would
scramble a band column's natural order (e.g. "30-50" < "50-70" < "90+" but
also, alphabetically, "90+" < "output" -- string sort order isn't the band's
risk order at all). `ORDINAL_CATEGORY_ORDER` pins the real order explicitly.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from creditguard.features.behavioural import QUANTILE_LABELS, UTILIZATION_BAND_LABELS

ORDINAL_CATEGORY_ORDER: dict[str, list[str]] = {
    "utilization_band": list(UTILIZATION_BAND_LABELS),
    "age_band": list(QUANTILE_LABELS),
    "tenure_band": list(QUANTILE_LABELS),
    "income_band": list(QUANTILE_LABELS),
}


def build_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    ordinal_columns: list[str],
    min_frequency: float = 0.01,
) -> ColumnTransformer:
    """The Phase 4 `ColumnTransformer`: median-impute+scale numeric columns,
    impute+one-hot categorical columns, impute+explicitly-ordered-ordinal the
    `*_band` columns.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    unknown_categories = [ORDINAL_CATEGORY_ORDER[column] for column in ordinal_columns]
    ordinal_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=unknown_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
            ("ordinal", ordinal_pipeline, ordinal_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
