"""Random forest: a bagged-tree baseline between the linear model and
gradient boosting -- handles non-linear interactions without XGBoost's
extra tuning surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from creditguard.models.base import BaseCreditModel


class RandomForestModel(BaseCreditModel):
    """Bagged decision trees via `sklearn.ensemble.RandomForestClassifier`."""

    name = "random_forest"

    def build(self, params: dict[str, Any]) -> RandomForestClassifier:
        self.model_ = RandomForestClassifier(**params)
        return self.model_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> RandomForestModel:
        if self.model_ is None:
            self.build(self.params)
        self.feature_names_ = list(X.columns)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        importances = pd.Series(
            self.model_.feature_importances_, index=self.feature_names_
        )
        return importances.sort_values(ascending=False)
