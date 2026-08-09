"""Logistic regression: the interpretable baseline, kept in every comparison
even when a more flexible model wins on PR-AUC -- coefficients are directly
readable, which matters for `docs/scoring_methodology.md`-style explanations
later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from creditguard.models.base import BaseCreditModel


class LogisticRegressionModel(BaseCreditModel):
    """L2 or elastic-net logistic regression via the `saga` solver (the one
    solver that supports both penalties plus `class_weight`).
    """

    name = "logistic_regression"

    def build(self, params: dict[str, Any]) -> LogisticRegression:
        params = dict(params)
        if params.get("penalty") != "elasticnet":
            params.pop("l1_ratio", None)
        self.model_ = LogisticRegression(**params)
        return self.model_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> LogisticRegressionModel:
        if self.model_ is None:
            self.build(self.params)
        self.feature_names_ = list(X.columns)
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        coefs = pd.Series(self.model_.coef_.ravel(), index=self.feature_names_)
        return coefs.reindex(coefs.abs().sort_values(ascending=False).index)
