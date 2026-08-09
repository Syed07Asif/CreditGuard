"""XGBoost: gradient-boosted trees, the strongest of the three families on
tabular data with interactions. Supports early stopping on a real held-out
validation set via `fit(..., eval_set=(X_val, y_val))` -- pass
`early_stopping_rounds` in `params` (see `config/model_config.yaml`) and an
`eval_set` at fit time together; early stopping without an eval_set is a
caller error XGBoost itself raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from creditguard.models.base import BaseCreditModel


class XGBoostModel(BaseCreditModel):
    """Gradient-boosted trees via `xgboost.XGBClassifier`."""

    name = "xgboost"

    def build(self, params: dict[str, Any]) -> XGBClassifier:
        self.model_ = XGBClassifier(**params)
        return self.model_

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> XGBoostModel:
        if self.model_ is None:
            self.build(self.params)
        self.feature_names_ = list(X.columns)
        if eval_set is not None:
            X_val, y_val = eval_set
            self.model_.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)[:, 1]

    def feature_importance(self) -> pd.Series:
        importances = pd.Series(
            self.model_.feature_importances_, index=self.feature_names_
        )
        return importances.sort_values(ascending=False)
