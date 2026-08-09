"""Abstract model interface every concrete model implements, so the trainer
and (later) the API never branch on model type.

`save`/`load` are implemented once here via `joblib` since persistence is
identical across every concrete model (they're all picklable Python objects
holding a fitted scikit-learn-compatible estimator); `build`/`fit`/
`predict_proba`/`feature_importance` are abstract because each model family
does them differently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class BaseCreditModel(ABC):
    """Common interface for every credit-risk model family.

    Concrete subclasses store the fitted underlying estimator on
    `self.model_` and the training feature names on `self.feature_names_`
    (set from `X.columns` in `fit`, so `feature_importance` can return a
    properly-indexed `pd.Series` regardless of model family).
    """

    name: str

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params) if params else {}
        self.model_: Any = None
        self.feature_names_: list[str] | None = None

    @abstractmethod
    def build(self, params: dict[str, Any]) -> Any:
        """Construct and return the untrained underlying estimator."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
    ) -> BaseCreditModel:
        """Fit on training data. `eval_set`, if given, is used for early
        stopping where the model family supports it (XGBoost); ignored
        otherwise. Returns self.
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(default_12m == 1) for each row, shape (n_samples,)."""

    def get_params(self) -> dict[str, Any]:
        """Return a copy of the hyperparameters this instance was built with."""
        return dict(self.params)

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Feature importances/coefficients, indexed by feature name,
        sorted descending by magnitude.
        """

    def save(self, path: str | Path) -> None:
        """Persist this fitted model (wrapper and all) via joblib."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseCreditModel:
        """Load a model previously written by `save`."""
        return joblib.load(path)
