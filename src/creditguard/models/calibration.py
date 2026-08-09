"""Calibration: wrap the selected, already-fitted model in
`CalibratedClassifierCV`, pick isotonic vs sigmoid by Brier score on the
validation set, and report before/after calibration quality plus a
reliability diagram. The returned calibrated estimator -- not the raw base
model -- is what `train.py` registers.

sklearn >=1.6 removed `CalibratedClassifierCV(cv="prefit")` in favour of
wrapping the already-fitted estimator in `sklearn.frozen.FrozenEstimator`;
this module targets that current API (installed sklearn is 1.9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

from creditguard.models.evaluate import calibration_slope_intercept

CALIBRATION_METHODS: tuple[str, ...] = ("isotonic", "sigmoid")


def calibrate_model(
    fitted_estimator: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    methods: tuple[str, ...] = CALIBRATION_METHODS,
) -> dict[str, Any]:
    """Calibrate `fitted_estimator` (already fit on the *train* split) using
    `X_val`/`y_val`, trying every method in `methods` and keeping whichever
    gives the lower Brier score on that same validation set.

    Returns a dict with the winning `calibrated_model`
    (a `CalibratedClassifierCV`), its `method`, and before/after Brier score
    and calibration slope/intercept -- all measured on `X_val`/`y_val`, so
    "before" (the uncalibrated base estimator) and "after" are directly
    comparable.
    """
    p_before = fitted_estimator.predict_proba(X_val)[:, 1]
    brier_before = float(brier_score_loss(y_val, p_before))
    slope_before, intercept_before = calibration_slope_intercept(y_val, p_before)

    frozen = FrozenEstimator(fitted_estimator)
    candidates: dict[str, tuple[CalibratedClassifierCV, float]] = {}
    for method in methods:
        calibrated = CalibratedClassifierCV(frozen, method=method)
        calibrated.fit(X_val, y_val)
        p = calibrated.predict_proba(X_val)[:, 1]
        candidates[method] = (calibrated, float(brier_score_loss(y_val, p)))

    best_method = min(candidates, key=lambda m: candidates[m][1])
    best_model, brier_after = candidates[best_method]
    p_after = best_model.predict_proba(X_val)[:, 1]
    slope_after, intercept_after = calibration_slope_intercept(y_val, p_after)

    return {
        "calibrated_model": best_model,
        "method": best_method,
        "brier_before": brier_before,
        "brier_after": brier_after,
        "calibration_slope_before": slope_before,
        "calibration_slope_after": slope_after,
        "calibration_intercept_before": intercept_before,
        "calibration_intercept_after": intercept_after,
        "candidate_brier_scores": {m: brier for m, (_, brier) in candidates.items()},
        "p_before": p_before,
        "p_after": p_after,
    }


def plot_reliability_diagram(
    y_val: Any,
    p_before: Any,
    p_after: Any,
    save_path: str | Path | None = None,
    n_bins: int = 10,
) -> Figure:
    """Reliability diagram: observed default rate vs. mean predicted
    probability, per quantile bin, before and after calibration.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        [0, 1], [0, 1], linestyle="--", color="#999999", label="Perfectly calibrated"
    )

    frac_before, mean_before = calibration_curve(
        y_val, p_before, n_bins=n_bins, strategy="quantile"
    )
    frac_after, mean_after = calibration_curve(
        y_val, p_after, n_bins=n_bins, strategy="quantile"
    )
    ax.plot(
        mean_before,
        frac_before,
        marker="o",
        color="#D55E00",
        label="Before calibration",
    )
    ax.plot(
        mean_after, frac_after, marker="o", color="#0072B2", label="After calibration"
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Reliability diagram (n={len(y_val):,})")
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
