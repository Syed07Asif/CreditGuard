"""The single metric suite every other Phase 6 module reports through --
cross-validation, per-segment breakdowns and imbalance-strategy comparisons
all call `full_metric_suite` rather than each defining their own version of
"PR-AUC" or "Brier score."

`select_best_model` is where FR-010 is enforced in code, not just by
convention: with an ~11% default rate (see `reports/eda/findings.md`),
predicting "no default" for every applicant scores ~89% accuracy while
having zero discriminatory power, so `accuracy` is refused as a selection
metric rather than merely discouraged in a docstring.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

DEFAULT_TARGET_COLUMN = "default_12m"
FORBIDDEN_SELECTION_METRICS: frozenset[str] = frozenset({"accuracy"})


def compute_ks_statistic(y_true: Any, y_score: Any) -> float:
    """KS statistic: max separation between the score distributions of the
    positive and negative classes (the two-sample Kolmogorov-Smirnov
    statistic, computed on model scores rather than raw samples).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic)


def compute_gini(roc_auc: float) -> float:
    """Gini coefficient: `2 * ROC-AUC - 1`."""
    return float(2.0 * roc_auc - 1.0)


def calibration_slope_intercept(y_true: Any, y_prob: Any) -> tuple[float, float]:
    """Cox calibration regression: fit `y ~ logit(p)` via unregularised
    logistic regression on the single feature `logit(clipped p)`. Slope near
    1.0 means predicted probabilities spread correctly; intercept near 0.0
    means the overall predicted level matches the observed rate.
    """
    y_true = np.asarray(y_true, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)
    reg = LogisticRegression(penalty=None)
    reg.fit(logit_p, y_true)
    return float(reg.coef_[0][0]), float(reg.intercept_[0])


def confusion_counts_at_threshold(
    y_true: Any, y_prob: Any, threshold: float
) -> dict[str, int]:
    """`{tn, fp, fn, tp}` at the given decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def classification_metrics_at_threshold(
    y_true: Any, y_prob: Any, threshold: float
) -> dict[str, float]:
    """Accuracy/precision/recall/F1 at a chosen decision threshold."""
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def recall_at_precision(y_true: Any, y_prob: Any, target_precision: float) -> float:
    """Best recall achievable at precision >= `target_precision`, read off
    the precision-recall curve. 0.0 if no threshold reaches that precision.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    mask = precision >= target_precision
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def full_metric_suite(
    y_true: Any, y_prob: Any, threshold: float = 0.5
) -> dict[str, float]:
    """Every threshold-free ranking/calibration metric plus the
    threshold-dependent classification metrics, in one flat dict.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    has_both_classes = len(np.unique(y_true)) > 1
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    pr_auc = (
        float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else float("nan")
    )
    slope, intercept = calibration_slope_intercept(y_true, y_prob)
    return {
        **classification_metrics_at_threshold(y_true, y_prob, threshold),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "average_precision": pr_auc,
        "ks_statistic": compute_ks_statistic(y_true, y_prob),
        "gini": compute_gini(roc_auc) if has_both_classes else float("nan"),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": (
            float(log_loss(y_true, y_prob, labels=[0, 1]))
            if has_both_classes
            else float("nan")
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
    }


def lift_gains_table(y_true: Any, y_prob: Any, n_bins: int = 10) -> pd.DataFrame:
    """Lift/gains by decile of predicted risk. `decile == 1` is the
    *highest*-predicted-risk group (rows are sorted by descending `y_prob`
    before binning) -- the standard credit-risk reading, where decile 1
    should capture a disproportionate share of actual defaults.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_prob, dtype=float)}
    )
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    frame["decile"] = (
        pd.qcut(frame.index, q=n_bins, labels=False, duplicates="drop") + 1
    )

    total_positive = int(frame["y"].sum())
    total = len(frame)
    base_rate = total_positive / total if total else float("nan")

    grouped = frame.groupby("decile").agg(n=("y", "size"), n_positive=("y", "sum"))
    grouped = grouped.reset_index()
    grouped["default_rate"] = grouped["n_positive"] / grouped["n"]
    grouped["cum_n"] = grouped["n"].cumsum()
    grouped["cum_positive"] = grouped["n_positive"].cumsum()
    grouped["cum_gain"] = (
        grouped["cum_positive"] / total_positive if total_positive else float("nan")
    )
    grouped["lift"] = grouped["default_rate"] / base_rate
    grouped["cum_lift"] = (grouped["cum_positive"] / grouped["cum_n"]) / base_rate
    return grouped


def stratified_cv_metrics(
    model_builder: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Mean and standard deviation of every `full_metric_suite` metric across
    a fresh `StratifiedKFold(n_splits)`. `model_builder` is a zero-argument
    callable returning a new, unfitted model each fold (never reuses a
    fitted model across folds).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    rows = []
    for train_idx, test_idx in skf.split(X, y):
        model = model_builder()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = model.predict_proba(X.iloc[test_idx])
        rows.append(full_metric_suite(y.iloc[test_idx], p, threshold))
    return pd.DataFrame(rows).agg(["mean", "std"])


def per_segment_metrics(
    segment_frame: pd.DataFrame,
    y_true: Any,
    y_prob: Any,
    segment_columns: list[str],
    threshold: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """`full_metric_suite` broken out by each column in `segment_columns`
    (e.g. `loan_type`, `income_band`, `age_band`) -- exposes any segment
    where the model is materially worse than its overall performance.

    `segment_frame`, `y_true` and `y_prob` must share the same row order.
    """
    index = segment_frame.index
    y_true_s = pd.Series(np.asarray(y_true), index=index)
    y_prob_s = pd.Series(np.asarray(y_prob, dtype=float), index=index)

    result: dict[str, pd.DataFrame] = {}
    for column in segment_columns:
        rows = []
        for value, group_index in segment_frame.groupby(column).groups.items():
            metrics = full_metric_suite(
                y_true_s.loc[group_index], y_prob_s.loc[group_index], threshold
            )
            metrics[column] = value
            metrics["n"] = len(group_index)
            rows.append(metrics)
        result[column] = pd.DataFrame(rows).sort_values(column).reset_index(drop=True)
    return result


def select_best_model(results: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Pick the entry of `results` (each `{"name": ..., "metrics": {...}, ...}`)
    with the highest `metrics[metric]`.

    Raises `ValueError` if `metric == "accuracy"` (FR-010): under class
    imbalance, accuracy rewards the trivial always-predict-majority-class
    model, so it is refused here rather than merely discouraged.
    """
    if metric in FORBIDDEN_SELECTION_METRICS:
        raise ValueError(
            f"{metric!r} cannot be used to select the best model under class "
            "imbalance (FR-010) -- use 'average_precision' (PR-AUC), 'roc_auc', "
            "or another imbalance-aware metric instead."
        )
    if not results:
        raise ValueError("select_best_model got an empty results list")
    return max(results, key=lambda r: r["metrics"][metric])
