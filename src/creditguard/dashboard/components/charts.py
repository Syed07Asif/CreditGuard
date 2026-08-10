"""Matplotlib chart builders for the dashboard -- same Okabe-Ito palette and
`whitegrid` style as `creditguard.eda.plots` (the Phase 9 brief's
"consistent theme... same colour palette as the EDA figures" requirement),
kept as an independent module rather than importing `creditguard.eda` (a
Phase 5 module with its own plotting conventions unrelated to a live page).

Every function takes plain data (a DataFrame, dict or list of dicts --
never a model/pipeline object) and returns a `matplotlib.figure.Figure`.
Every function tolerates an empty/near-empty input by returning a figure
with a "no data" message instead of raising, so a page can call
`st.pyplot(fig)` unconditionally.
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from creditguard.dashboard.state import (  # noqa: E402
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_POSITIVE,
    COLOR_VOLUME,
    OKABE_ITO,
    RISK_BAND_COLORS,
    SCORE_BAND_RANGES,
)

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        "figure.dpi": 100,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": "#4d4d4d",
    }
)


def _empty_figure(message: str = "No data available") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, color="#666666")
    ax.set_axis_off()
    return fig


def score_gauge(score: int, risk_category: str) -> plt.Figure:
    """A 300-900 semicircular gauge, coloured by risk band, with a needle
    at `score`. Streamlit has no built-in gauge widget, and the brief
    disallows adding a heavyweight charting library beyond the fixed
    stack's Matplotlib/Seaborn -- this draws one directly with `Wedge`.
    """
    from matplotlib.patches import Wedge

    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"aspect": "equal"})
    min_score, max_score = 300, 900
    span = max_score - min_score

    for band_name, band_min, band_max in SCORE_BAND_RANGES:
        theta1 = 180 * (1 - (band_max - min_score) / span)
        theta2 = 180 * (1 - (band_min - min_score) / span)
        ax.add_patch(
            Wedge(
                (0, 0),
                1.0,
                theta1,
                theta2,
                width=0.35,
                facecolor=RISK_BAND_COLORS[band_name],
                edgecolor="white",
            )
        )

    clamped = max(min_score, min(max_score, score))
    angle_deg = 180 * (1 - (clamped - min_score) / span)
    angle_rad = np.deg2rad(angle_deg)
    ax.plot(
        [0, 0.62 * np.cos(angle_rad)],
        [0, 0.62 * np.sin(angle_rad)],
        color="#222222",
        linewidth=3,
    )
    ax.add_patch(plt.Circle((0, 0), 0.05, color="#222222"))

    ax.text(
        0, -0.25, f"{score}", ha="center", va="center", fontsize=28, fontweight="bold"
    )
    ax.text(
        0,
        -0.45,
        risk_category.replace("_", " "),
        ha="center",
        va="center",
        fontsize=12,
        color=RISK_BAND_COLORS.get(risk_category, "#333333"),
    )
    ax.text(-1.05, 0.05, str(min_score), ha="center", fontsize=9, color="#666666")
    ax.text(1.05, 0.05, str(max_score), ha="center", fontsize=9, color="#666666")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.55, 1.1)
    ax.set_axis_off()
    return fig


def shap_contribution_chart(
    risk_factors: list[dict[str, Any]], positive_factors: list[dict[str, Any]]
) -> plt.Figure:
    """Horizontal SHAP-style bar chart: risk factors (positive impact on
    default probability) in red on one side, positive factors (negative
    impact) in green on the other, sorted so the largest magnitude bars sit
    at the top/bottom edges nearest the axis.
    """
    if not risk_factors and not positive_factors:
        return _empty_figure("No contributing factors returned")

    rows = [
        {"label": f["feature"], "impact": abs(f["impact"]), "color": COLOR_NEGATIVE}
        for f in risk_factors
    ] + [
        {"label": f["feature"], "impact": -abs(f["impact"]), "color": COLOR_POSITIVE}
        for f in positive_factors
    ]
    frame = pd.DataFrame(rows).sort_values("impact")

    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.4 * len(frame))))
    ax.barh(frame["label"], frame["impact"], color=frame["color"])
    ax.axvline(0, color="#4d4d4d", linewidth=0.8)
    ax.set_xlabel("SHAP contribution to default probability")
    ax.set_title("Risk factors (red) vs. positive factors (green)")
    fig.tight_layout()
    return fig


def score_distribution_histogram(scores: pd.Series) -> plt.Figure:
    if scores.empty:
        return _empty_figure()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=30, color=COLOR_VOLUME, edgecolor="white")
    ax.set_xlabel("Credit score")
    ax.set_ylabel("Number of applications")
    ax.set_title("Score distribution")
    fig.tight_layout()
    return fig


def risk_category_bar(counts: pd.Series, order: tuple[str, ...]) -> plt.Figure:
    if counts.empty:
        return _empty_figure()
    counts = counts.reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [RISK_BAND_COLORS.get(name, COLOR_NEUTRAL) for name in counts.index]
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=colors)
    ax.set_ylabel("Applications")
    ax.set_title("Risk category distribution")
    fig.tight_layout()
    return fig


def applications_over_time(
    daily: pd.DataFrame, rate_label: str = "Predicted default rate"
) -> plt.Figure:
    """`daily` needs columns `date`, `n`, `default_rate` -- the caller
    decides what "default rate" means for live (not-yet-outcome-observed)
    predictions, e.g. the share at/above the model's chosen threshold.
    """
    if daily.empty:
        return _empty_figure()
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(
        daily["date"], daily["n"], color=COLOR_VOLUME, alpha=0.6, label="Applications"
    )
    ax1.set_ylabel("Applications", color=COLOR_VOLUME)
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()
    ax2.plot(
        daily["date"],
        daily["default_rate"],
        color=COLOR_NEGATIVE,
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel(rate_label, color=COLOR_NEGATIVE)
    ax2.set_ylim(0, 1)

    ax1.set_title(f"Applications and {rate_label.lower()} over time")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def rate_by_segment_bar(
    segment_rates: pd.Series,
    ylabel: str,
    title: str,
    order: tuple[str, ...] | None = None,
) -> plt.Figure:
    if segment_rates.empty:
        return _empty_figure()
    if order is not None:
        segment_rates = segment_rates.reindex(
            [o for o in order if o in segment_rates.index]
        )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        segment_rates.index.astype(str), segment_rates.to_numpy(), color=OKABE_ITO[0]
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig


def roc_curve_chart(
    fpr: list[float], tpr: list[float], roc_auc: float | None = None
) -> plt.Figure:
    if not fpr or not tpr:
        return _empty_figure()
    fig, ax = plt.subplots(figsize=(5, 5))
    label = f"ROC (AUC={roc_auc:.3f})" if roc_auc is not None else "ROC"
    ax.plot(fpr, tpr, color=OKABE_ITO[0], linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_curve_chart(
    precision: list[float], recall: list[float], pr_auc: float | None = None
) -> plt.Figure:
    if not precision or not recall:
        return _empty_figure()
    fig, ax = plt.subplots(figsize=(5, 5))
    label = f"PR (AUC={pr_auc:.3f})" if pr_auc is not None else "PR"
    ax.plot(recall, precision, color=OKABE_ITO[1], linewidth=2, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def confusion_matrix_heatmap(confusion: dict[str, int]) -> plt.Figure:
    if not confusion:
        return _empty_figure()
    matrix = np.array(
        [[confusion["tn"], confusion["fp"]], [confusion["fn"], confusion["tp"]]]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred: No default", "Pred: Default"],
        yticklabels=["Actual: No default", "Actual: Default"],
        ax=ax,
    )
    ax.set_title("Confusion matrix (at chosen threshold)")
    fig.tight_layout()
    return fig


def calibration_chart(
    mean_predicted: list[float], fraction_positive: list[float]
) -> plt.Figure:
    if not mean_predicted or not fraction_positive:
        return _empty_figure()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(
        mean_predicted,
        fraction_positive,
        color=OKABE_ITO[2],
        marker="o",
        linewidth=2,
        label="Model",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="#999999",
        linestyle="--",
        linewidth=1,
        label="Perfectly calibrated",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration / reliability")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def lift_gains_chart(lift_gains: list[dict[str, Any]]) -> plt.Figure:
    if not lift_gains:
        return _empty_figure()
    frame = pd.DataFrame(lift_gains).sort_values("decile")
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(
        frame["decile"],
        frame["cum_gain"],
        color=COLOR_VOLUME,
        alpha=0.7,
        label="Cumulative gain",
    )
    ax1.set_xlabel("Decile (1 = highest predicted risk)")
    ax1.set_ylabel("Cumulative gain", color=COLOR_VOLUME)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        frame["decile"],
        frame["lift"],
        color=COLOR_NEGATIVE,
        marker="o",
        linewidth=2,
        label="Lift",
    )
    ax2.set_ylabel("Lift", color=COLOR_NEGATIVE)

    ax1.set_title("Gains and lift by decile")
    fig.tight_layout()
    return fig


def feature_importance_chart(
    items: list[dict[str, Any]], top_n: int = 20
) -> plt.Figure:
    if not items:
        return _empty_figure()
    frame = pd.DataFrame(items).head(top_n).iloc[::-1]
    colors = [COLOR_NEGATIVE if v < 0 else COLOR_POSITIVE for v in frame["importance"]]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(frame))))
    ax.barh(frame["feature"], frame["importance"], color=colors)
    ax.axvline(0, color="#4d4d4d", linewidth=0.8)
    ax.set_xlabel("Importance (model coefficient)")
    ax.set_title(f"Top {len(frame)} features by importance")
    fig.tight_layout()
    return fig
