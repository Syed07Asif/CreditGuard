"""One consistent Matplotlib/Seaborn plotting style for every EDA chart.

Every function here takes a DataFrame/Series (or the small summary table a
`creditguard.eda.bivariate`/`risk_analysis` function already produced),
returns a `matplotlib.figure.Figure`, and accepts an optional `save_path` to
write a 150-dpi PNG. Nothing in this module calls `plt.show()` -- library
code only builds figures; the caller (a notebook cell, or
`creditguard.eda.run_eda`) decides whether to display or save them. Backend
selection (e.g. forcing headless "Agg") is also the caller's job, not this
module's, so it behaves the same whether called from a notebook or a CI run.

Colours use the Okabe-Ito palette, chosen because it is colour-blind-safe
(deuteranopia/protanopia/tritanopia) and is the de facto standard for this in
scientific plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

OKABE_ITO: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#E69F00",  # orange
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # grey
)
COLOR_NON_DEFAULT = OKABE_ITO[0]
COLOR_DEFAULT = OKABE_ITO[1]
COLOR_VOLUME = OKABE_ITO[7]
COLOR_RATE = OKABE_ITO[1]

IV_BAND_COLOR: dict[str, str] = {
    "useless": OKABE_ITO[7],
    "weak": OKABE_ITO[5],
    "medium": OKABE_ITO[0],
    "strong": OKABE_ITO[2],
    "suspiciously strong - investigate for leakage": OKABE_ITO[1],
}


def set_style() -> None:
    """Apply the one shared style used by every plot function in this module."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#4d4d4d",
        }
    )


def _save(fig: Figure, save_path: str | Path | None) -> None:
    """Write `fig` to `save_path` if given, then release it from pyplot's
    open-figure registry -- `run_eda` renders dozens of figures per run and
    would otherwise leak memory and trip matplotlib's "too many open
    figures" warning. The returned `Figure` object stays fully usable (a
    notebook cell can still display it) since closing only detaches it from
    pyplot's global registry, not from the caller's reference.
    """
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_class_balance(y: pd.Series, save_path: str | Path | None = None) -> Figure:
    """Bar chart of the two `default_12m` classes, with exact counts."""
    set_style()
    y = y.astype(int)
    n = len(y)
    counts = y.value_counts().reindex([0, 1], fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        ["Non-default", "Default"],
        counts.to_numpy(),
        color=[COLOR_NON_DEFAULT, COLOR_DEFAULT],
    )
    for i, v in enumerate(counts.to_numpy()):
        ax.text(i, v, f"{v:,}\n({v / n:.1%})", ha="center", va="bottom")
    ax.set_ylabel("Number of loans")
    ax.set_title(f"Class balance: default_12m (n={n:,})")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_numeric_distribution(
    df: pd.DataFrame,
    column: str,
    unit: str = "",
    save_path: str | Path | None = None,
) -> Figure:
    """Histogram + boxplot of one numeric column, sharing the x-axis."""
    set_style()
    series = df[column].dropna().astype(float)
    n = len(series)
    label = f"{column}{f' ({unit})' if unit else ''}"

    fig, (ax_hist, ax_box) = plt.subplots(
        2, 1, figsize=(6, 5), sharex=True, gridspec_kw={"height_ratios": [4, 1]}
    )
    ax_hist.hist(series, bins=50, color=COLOR_NON_DEFAULT, edgecolor="white")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title(f"Distribution of {column} (n={n:,})")

    ax_box.boxplot(series, orientation="horizontal", widths=0.6)
    ax_box.set_yticks([])
    ax_box.set_xlabel(label)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_categorical_frequency(
    df: pd.DataFrame, column: str, save_path: str | Path | None = None
) -> Figure:
    """Horizontal bar chart of category counts, largest first."""
    set_style()
    counts = df[column].value_counts(dropna=False)
    n = len(df)

    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.5 * len(counts))))
    ax.barh(counts.index.astype(str)[::-1], counts.to_numpy()[::-1], color=OKABE_ITO[0])
    for i, v in enumerate(counts.to_numpy()[::-1]):
        ax.text(v, i, f" {v:,} ({v / n:.1%})", va="center")
    ax.set_xlabel("Number of loans")
    ax.set_title(f"{column} frequency (n={n:,})")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_default_rate_by_decile(
    decile_df: pd.DataFrame,
    unit: str = "",
    save_path: str | Path | None = None,
) -> Figure:
    """Dual-axis chart: bar volume per decile, line for default rate --
    "the single most useful chart in credit risk."
    """
    set_style()
    feature = decile_df["feature"].iloc[0]
    n_total = int(decile_df["n"].sum())
    label = f"{feature}{f' ({unit})' if unit else ''}"

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(
        decile_df["decile"], decile_df["n"], color=COLOR_VOLUME, alpha=0.6, label="n"
    )
    ax1.set_xlabel(f"Decile of {label} (low -> high)")
    ax1.set_ylabel("Number of loans", color=COLOR_VOLUME)
    ax1.set_xticks(decile_df["decile"])

    ax2 = ax1.twinx()
    ax2.plot(
        decile_df["decile"],
        decile_df["default_rate"],
        color=COLOR_RATE,
        marker="o",
        linewidth=2,
    )
    ax2.set_ylabel("Default rate", color=COLOR_RATE)
    ax2.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    ax2.set_ylim(bottom=0)

    ax1.set_title(f"Default rate by decile of {feature} (n={n_total:,})")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_default_rate_by_band(
    band_df: pd.DataFrame,
    band_col: str,
    save_path: str | Path | None = None,
) -> Figure:
    """Bar chart of default rate per category/band, with `n` annotated per bar."""
    set_style()
    n_total = int(band_df["n"].sum())

    fig, ax = plt.subplots(figsize=(max(5, 0.9 * len(band_df)), 4.5))
    labels = band_df[band_col].astype(str)
    ax.bar(labels, band_df["default_rate"], color=OKABE_ITO[0])
    for i, (rate, n) in enumerate(
        zip(band_df["default_rate"], band_df["n"], strict=True)
    ):
        ax.text(i, rate, f"{rate:.1%}\n(n={n:,})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Default rate")
    ax.set_xlabel(band_col)
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x:.0%}")
    ax.set_title(f"Default rate by {band_col} (n={n_total:,})")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_iv_table(iv_df: pd.DataFrame, save_path: str | Path | None = None) -> Figure:
    """Horizontal bar chart of IV per feature, sorted descending, coloured
    by interpretation band.
    """
    set_style()
    ordered = iv_df.sort_values("iv")
    colors = [
        IV_BAND_COLOR.get(label, OKABE_ITO[7]) for label in ordered["interpretation"]
    ]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.3 * len(ordered))))
    ax.barh(ordered["feature"], ordered["iv"], color=colors)
    for threshold in (0.02, 0.1, 0.3, 0.5):
        ax.axvline(threshold, color="#4d4d4d", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Information Value (IV)")
    ax.set_title(f"Information Value by feature ({len(ordered)} features)")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_correlation_heatmap(
    corr: pd.DataFrame, save_path: str | Path | None = None
) -> Figure:
    """Heatmap of a numeric feature correlation matrix."""
    set_style()
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(corr)), max(7, 0.4 * len(corr))))
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title(f"Correlation heatmap ({len(corr)} numeric features)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def plot_temporal_trend(
    monthly_df: pd.DataFrame, save_path: str | Path | None = None
) -> Figure:
    """Dual-axis chart: application volume and default rate by month."""
    set_style()
    n_total = int(monthly_df["n"].sum())

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(monthly_df))
    ax1.bar(x, monthly_df["n"], color=COLOR_VOLUME, alpha=0.6, label="n")
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly_df["month"], rotation=60, ha="right")
    ax1.set_ylabel("Applications", color=COLOR_VOLUME)
    ax1.set_xlabel("Month")

    ax2 = ax1.twinx()
    ax2.plot(x, monthly_df["default_rate"], color=COLOR_RATE, marker="o", linewidth=2)
    ax2.set_ylabel("Default rate", color=COLOR_RATE)
    ax2.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0%}")
    ax2.set_ylim(bottom=0)

    ax1.set_title(f"Application volume and default rate by month (n={n_total:,})")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
