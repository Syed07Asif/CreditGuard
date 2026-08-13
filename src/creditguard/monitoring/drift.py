"""Drift detection against a promoted model's baseline (`monitoring/baseline.py`).

Five kinds of check, each producing one or more `drift_reports` rows:

  * Population Stability Index (numeric features, binned into the
    baseline's own decile edges -- never rebinned from current data, or the
    statistic would just measure how spread out "today" is against itself)
    and (categorical features, via category-frequency bins).
  * A genuine two-sample Kolmogorov-Smirnov test for continuous features,
    against the baseline's persisted reference sample.
  * A chi-square test for categorical features, plus an explicit
    new-category check (a category with zero baseline mass forces DRIFT
    regardless of the chi-square p-value, since chi-square alone can miss a
    single popular new category if it's small relative to the whole table).
  * Prediction drift: PSI on the scored population's predicted-probability
    distribution -- reported separately and first, since a shift here often
    precedes any single feature crossing its own DRIFT threshold (the
    earliest warning signal per the Phase 10 brief).
  * A concept-drift proxy: the rolling observed default rate among matured
    loans vs. the baseline's training-time default rate, with a Wilson-score
    binomial confidence interval (more reliable than the naive normal
    approximation at typical monitoring-window sample sizes).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import select

from creditguard.config import get_settings
from creditguard.db.engine import get_engine
from creditguard.db.models import Prediction
from creditguard.db.repository import DriftReportRepository
from creditguard.features.leakage import point_in_time_join
from creditguard.models import registry
from creditguard.monitoring.baseline import (
    BaselineProfile,
    load_baseline,
    load_baseline_sample,
    load_monitoring_config,
)
from creditguard.monitoring.data_quality import fetch_production_tables

STATUS_OK = "OK"
STATUS_WARNING = "WARNING"
STATUS_DRIFT = "DRIFT"

PREDICTION_PROBABILITY_FEATURE_NAME = "__prediction_probability__"


@dataclass(frozen=True)
class DriftFinding:
    """One `drift_reports` row's worth of data."""

    feature_name: str
    method: str
    baseline_stat: float
    current_stat: float
    drift_score: float
    status: str
    detail: str = ""


@dataclass(frozen=True)
class ConceptDriftResult:
    n: int
    current_rate: float
    ci_low: float
    ci_high: float
    baseline_rate: float
    status: str


@dataclass(frozen=True)
class DriftRunResult:
    model_id: str
    window_start: date
    window_end: date
    findings: list[DriftFinding]
    concept_drift: ConceptDriftResult
    n_rows_scored: int
    report_md_path: Path | None = None
    report_html_path: Path | None = None
    heatmap_path: Path | None = None

    @property
    def any_drift(self) -> bool:
        return any(f.status == STATUS_DRIFT for f in self.findings)

    @property
    def prediction_drift(self) -> DriftFinding | None:
        for finding in self.findings:
            if finding.feature_name == PREDICTION_PROBABILITY_FEATURE_NAME:
                return finding
        return None


def psi_from_percentages(
    expected_pct: np.ndarray, actual_pct: np.ndarray, epsilon: float
) -> float:
    """Population Stability Index: `sum((actual% - expected%) * ln(actual% /
    expected%))` over bins. Bins are floored to `epsilon` on both sides
    before the log/ratio -- an exactly-empty expected or actual bin would
    otherwise produce `ln(0)` or a division by zero. There is no single
    "correct" empty-bin treatment in the PSI literature; flooring to a small
    epsilon is the conventional choice (documented here, per the Phase 10
    brief's "handle empty bins with a small epsilon and document the
    choice") -- it keeps the statistic finite while still penalising the
    emptied/filled bin heavily.
    """
    expected = np.clip(np.asarray(expected_pct, dtype=float), epsilon, None)
    actual = np.clip(np.asarray(actual_pct, dtype=float), epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi_value: float, warning: float, drift: float) -> str:
    if psi_value >= drift:
        return STATUS_DRIFT
    if psi_value >= warning:
        return STATUS_WARNING
    return STATUS_OK


def numeric_psi(
    baseline_edges: list[float], current_values: pd.Series, epsilon: float
) -> float:
    """PSI for one numeric feature, current values binned into the
    baseline's own decile edges (equal-mass bins by construction at
    baseline time, so `expected_pct` is uniform).
    """
    edges = np.asarray(baseline_edges, dtype=float)
    values = pd.to_numeric(current_values, errors="coerce").dropna().to_numpy()
    if len(values) == 0 or len(edges) < 2:
        return 0.0
    current_counts, _ = np.histogram(values, bins=edges)
    total = current_counts.sum()
    if total == 0:
        return 0.0
    current_pct = current_counts / total
    expected_pct = np.full(len(current_counts), 1.0 / len(current_counts))
    return psi_from_percentages(expected_pct, current_pct, epsilon)


def categorical_psi(
    baseline_frequencies: dict[str, float], current_values: pd.Series, epsilon: float
) -> tuple[float, list[str]]:
    """PSI for one categorical feature, plus any category present in
    `current_values` that never appeared in the baseline.
    """
    current = current_values.astype(str)
    if current.empty:
        return 0.0, []
    current_freq = current.value_counts(normalize=True)
    categories = sorted(set(baseline_frequencies) | set(current_freq.index))
    expected_pct = np.array([baseline_frequencies.get(c, 0.0) for c in categories])
    actual_pct = np.array([float(current_freq.get(c, 0.0)) for c in categories])
    new_categories = sorted(set(current_freq.index) - set(baseline_frequencies))
    return psi_from_percentages(expected_pct, actual_pct, epsilon), new_categories


def ks_test_feature(
    baseline_values: pd.Series, current_values: pd.Series
) -> tuple[float, float]:
    """Two-sample KS test between the baseline's persisted reference sample
    and current production values for one continuous feature.
    """
    baseline = pd.to_numeric(baseline_values, errors="coerce").dropna()
    current = pd.to_numeric(current_values, errors="coerce").dropna()
    if len(baseline) < 2 or len(current) < 2:
        return float("nan"), float("nan")
    result = stats.ks_2samp(baseline, current)
    return float(result.statistic), float(result.pvalue)


def chi_square_test_feature(
    baseline_frequencies: dict[str, float], current_values: pd.Series
) -> tuple[float, float, list[str]]:
    """Chi-square goodness-of-fit test: current category counts vs. the
    counts implied by baseline frequencies at the current sample size, plus
    the same new-category check `categorical_psi` reports.
    """
    current = current_values.astype(str)
    if current.empty:
        return float("nan"), float("nan"), []
    current_counts = current.value_counts()
    categories = sorted(set(baseline_frequencies) | set(current_counts.index))
    observed = np.array([float(current_counts.get(c, 0)) for c in categories])
    baseline_weights = np.array([baseline_frequencies.get(c, 0.0) for c in categories])
    if baseline_weights.sum() <= 0:
        return float("nan"), float("nan"), []
    expected = baseline_weights / baseline_weights.sum() * observed.sum()
    expected = np.clip(expected, 1e-6, None)
    # Renormalise so observed/expected totals match exactly (scipy requires
    # this to the precision of its own internal check).
    expected = expected * (observed.sum() / expected.sum())
    statistic, p_value = stats.chisquare(observed, f_exp=expected)
    new_categories = sorted(set(current_counts.index) - set(baseline_frequencies))
    return float(statistic), float(p_value), new_categories


def concept_drift_proxy(
    baseline_default_rate: float,
    matured_outcomes: pd.Series,
    confidence: float = 0.95,
) -> ConceptDriftResult:
    """Rolling default rate among matured loans vs. the baseline training
    default rate, with a Wilson-score binomial confidence interval on the
    current rate.
    """
    n = len(matured_outcomes)
    if n == 0:
        return ConceptDriftResult(
            n=0,
            current_rate=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            baseline_rate=baseline_default_rate,
            status=STATUS_OK,
        )
    successes = int(matured_outcomes.sum())
    current_rate = successes / n
    z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
    denom = 1 + z**2 / n
    center = (current_rate + z**2 / (2 * n)) / denom
    margin = (
        z * math.sqrt((current_rate * (1 - current_rate) + z**2 / (4 * n)) / n)
    ) / denom
    ci_low, ci_high = max(0.0, center - margin), min(1.0, center + margin)
    baseline_outside_ci = not (ci_low <= baseline_default_rate <= ci_high)
    return ConceptDriftResult(
        n=n,
        current_rate=current_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        baseline_rate=baseline_default_rate,
        status=STATUS_WARNING if baseline_outside_ci else STATUS_OK,
    )


def build_current_feature_frame(window_start: date, window_end: date) -> pd.DataFrame:
    """Production applications in `[window_start, window_end]`, run through
    the active model's own fitted ratios/behavioural transformer -- the
    same human-readable feature shape `monitoring/baseline.py`'s
    `train_frame` used to build the baseline, so PSI/KS/chi-square compare
    like with like.
    """
    from creditguard.scoring import engine as scoring_engine

    tables = fetch_production_tables(window_start, window_end)
    if tables["loan_applications"].empty:
        return pd.DataFrame()
    merged = point_in_time_join(
        tables["customers"],
        tables["loan_applications"],
        tables["financial_profiles"],
        tables["credit_history"],
    )
    loaded = scoring_engine._get_loaded_model()
    return loaded.ratios_behavioural.transform(merged)


def fetch_predictions_window(
    model_id: str, window_start: datetime, window_end: datetime
) -> pd.DataFrame:
    """Every `predictions` row for `model_id` scored within the window --
    the scored population's own predicted-probability distribution, used
    for prediction drift.
    """
    with get_engine().connect() as conn:
        return pd.read_sql(
            select(Prediction).where(
                Prediction.model_id == model_id,
                Prediction.created_at >= window_start,
                Prediction.created_at <= window_end,
            ),
            conn,
        )


def run_feature_drift(
    baseline: BaselineProfile,
    baseline_sample: pd.DataFrame,
    current_frame: pd.DataFrame,
    *,
    psi_warning: float,
    psi_drift: float,
    epsilon: float,
    significance_level: float,
) -> list[DriftFinding]:
    """PSI + KS (numeric) and PSI + chi-square (categorical) for every
    feature the baseline covers that's also present in `current_frame`.
    """
    findings: list[DriftFinding] = []

    for name, feat in baseline.numeric_features.items():
        if name not in current_frame.columns:
            continue
        psi_value = numeric_psi(feat.decile_edges, current_frame[name], epsilon)
        findings.append(
            DriftFinding(
                feature_name=name,
                method="psi",
                baseline_stat=0.0,
                current_stat=psi_value,
                drift_score=psi_value,
                status=psi_status(psi_value, psi_warning, psi_drift),
            )
        )
        if name in baseline_sample.columns:
            statistic, p_value = ks_test_feature(
                baseline_sample[name], current_frame[name]
            )
            ks_status = (
                STATUS_DRIFT
                if (not math.isnan(p_value) and p_value < significance_level)
                else STATUS_OK
            )
            findings.append(
                DriftFinding(
                    feature_name=name,
                    method="ks",
                    baseline_stat=0.0,
                    current_stat=statistic,
                    drift_score=statistic,
                    status=ks_status,
                    detail=f"p_value={p_value:.4g}",
                )
            )

    for name, categorical_feat in baseline.categorical_features.items():
        if name not in current_frame.columns:
            continue
        psi_value, new_categories = categorical_psi(
            categorical_feat.frequencies, current_frame[name], epsilon
        )
        status = psi_status(psi_value, psi_warning, psi_drift)
        detail = ""
        if new_categories:
            status = STATUS_DRIFT
            detail = f"new categories not seen in training: {new_categories}"
        findings.append(
            DriftFinding(
                feature_name=name,
                method="psi",
                baseline_stat=0.0,
                current_stat=psi_value,
                drift_score=psi_value,
                status=status,
                detail=detail,
            )
        )
        chi2_stat, chi2_p, chi2_new = chi_square_test_feature(
            categorical_feat.frequencies, current_frame[name]
        )
        chi2_status = (
            STATUS_DRIFT
            if (not math.isnan(chi2_p) and chi2_p < significance_level) or chi2_new
            else STATUS_OK
        )
        findings.append(
            DriftFinding(
                feature_name=name,
                method="chi_square",
                baseline_stat=0.0,
                current_stat=chi2_stat,
                drift_score=chi2_stat,
                status=chi2_status,
                detail=f"p_value={chi2_p:.4g}",
            )
        )

    return findings


def run_prediction_drift(
    baseline: BaselineProfile,
    current_probabilities: pd.Series,
    *,
    psi_warning: float,
    psi_drift: float,
    epsilon: float,
) -> DriftFinding:
    """PSI on the scored population's predicted-probability distribution --
    reported separately and first: this is the earliest warning signal,
    since a shift in overall predicted risk can appear before any single
    input feature crosses its own DRIFT threshold.
    """
    psi_value = numeric_psi(
        baseline.prediction_probability_deciles, current_probabilities, epsilon
    )
    return DriftFinding(
        feature_name=PREDICTION_PROBABILITY_FEATURE_NAME,
        method="psi",
        baseline_stat=0.0,
        current_stat=psi_value,
        drift_score=psi_value,
        status=psi_status(psi_value, psi_warning, psi_drift),
        detail="prediction-probability distribution drift (earliest warning signal)",
    )


def persist_drift_findings(model_id: str, findings: list[DriftFinding]) -> int:
    records = [
        {
            "model_id": model_id,
            "feature_name": f.feature_name,
            "method": f.method,
            "baseline_stat": f.baseline_stat,
            "current_stat": f.current_stat,
            "drift_score": f.drift_score,
            "status": f.status,
        }
        for f in findings
    ]
    return DriftReportRepository().insert_many(records)


def _psi_heatmap(model_id: str, reports_dir: Path) -> Path | None:
    """A heatmap of PSI (feature x run date) built from every historical
    `drift_reports` row for `model_id` -- shows whether a feature's drift
    is a one-off blip or a sustained trend.
    """
    history = DriftReportRepository().fetch_dataframe(filters={"model_id": model_id})
    history = history.loc[history["method"] == "psi"]
    if history.empty:
        return None

    history = history.copy()
    history["run_date"] = pd.to_datetime(history["created_at"]).dt.date
    # Postgres NUMERIC columns come back as decimal.Decimal via psycopg, not
    # a native float -- pivot_table would otherwise build an object-dtype
    # table matplotlib's imshow can't render ("Image data of dtype object
    # cannot be converted to float").
    history["drift_score"] = history["drift_score"].astype(float)
    pivot = history.pivot_table(
        index="feature_name", columns="run_date", values="drift_score", aggfunc="mean"
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(
        figsize=(max(6, len(pivot.columns) * 0.6), max(4, len(pivot) * 0.35))
    )
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_title(f"PSI by feature over time -- {model_id}")
    fig.colorbar(im, ax=ax, label="PSI")
    fig.tight_layout()

    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{model_id}_psi_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_drift_report(
    result: DriftRunResult, reports_dir: Path
) -> tuple[Path, Path, Path | None]:
    """A markdown + HTML drift report, plus a PSI-over-time heatmap."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        f"# Drift report -- {result.model_id}",
        "",
        f"Window: {result.window_start} to {result.window_end} "
        f"({result.n_rows_scored} scored predictions)",
        "",
        "## Prediction drift (earliest warning signal)",
        "",
    ]
    pred = result.prediction_drift
    if pred is not None:
        lines.append(
            f"- PSI = {pred.current_stat:.4f} -> **{pred.status}** ({pred.detail})"
        )
    lines += [
        "",
        "## Concept drift proxy (rolling default rate vs. baseline)",
        "",
        f"- baseline rate: {result.concept_drift.baseline_rate:.4f}",
        f"- current rate: {result.concept_drift.current_rate:.4f} "
        f"(n={result.concept_drift.n}, "
        f"{result.concept_drift.ci_low:.4f}-{result.concept_drift.ci_high:.4f} "
        f"95% CI) -> **{result.concept_drift.status}**",
        "",
        "## Per-feature findings",
        "",
        "| feature | method | statistic | status | detail |",
        "|---|---|---:|---|---|",
    ]
    for finding in result.findings:
        if finding.feature_name == PREDICTION_PROBABILITY_FEATURE_NAME:
            continue
        lines.append(
            f"| {finding.feature_name} | {finding.method} | "
            f"{finding.current_stat:.4f} | {finding.status} | {finding.detail} |"
        )
    markdown = "\n".join(lines) + "\n"

    md_path = reports_dir / f"{result.model_id}_{timestamp}.md"
    md_path.write_text(markdown, encoding="utf-8")

    html_rows = "\n".join(
        f"<tr><td>{f.feature_name}</td><td>{f.method}</td>"
        f"<td>{f.current_stat:.4f}</td><td>{f.status}</td><td>{f.detail}</td></tr>"
        for f in result.findings
        if f.feature_name != PREDICTION_PROBABILITY_FEATURE_NAME
    )
    html = f"""<html><head><title>Drift report -- {result.model_id}</title></head>
<body>
<h1>Drift report -- {result.model_id}</h1>
<p>Window: {result.window_start} to {result.window_end}
({result.n_rows_scored} scored predictions)</p>
<h2>Prediction drift</h2>
<p>{"N/A" if pred is None else f"PSI = {pred.current_stat:.4f} -> {pred.status}"}</p>
<h2>Concept drift proxy</h2>
<p>baseline={result.concept_drift.baseline_rate:.4f}, \
current={result.concept_drift.current_rate:.4f}
(n={result.concept_drift.n}) -> {result.concept_drift.status}</p>
<h2>Per-feature findings</h2>
<table border="1" cellpadding="4">
<tr><th>feature</th><th>method</th><th>statistic</th><th>status</th><th>detail</th></tr>
{html_rows}
</table>
</body></html>
"""
    html_path = reports_dir / f"{result.model_id}_{timestamp}.html"
    html_path.write_text(html, encoding="utf-8")

    heatmap_path = _psi_heatmap(result.model_id, reports_dir)
    return md_path, html_path, heatmap_path


def run_drift_check(
    *,
    model_id: str | None = None,
    window_days: int | None = None,
    monitoring_config_path: str = "config/monitoring.yaml",
    reports_dir: Path | None = None,
    as_of: datetime | None = None,
) -> DriftRunResult:
    """Full drift check for one model: fetch production data over the
    trailing `window_days`, compare against that model's persisted
    baseline, persist every finding to `drift_reports`, and write a report.
    """
    config = load_monitoring_config(monitoring_config_path)["drift"]
    as_of = as_of or datetime.now(UTC)
    window_days = window_days if window_days is not None else config["window_days"]
    window_end = as_of.date()
    window_start = window_end - timedelta(days=window_days)

    if model_id is None:
        model_row = registry.get_active_model()
        if model_row is None:
            raise RuntimeError(
                "No active model registered -- nothing to check drift for."
            )
        model_id = model_row["model_id"]

    baseline = load_baseline(model_id)
    baseline_sample = load_baseline_sample(baseline)

    current_frame = build_current_feature_frame(window_start, window_end)
    findings: list[DriftFinding] = []
    if not current_frame.empty:
        findings.extend(
            run_feature_drift(
                baseline,
                baseline_sample,
                current_frame,
                psi_warning=config["psi_warning"],
                psi_drift=config["psi_drift"],
                epsilon=config["epsilon"],
                significance_level=config["significance_level"],
            )
        )

    window_start_dt = datetime.combine(window_start, datetime.min.time(), tzinfo=UTC)
    window_end_dt = datetime.combine(window_end, datetime.max.time(), tzinfo=UTC)
    predictions = fetch_predictions_window(model_id, window_start_dt, window_end_dt)
    if not predictions.empty:
        findings.append(
            run_prediction_drift(
                baseline,
                predictions["default_probability"],
                psi_warning=config["psi_warning"],
                psi_drift=config["psi_drift"],
                epsilon=config["epsilon"],
            )
        )

    production_tables = fetch_production_tables(window_start, window_end)
    matured = production_tables["loan_outcomes"]
    concept_drift = concept_drift_proxy(
        baseline.observed_default_rate,
        matured["default_12m"] if not matured.empty else pd.Series(dtype=int),
        confidence=config["concept_drift_confidence"],
    )

    persist_drift_findings(model_id, findings)

    result = DriftRunResult(
        model_id=model_id,
        window_start=window_start,
        window_end=window_end,
        findings=findings,
        concept_drift=concept_drift,
        n_rows_scored=len(predictions),
    )

    reports_dir = reports_dir or (get_settings().reports_dir / "monitoring" / "drift")
    md_path, html_path, heatmap_path = write_drift_report(result, Path(reports_dir))
    return DriftRunResult(
        **{
            **result.__dict__,
            "report_md_path": md_path,
            "report_html_path": html_path,
            "heatmap_path": heatmap_path,
        }
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: run a drift check for a model (the active model if omitted)."""
    parser = argparse.ArgumentParser(description="CreditGuard Phase 10 drift check.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--window-days", type=int, default=None)
    parser.add_argument("--monitoring-config", default="config/monitoring.yaml")
    args = parser.parse_args(argv)

    result = run_drift_check(
        model_id=args.model_id,
        window_days=args.window_days,
        monitoring_config_path=args.monitoring_config,
    )
    print(
        f"Drift check for {result.model_id}: {len(result.findings)} findings, "
        f"any_drift={result.any_drift}"
    )
    print(f"Report: {result.report_md_path}")


if __name__ == "__main__":
    main()
