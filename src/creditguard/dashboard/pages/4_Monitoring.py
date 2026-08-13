"""Page 4: Monitoring (Phase 10).

Every panel reads back rows `creditguard.monitoring.scheduler`'s periodic
cycle (or the pipeline orchestrator's `--monitor` stage) already wrote, via
`GET /api/v1/monitoring/*` -- this page never computes drift/performance/
data-quality itself, matching the "dashboard only ever talks to the API"
rule the whole Phase 9 brief establishes. An empty result (no monitoring
runs yet for the active model) is shown as an explicit informational state,
not an error.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from creditguard.dashboard.api_client import (
    ApiClientError,
    cached_model_info,
    cached_monitoring_data_quality,
    cached_monitoring_drift,
    cached_monitoring_performance,
)
from creditguard.dashboard.components import cards

st.title("Monitoring")
st.caption(
    "Drift, performance and data-quality monitoring for the active model, "
    "computed by the scheduled monitoring job."
)

try:
    model_info = cached_model_info()
    model_id = model_info["model_id"]
    cards.render_model_caption(model_info["model_id"], model_info["model_version"])
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load model info")
    st.stop()

st.divider()

# -- feature drift status -----------------------------------------------

st.subheader("Feature drift status")
st.caption(
    "Per-feature drift status (OK / WARNING / DRIFT) and PSI values, most recent run."
)

drift_data: dict[str, Any] | None
try:
    drift_data = cached_monitoring_drift(model_id)
except ApiClientError as exc:
    cards.render_api_error(exc, context="Could not load drift status")
    drift_data = None

if drift_data is None:
    pass
elif not drift_data["findings"]:
    st.info(
        "No monitoring runs yet -- start the monitoring container "
        "(`docker compose up monitoring`) or run "
        "`python -m creditguard.monitoring.scheduler --once`.",
        icon="ℹ️",
    )
else:
    status_cols = st.columns(3)
    status_cols[0].metric("OK", drift_data["n_ok"])
    status_cols[1].metric("WARNING", drift_data["n_warning"])
    status_cols[2].metric("DRIFT", drift_data["n_drift"])
    st.caption(f"Latest run: {drift_data['latest_run_at']}")

    findings = pd.DataFrame(drift_data["findings"])
    prediction_findings = findings.loc[
        findings["feature_name"] == "__prediction_probability__"
    ]
    feature_findings = findings.loc[
        findings["feature_name"] != "__prediction_probability__"
    ]

    if not prediction_findings.empty:
        st.subheader("Prediction distribution drift")
        st.caption(
            "Scored-population predicted-probability distribution vs. the "
            "training baseline -- the earliest warning signal."
        )
        row = prediction_findings.iloc[0]
        st.metric("Prediction PSI", f"{row['current_stat']:.4f}", help=row["status"])

    if not feature_findings.empty:

        def _status_color(status: str) -> str:
            return {
                "OK": "background-color:#d4edda",
                "WARNING": "background-color:#fff3cd",
                "DRIFT": "background-color:#f8d7da",
            }.get(status, "")

        display = feature_findings[
            ["feature_name", "method", "current_stat", "status", "detail"]
        ].sort_values(["status", "current_stat"], ascending=[True, False])
        st.dataframe(
            display.style.map(_status_color, subset=["status"]),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

col1, col2 = st.columns(2)

# -- model performance over time -----------------------------------------

with col1:
    st.subheader("Model performance over time")
    st.caption("Rolling predictive/operational metrics on newly-labelled outcomes.")
    perf_data: dict[str, Any] | None
    try:
        perf_data = cached_monitoring_performance(model_id)
    except ApiClientError as exc:
        cards.render_api_error(exc, context="Could not load performance metrics")
        perf_data = None

    if perf_data is not None:
        if not perf_data["metrics"]:
            st.info("No monitoring runs yet.", icon="ℹ️")
        else:
            metrics_df = pd.DataFrame(perf_data["metrics"])
            predictive_names = [
                "roc_auc",
                "pr_auc",
                "ks_statistic",
                "brier_score",
                "precision",
                "recall",
                "calibration_slope",
            ]
            predictive = metrics_df.loc[
                metrics_df["metric_name"].isin(predictive_names)
            ]
            if predictive.empty:
                st.caption("No matured-loan predictive metrics yet (too few outcomes).")
            else:
                pivot = predictive.pivot_table(
                    index="window_end", columns="metric_name", values="metric_value"
                )
                st.line_chart(pivot)
            latest = metrics_df.sort_values("created_at", ascending=False)
            latest_at = latest.iloc[0]["created_at"]
            st.caption(
                f"{len(metrics_df)} metric points recorded, latest at {latest_at}"
            )

# -- data quality trend ---------------------------------------------------

with col2:
    st.subheader("Data quality trend")
    st.caption(
        "Validation rule violation counts over time (Phase 3's rule engine, scheduled)."
    )
    dq_data: dict[str, Any] | None
    try:
        dq_data = cached_monitoring_data_quality()
    except ApiClientError as exc:
        cards.render_api_error(exc, context="Could not load data-quality trend")
        dq_data = None

    if dq_data is not None:
        if not dq_data["trend"]:
            st.info("No monitoring runs yet.", icon="ℹ️")
        else:
            trend_df = pd.DataFrame(dq_data["trend"])
            pivot = trend_df.pivot_table(
                index="day", columns="rule_name", values="n", aggfunc="sum"
            ).fillna(0)
            st.bar_chart(pivot)
            st.caption(f"{dq_data['total_issues']} issues over the trailing window.")
